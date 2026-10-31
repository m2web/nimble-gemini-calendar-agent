"""Gemini System 2 Organizer Agent orchestrating local Ollama `nimble` (System 1) and Google Calendar."""

import json
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .config import CATEGORY_COLORS, AgentSettings, load_criteria
from .gcal_client import CalendarEventPayload, GoogleCalendarClient
from .nimble_classifier import NimbleClassification, classify_event_state


def format_probability_bars(probabilities: Dict[str, float]) -> str:
    """Format the 4 category probabilities as readable ASCII bars."""
    lines: List[str] = []
    sorted_items = sorted(
        probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    for cat, prob in sorted_items:
        color_info = CATEGORY_COLORS.get(cat)
        color_label = (
            f"{color_info.color_name} [colorId={color_info.color_id}]"
            if color_info
            else "Default"
        )
        filled = int(round(prob * 20))
        bar = "#" * filled + "-" * (20 - filled)
        lines.append(f"  - {cat:<8} [{bar}] {prob:6.2%}  -> {color_label}")
    return "\n".join(lines)


class CalendarOrganizerAgent:
    """Hybrid System 2 (Gemini) + System 1 (Ollama `nimble`) Calendar Organizer."""

    def __init__(
        self,
        settings: Optional[AgentSettings] = None,
        dry_run: bool = False,
        save_audit_file: bool = True,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.settings = settings or AgentSettings()
        self.dry_run = dry_run
        self.save_audit_file = save_audit_file
        self.interactive = interactive
        self.input_fn = input_fn
        self.gcal = GoogleCalendarClient(
            settings=self.settings, dry_run=self.dry_run
        )
        self._genai_client = None

    def _get_genai_client(self):
        """Initialize and return the `google.genai.Client` instance."""
        if self._genai_client is not None:
            return self._genai_client

        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add GEMINI_API_KEY to your .env file "
                "or environment variables to run the Gemini System 2 Organizer."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai SDK is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._genai_client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._genai_client

    def resolve_low_confidence_with_gemini(
        self,
        event_text: str,
        classification: NimbleClassification,
    ) -> NimbleClassification:
        """Ask Gemini to review ambiguous probabilities and prompt the user in CLI."""
        print("\n" + "=" * 72)
        print(
            f"[!] LOW CONFIDENCE DETECTED ({classification.confidence:.2%} < "
            f"{self.settings.confidence_threshold:.2%})"
        )
        print(f"    Event Text : {event_text}")
        print(f"    Top Choice : {classification.choice}")
        print("    Simplex    :")
        print(format_probability_bars(classification.probabilities))

        gemini_advice = ""
        if self.settings.gemini_api_key:
            try:
                client = self._get_genai_client()
                criteria = load_criteria()
                prompt = (
                    "You are a calendar triage assistant. The local System 1 classifier "
                    "(`nimble`) returned a low-confidence split across categories for "
                    "this calendar entry.\n\n"
                    f"Calendar entry: {event_text!r}\n"
                    f"Criteria definitions: {json.dumps(criteria)}\n"
                    f"Nimble probabilities: {json.dumps(classification.probabilities)}\n"
                    f"Nimble confidence: {classification.confidence:.4f}\n\n"
                    "In 1-2 short sentences, explain why this entry straddles multiple "
                    "categories and state which category you recommend."
                )
                response = client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt,
                )
                gemini_advice = (response.text or "").strip()
            except Exception as exc:
                gemini_advice = f"(Could not fetch Gemini advice: {exc})"

        if gemini_advice:
            print(f"\n[Gemini System 2 Review]\n  {gemini_advice}")

        if not self.interactive or not sys.stdin.isatty():
            print(
                f"\n[Non-interactive mode] Keeping top probability choice: "
                f"'{classification.choice}' ({classification.color_name})"
            )
            print("=" * 72 + "\n")
            return classification

        categories = list(CATEGORY_COLORS.keys())
        print("\nChoose category color for this event:")
        for idx, cat in enumerate(categories, start=1):
            c_meta = CATEGORY_COLORS[cat]
            prob = classification.probabilities.get(cat, 0.0)
            default_marker = (
                " (default)" if cat == classification.choice else ""
            )
            print(
                f"  [{idx}] {cat:<8} - {c_meta.color_name:<16} "
                f"(colorId={c_meta.color_id}, p={prob:.2%}){default_marker}"
            )

        raw_choice = self.input_fn(
            f"Select [1-{len(categories)}] or press Enter for '{classification.choice}': "
        ).strip()

        selected_cat = classification.choice
        if raw_choice.isdigit():
            idx = int(raw_choice) - 1
            if 0 <= idx < len(categories):
                selected_cat = categories[idx]
        elif raw_choice.lower() in CATEGORY_COLORS:
            selected_cat = raw_choice.lower()

        chosen_color = CATEGORY_COLORS[selected_cat]
        print(
            f"-> Selected '{selected_cat}' -> {chosen_color.color_name} "
            f"(colorId={chosen_color.color_id})"
        )
        print("=" * 72 + "\n")

        return NimbleClassification(
            state=classification.state,
            choice=selected_cat,
            confidence=classification.confidence,
            probabilities=classification.probabilities,
            color_id=chosen_color.color_id,
            color_name=chosen_color.color_name,
            is_high_confidence=True,
            input_tokens=classification.input_tokens,
            output_tokens=classification.output_tokens,
            audit_file=classification.audit_file,
        )

    def classify_with_guardrail(self, event_text: str) -> NimbleClassification:
        """Run System 1 `nimble` classification and trigger System 2 review if needed."""
        result = classify_event_state(
            state=event_text,
            settings=self.settings,
            save_audit_file=self.save_audit_file,
        )
        if not result.is_high_confidence:
            result = self.resolve_low_confidence_with_gemini(event_text, result)
        return result

    def add_event_from_text(self, raw_entry: str) -> Dict[str, Any]:
        """Use Gemini (System 2) + Local `nimble` (System 1) to parse, classify, and schedule."""
        from google.genai import types

        client = self._get_genai_client()
        now_local = datetime.now().astimezone()
        tz_name = str(now_local.tzinfo) if now_local.tzinfo else "America/New_York"

        classifications: List[NimbleClassification] = []
        created_events: List[Dict[str, Any]] = []

        def classify_calendar_event(event_description: str) -> Dict[str, Any]:
            """Classify a calendar entry into family, work, leisure, or faith using local Ollama nimble (System 1).

            Args:
                event_description: The text or summary of the calendar event to classify.
            """
            result = self.classify_with_guardrail(event_description)
            classifications.append(result)
            return result.to_dict()

        def create_color_coded_calendar_event(
            summary: str,
            start_iso: str,
            end_iso: str,
            category: str,
            color_id: str,
            color_name: str,
            confidence: float,
            description: str = "",
            location: str = "",
            timezone: str = "America/New_York",
        ) -> Dict[str, Any]:
            """Create the color-coded event on Google Calendar.

            Args:
                summary: Short title of the calendar event.
                start_iso: ISO-8601 start datetime (e.g. '2026-11-01T09:00:00-04:00').
                end_iso: ISO-8601 end datetime (e.g. '2026-11-01T10:00:00-04:00').
                category: Winning Nimble category ('family', 'work', 'leisure', or 'faith').
                color_id: Google Calendar colorId ('7', '8', '2', or '3').
                color_name: Human-readable color name (e.g. 'Grape (Purple)').
                confidence: Confidence score from Nimble (0.0 to 1.0).
                description: Optional notes or context about the event.
                location: Optional physical or virtual location.
                timezone: IANA timezone name (defaults to 'America/New_York').
            """
            payload = CalendarEventPayload(
                summary=summary,
                start_iso=start_iso,
                end_iso=end_iso,
                category=category,
                color_id=color_id,
                color_name=color_name,
                confidence=confidence,
                description=description,
                location=location,
                timezone=timezone,
            )
            outcome = self.gcal.create_event(payload)
            created_events.append(outcome)
            return outcome

        system_instruction = (
            "You are the System 2 Calendar Organizer Agent. "
            f"The current local datetime is {now_local.isoformat()} ({tz_name}).\n"
            "When the user gives you a written calendar entry:\n"
            "1. ALWAYS call `classify_calendar_event` first with the user's event text "
            "so the local Ollama `nimble` System 1 model determines the category "
            "(`family`, `work`, `leisure`, or `faith`), `color_id`, `color_name`, and `confidence`.\n"
            "2. Parse the date, start time, and end time from the user's text relative to "
            f"{now_local.isoformat()}. If no duration or end time is stated, default to 1 hour. "
            "If no specific time of day is stated, default to 09:00 local time.\n"
            "3. Call `create_color_coded_calendar_event` using the exact `category`, "
            "`color_id`, `color_name`, and `confidence` returned by `classify_calendar_event`.\n"
            "4. Summarize the scheduled event, its assigned category, color, confidence, "
            "and zero-output-token System 1 telemetry."
        )

        chat = client.chats.create(
            model=self.settings.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[
                    classify_calendar_event,
                    create_color_coded_calendar_event,
                ],
                temperature=0.1,
            ),
        )
        response = chat.send_message(raw_entry)

        return {
            "raw_entry": raw_entry,
            "agent_summary": (response.text or "").strip(),
            "classifications": [c.to_dict() for c in classifications],
            "calendar_results": created_events,
        }

    def triage_uncolored_events(self, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Fetch upcoming uncolored events, classify each with `nimble`, and apply colors."""
        events = self.gcal.list_uncolored_events(days_ahead=days_ahead)
        results: List[Dict[str, Any]] = []

        for evt in events:
            event_id = str(evt.get("id", ""))
            summary = str(evt.get("summary", "(No title)"))
            description = str(evt.get("description", ""))
            combined_state = (
                f"{summary}. {description}".strip()
                if description
                else summary
            )

            classification = self.classify_with_guardrail(combined_state)
            update_result = self.gcal.update_event_color(
                event_id=event_id,
                category=classification.choice,
                color_id=classification.color_id,
                color_name=classification.color_name,
                confidence=classification.confidence,
            )
            results.append(
                {
                    "event_id": event_id,
                    "summary": summary,
                    "classification": classification.to_dict(),
                    "calendar_update": update_result,
                }
            )

        return results
