"""Command-line interface for the Gemini + Ollama `nimble` Calendar Organizer Agent."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import AgentSettings
from .organizer_agent import CalendarOrganizerAgent, format_probability_bars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calendar_agent",
        description=(
            "Hybrid System 2 (Gemini) + System 1 (Local Ollama nimble) "
            "Google Calendar Organizer and Color-Coder."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `add` subcommand
    add_parser = subparsers.add_parser(
        "add",
        help="Parse a written calendar entry with Gemini, classify with Nimble, and add to Google Calendar.",
    )
    add_parser.add_argument(
        "entry",
        type=str,
        help="Natural-language calendar note (e.g., 'Dinner with Jackie and the grandkids Friday at 6pm').",
    )
    add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the Google Calendar API payload without writing to Google Calendar.",
    )
    add_parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not write a timestamped response-<choice>-<timestamp>.json file.",
    )
    add_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-select top probability without interactive CLI prompt when confidence < threshold.",
    )

    # 2. `triage` subcommand
    triage_parser = subparsers.add_parser(
        "triage",
        help="Fetch upcoming uncolored Google Calendar events and color-code them with Nimble.",
    )
    triage_parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of days ahead to scan for uncolored events (default: 14).",
    )
    triage_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview color updates (uses sample events if Google Calendar token.json is absent).",
    )
    triage_parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not write timestamped response-<choice>-<timestamp>.json files.",
    )
    triage_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-select top probability without interactive CLI prompt when confidence < threshold.",
    )

    # 3. `classify` subcommand
    classify_parser = subparsers.add_parser(
        "classify",
        help="Run local Ollama nimble (/v1/systemone) + color mapping on a single event string.",
    )
    classify_parser.add_argument(
        "state",
        type=str,
        help="Event description to classify into family, work, leisure, or faith.",
    )
    classify_parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not write a timestamped response-<choice>-<timestamp>.json file.",
    )
    classify_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-select top probability without interactive CLI prompt when confidence < threshold.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = AgentSettings()

    if args.command == "classify":
        agent = CalendarOrganizerAgent(
            settings=settings,
            dry_run=True,
            save_audit_file=not args.no_audit,
            interactive=not args.non_interactive,
        )
        result = agent.classify_with_guardrail(args.state)
        print("\n=== System 1 (Ollama `nimble`) Classification ===")
        print(f"Input State   : {result.state}")
        print(f"Winner Choice : {result.choice}")
        print(
            f"Assigned Color: {result.color_name} (Google Calendar colorId={result.color_id})"
        )
        print(f"Confidence    : {result.confidence:.2%}")
        print(
            f"Token Usage   : {result.input_tokens} in / {result.output_tokens} out"
        )
        print("Probabilities :")
        print(format_probability_bars(result.probabilities))
        if result.audit_file:
            print(f"Audit File    : {result.audit_file}")
        return 0

    if args.command == "add":
        agent = CalendarOrganizerAgent(
            settings=settings,
            dry_run=args.dry_run,
            save_audit_file=not args.no_audit,
            interactive=not args.non_interactive,
        )
        outcome = agent.add_event_from_text(args.entry)
        print("\n=== System 2 (Gemini) + System 1 (Nimble) Result ===")
        if outcome["agent_summary"]:
            print(outcome["agent_summary"])
        print("\n--- Calendar Wire Output ---")
        print(json.dumps(outcome["calendar_results"], indent=2))
        return 0

    if args.command == "triage":
        agent = CalendarOrganizerAgent(
            settings=settings,
            dry_run=args.dry_run,
            save_audit_file=not args.no_audit,
            interactive=not args.non_interactive,
        )
        results = agent.triage_uncolored_events(days_ahead=args.days)
        print(f"\n=== Triaged {len(results)} Uncolored Calendar Event(s) ===")
        for item in results:
            cls_info = item["classification"]
            print(
                f"- {item['summary']!r}\n"
                f"    -> Category  : {cls_info['choice']} "
                f"({cls_info['color_name']}, colorId={cls_info['color_id']})\n"
                f"    -> Confidence: {cls_info['confidence']:.2%} "
                f"(tokens: {cls_info['input_tokens']} in / {cls_info['output_tokens']} out)"
            )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
