"""Google Calendar API client with Desktop OAuth2 and `--dry-run` support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .config import CATEGORY_COLORS, AgentSettings

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


@dataclass
class CalendarEventPayload:
    """Structured event payload ready for Google Calendar insertion or update."""

    summary: str
    start_iso: str
    end_iso: str
    category: str
    color_id: str
    color_name: str
    confidence: float
    description: str = ""
    location: str = ""
    timezone: str = "America/New_York"

    def to_google_body(self) -> Dict[str, Any]:
        """Convert to a Google Calendar `events.insert` request body."""
        tag_note = (
            f"[System-1 Nimble: category={self.category}, "
            f"color={self.color_name} (id={self.color_id}), "
            f"confidence={self.confidence:.2%}]"
        )
        full_description = (
            f"{self.description}\n\n{tag_note}".strip()
            if self.description
            else tag_note
        )
        body: Dict[str, Any] = {
            "summary": self.summary,
            "description": full_description,
            "colorId": self.color_id,
            "start": {
                "dateTime": self.start_iso,
                "timeZone": self.timezone,
            },
            "end": {
                "dateTime": self.end_iso,
                "timeZone": self.timezone,
            },
        }
        if self.location:
            body["location"] = self.location
        return body

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoogleCalendarClient:
    """Wrapper around Google Calendar v3 API with `--dry-run` mode."""

    def __init__(
        self,
        settings: AgentSettings | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings or AgentSettings()
        self.dry_run = dry_run
        self._service = None

    def _get_service(self):
        """Authenticate via Desktop OAuth2 and return the Calendar v3 service."""
        if self._service is not None:
            return self._service

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "Google Calendar dependencies are missing. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        creds = None
        token_path = self.settings.google_token_file
        creds_path = self.settings.google_credentials_file

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    raise FileNotFoundError(
                        f"Google OAuth credentials file not found at '{creds_path}'. "
                        "Download your Desktop OAuth 2.0 Client JSON from Google Cloud "
                        "Console and save it as 'credentials.json', or pass --dry-run."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), SCOPES
                )
                print(
                    "[Google Calendar OAuth] Starting local auth server and opening browser...",
                    flush=True,
                )
                creds = flow.run_local_server(port=0, open_browser=True)

            token_path.write_text(creds.to_json(), encoding="utf-8")

        self._service = build("calendar", "v3", credentials=creds)
        return self._service

    def create_event(self, event: CalendarEventPayload) -> Dict[str, Any]:
        """Insert a new color-coded event into Google Calendar (or preview in dry-run)."""
        body = event.to_google_body()
        if self.dry_run:
            return {
                "dry_run": True,
                "calendar_id": self.settings.google_calendar_id,
                "action": "events.insert",
                "event_body": body,
            }

        service = self._get_service()
        created = (
            service.events()
            .insert(calendarId=self.settings.google_calendar_id, body=body)
            .execute(num_retries=3)
        )
        return {
            "dry_run": False,
            "id": created.get("id"),
            "htmlLink": created.get("htmlLink"),
            "summary": created.get("summary"),
            "colorId": created.get("colorId"),
            "start": created.get("start"),
            "end": created.get("end"),
        }

    def list_uncolored_events(self, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Return upcoming events that do not have one of the 4 category colorIds set."""
        managed_color_ids = {c.color_id for c in CATEGORY_COLORS.values()}

        if self.dry_run and not self.settings.google_token_file.exists():
            now = datetime.now(timezone.utc).astimezone()
            return [
                {
                    "id": "dryrun-evt-1",
                    "summary": "My wife Jackie wants us to go see the grandkids.",
                    "description": "Weekend family visit",
                    "start": {"dateTime": (now + timedelta(days=1)).isoformat()},
                    "end": {
                        "dateTime": (now + timedelta(days=1, hours=2)).isoformat()
                    },
                },
                {
                    "id": "dryrun-evt-2",
                    "summary": "Reviewing AI development roadmap and team oversight sprint tasks.",
                    "description": "Q4 engineering planning",
                    "start": {"dateTime": (now + timedelta(days=2)).isoformat()},
                    "end": {
                        "dateTime": (now + timedelta(days=2, hours=1)).isoformat()
                    },
                },
                {
                    "id": "dryrun-evt-3",
                    "summary": "Heading to a live music show and relaxing with friends this weekend.",
                    "description": "Saturday night concert",
                    "start": {"dateTime": (now + timedelta(days=3)).isoformat()},
                    "end": {
                        "dateTime": (now + timedelta(days=3, hours=3)).isoformat()
                    },
                },
                {
                    "id": "dryrun-evt-4",
                    "summary": "Preparing my Sunday school lesson and meeting with our church community group.",
                    "description": "Sunday morning prep",
                    "start": {"dateTime": (now + timedelta(days=4)).isoformat()},
                    "end": {
                        "dateTime": (now + timedelta(days=4, hours=2)).isoformat()
                    },
                },
            ]

        service = self._get_service()
        now_utc = datetime.now(timezone.utc)
        time_min = now_utc.isoformat()
        time_max = (now_utc + timedelta(days=days_ahead)).isoformat()

        events_result = (
            service.events()
            .list(
                calendarId=self.settings.google_calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute(num_retries=3)
        )
        items = events_result.get("items", [])
        return [
            item
            for item in items
            if str(item.get("colorId", "")) not in managed_color_ids
        ]

    def update_event_color(
        self,
        event_id: str,
        category: str,
        color_id: str,
        color_name: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Patch an existing Google Calendar event's `colorId`."""
        patch_body = {"colorId": color_id}
        if self.dry_run:
            return {
                "dry_run": True,
                "calendar_id": self.settings.google_calendar_id,
                "action": "events.patch",
                "event_id": event_id,
                "category": category,
                "color_id": color_id,
                "color_name": color_name,
                "confidence": confidence,
                "patch_body": patch_body,
            }

        service = self._get_service()
        updated = (
            service.events()
            .patch(
                calendarId=self.settings.google_calendar_id,
                eventId=event_id,
                body=patch_body,
            )
            .execute(num_retries=3)
        )
        return {
            "dry_run": False,
            "id": updated.get("id"),
            "summary": updated.get("summary"),
            "colorId": updated.get("colorId"),
            "htmlLink": updated.get("htmlLink"),
        }
