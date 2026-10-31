"""Configuration, Nimble criteria loader, and Google Calendar color mappings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "faith-template.json"

DEFAULT_CRITERIA: Dict[str, str] = {
    "family": "Wife Jackie, kids, grandkids, in-laws",
    "work": "AI Development, Team Oversight",
    "leisure": "Live music, football game, relax with friends",
    "faith": "Church on Sundays, teaching Sunday school, church community group",
}


@dataclass(frozen=True)
class CategoryColor:
    """Google Calendar color metadata for a Nimble category."""

    category: str
    color_id: str
    color_name: str
    hex_code: str


# Google Calendar Event colorId mapping (1-11 standard palette):
# 7 = Peacock (Blue), 8 = Graphite (Dark Gray), 2 = Sage (Green), 3 = Grape (Purple)
CATEGORY_COLORS: Dict[str, CategoryColor] = {
    "family": CategoryColor(
        category="family",
        color_id="7",
        color_name="Peacock (Blue)",
        hex_code="#039be5",
    ),
    "work": CategoryColor(
        category="work",
        color_id="8",
        color_name="Graphite (Gray)",
        hex_code="#616161",
    ),
    "leisure": CategoryColor(
        category="leisure",
        color_id="2",
        color_name="Sage (Green)",
        hex_code="#33b679",
    ),
    "faith": CategoryColor(
        category="faith",
        color_id="3",
        color_name="Grape (Purple)",
        hex_code="#8e24aa",
    ),
}


def load_criteria(template_path: Path | None = None) -> Dict[str, str]:
    """Load the 4 category criteria from the repo's Nimble JSON template file."""
    path = template_path or DEFAULT_TEMPLATE_PATH
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            criteria = (
                data.get("questions", {})
                .get("label", {})
                .get("criteria", {})
            )
            if isinstance(criteria, dict) and criteria:
                return {str(k): str(v) for k, v in criteria.items()}
        except (OSError, ValueError):
            pass
    return dict(DEFAULT_CRITERIA)


@dataclass
class AgentSettings:
    """Runtime configuration for the Calendar Organizer Agent."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    ollama_systemone_url: str = os.getenv(
        "OLLAMA_SYSTEMONE_URL", "http://localhost:11434/v1/systemone"
    )
    nimble_model: str = os.getenv("NIMBLE_MODEL", "nimble")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    google_calendar_id: str = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    google_credentials_file: Path = Path(
        os.getenv("GOOGLE_CREDENTIALS_FILE", str(REPO_ROOT / "credentials.json"))
    )
    google_token_file: Path = Path(
        os.getenv("GOOGLE_TOKEN_FILE", str(REPO_ROOT / "token.json"))
    )
