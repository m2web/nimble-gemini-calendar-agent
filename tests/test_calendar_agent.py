"""Unit and wire-contract verification tests for `calendar_agent`."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from calendar_agent.config import CATEGORY_COLORS, REPO_ROOT, AgentSettings, load_criteria
from calendar_agent.gcal_client import CalendarEventPayload, GoogleCalendarClient
from calendar_agent.nimble_classifier import (
    NimbleClassification,
    build_systemone_payload,
    classify_event_state,
)
from calendar_agent.organizer_agent import CalendarOrganizerAgent


class TestCalendarAgent(unittest.TestCase):
    def test_load_criteria_from_repo_template(self) -> None:
        criteria = load_criteria()
        self.assertEqual(
            set(criteria.keys()), {"family", "work", "leisure", "faith"}
        )
        self.assertIn("Jackie", criteria["family"])
        self.assertIn("Sunday school", criteria["faith"])

    def test_category_color_mappings(self) -> None:
        self.assertEqual(CATEGORY_COLORS["family"].color_id, "7")
        self.assertEqual(CATEGORY_COLORS["work"].color_id, "8")
        self.assertEqual(CATEGORY_COLORS["leisure"].color_id, "2")
        self.assertEqual(CATEGORY_COLORS["faith"].color_id, "3")

    def test_payload_matches_template_contract_and_64k_limit(self) -> None:
        payload = build_systemone_payload(
            state="Preparing my Sunday school lesson and meeting with our church community group."
        )
        expected = json.loads(
            (REPO_ROOT / "faith-template.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload, expected)

        with self.assertRaises(ValueError):
            classify_event_state("x" * 70000)

    def test_classify_all_four_repo_responses(self) -> None:
        cases = [
            ("fam-response.json", "family", "7"),
            ("work-response.json", "work", "8"),
            ("leisure-response.json", "leisure", "2"),
            ("faith-response.json", "faith", "3"),
        ]
        for filename, expected_choice, expected_color_id in cases:
            raw_bytes = (REPO_ROOT / filename).read_bytes()
            mock_resp = MagicMock()
            mock_resp.read.return_value = raw_bytes
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = classify_event_state(
                    state=f"Test state for {expected_choice}",
                    save_audit_file=False,
                )
                self.assertEqual(result.choice, expected_choice)
                self.assertEqual(result.color_id, expected_color_id)
                self.assertTrue(result.is_high_confidence)
                self.assertEqual(result.output_tokens, 0)

    def test_low_confidence_interactive_resolution(self) -> None:
        low_conf = NimbleClassification(
            state="Church picnic with Jackie and the grandkids",
            choice="faith",
            confidence=0.52,
            probabilities={
                "faith": 0.51,
                "family": 0.47,
                "leisure": 0.015,
                "work": 0.005,
            },
            color_id="3",
            color_name="Grape (Purple)",
            is_high_confidence=False,
            input_tokens=232,
            output_tokens=0,
        )
        settings = AgentSettings(gemini_api_key="")
        # Simulate user choosing option "1" (family -> colorId "7")
        agent = CalendarOrganizerAgent(
            settings=settings,
            dry_run=True,
            save_audit_file=False,
            interactive=True,
            input_fn=lambda _: "1",
        )
        with patch("sys.stdin.isatty", return_value=True):
            resolved = agent.resolve_low_confidence_with_gemini(
                low_conf.state, low_conf
            )
        self.assertEqual(resolved.choice, "family")
        self.assertEqual(resolved.color_id, "7")
        self.assertTrue(resolved.is_high_confidence)

    def test_dry_run_calendar_event_creation(self) -> None:
        client = GoogleCalendarClient(dry_run=True)
        payload = CalendarEventPayload(
            summary="Sunday School Prep",
            start_iso="2026-11-01T09:00:00-04:00",
            end_iso="2026-11-01T10:00:00-04:00",
            category="faith",
            color_id="3",
            color_name="Grape (Purple)",
            confidence=0.9966,
            description="Lesson notes",
        )
        res = client.create_event(payload)
        self.assertTrue(res["dry_run"])
        self.assertEqual(res["event_body"]["colorId"], "3")
        self.assertIn("System-1 Nimble: category=faith", res["event_body"]["description"])


if __name__ == "__main__":
    unittest.main()
