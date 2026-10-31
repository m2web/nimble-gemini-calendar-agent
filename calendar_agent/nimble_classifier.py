"""System 1 local classifier client for Ollama /v1/systemone (`nimble`)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from .config import (
    CATEGORY_COLORS,
    REPO_ROOT,
    AgentSettings,
    CategoryColor,
    load_criteria,
)

MAX_SYSTEMONE_PAYLOAD_BYTES = 65536  # 64 KiB strict ceiling


@dataclass
class NimbleClassification:
    """Structured result returned by the local Ollama `/v1/systemone` endpoint."""

    state: str
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    color_id: str
    color_name: str
    is_high_confidence: bool
    input_tokens: int
    output_tokens: int
    audit_file: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_systemone_payload(
    state: str,
    model: str = "nimble",
    criteria: Dict[str, str] | None = None,
    instructions: str = "Which type of calendar event?",
) -> dict:
    """Build the `/v1/systemone` wire payload matching the repo's JSON templates."""
    resolved_criteria = criteria or load_criteria()
    return {
        "model": model,
        "state": state,
        "questions": {
            "label": {
                "type": "choice",
                "instructions": instructions,
                "criteria": resolved_criteria,
            }
        },
    }


def classify_event_state(
    state: str,
    settings: AgentSettings | None = None,
    save_audit_file: bool = False,
    timeout_sec: int = 120,
) -> NimbleClassification:
    """Evaluate a calendar event string using local Ollama `/v1/systemone` (`nimble`).

    Enforces the 64 KiB UTF-8 byte ceiling before transmitting over HTTP.
    """
    cfg = settings or AgentSettings()
    payload = build_systemone_payload(state=state, model=cfg.nimble_model)

    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    utf8_bytes = compact_json.encode("utf-8")

    if len(utf8_bytes) > MAX_SYSTEMONE_PAYLOAD_BYTES:
        raise ValueError(
            f"Payload exceeds 64 KiB text/schema limit ({len(utf8_bytes)} bytes)."
        )

    req = urllib.request.Request(
        url=cfg.ollama_systemone_url,
        data=utf8_bytes,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_body = resp.read().decode("utf-8")
            response_data = json.loads(raw_body)
    except urllib.error.HTTPError as exc:
        err_detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"System One API call failed (HTTP {exc.code}): {err_detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach local Ollama System One endpoint at "
            f"{cfg.ollama_systemone_url}: {exc.reason}"
        ) from exc

    label_answer = response_data.get("answers", {}).get("label", {})
    choice = str(label_answer.get("choice", "")).strip()
    probabilities = {
        str(k): float(v)
        for k, v in (label_answer.get("probabilities") or {}).items()
    }
    confidence = float(label_answer.get("confidence", 0.0))

    usage = response_data.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))

    color_meta: CategoryColor | None = CATEGORY_COLORS.get(choice)
    color_id = color_meta.color_id if color_meta else "8"
    color_name = color_meta.color_name if color_meta else "Graphite (Default)"

    audit_path_str: str | None = None
    if save_audit_file and choice:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        audit_path: Path = REPO_ROOT / f"response-{choice}-{timestamp}.json"
        audit_path.write_text(
            json.dumps(response_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        audit_path_str = str(audit_path)

    return NimbleClassification(
        state=state,
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
        color_id=color_id,
        color_name=color_name,
        is_high_confidence=confidence >= cfg.confidence_threshold,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        audit_file=audit_path_str,
    )
