"""Deterministic, provider-free TypeSafe Jev transport fixture.

This is not a Jev simulator.  It replays pre-declared answers over the exact
documented wire contract (``POST https://api.typesafe.ai/v1/systemone``) so the
adapter's request shape, response validation, deadline and failure paths can be
exercised with no network egress and no provider spend.  A synthetic answer
proves the contract holds; it proves nothing about Jev's accuracy.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence


# Mirrors hermes_lcm.jev_candidates.JEV_ABSTAIN_OPTION.  The test module asserts
# the two stay equal so this fixture cannot silently drift from the adapter.
ABSTAIN_OPTION = "n0"


class SyntheticJevHttpResponse:
    """Minimal stand-in for the plugin's HTTP response value object."""

    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, body: bytes, headers: Mapping[str, str] | None = None):
        self.status = int(status)
        self.headers = dict(headers or {})
        self.body = bytes(body)


class SyntheticJevTransport:
    """Replay one scripted TypeSafe response and record every attempted call.

    ``oracle`` maps a question id (for example ``slot:answer``) to the option id
    the synthetic judge picks.  Any keyword switches one failure mode.
    """

    def __init__(
        self,
        *,
        oracle: Mapping[str, str] | None = None,
        status: int = 200,
        answer_type: str = "choice",
        response_model: str = "jev-1.13.0",
        body: Any = None,
        raw_body: bytes | str | None = None,
        error: BaseException | None = None,
        delay_seconds: float = 0.0,
        probabilities: Mapping[str, float] | None = None,
        extra_answer_fields: Mapping[str, Any] | None = None,
        drop_answers: bool = False,
        extra_answer_key: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        self.oracle = dict(oracle or {})
        self.status = int(status)
        self.answer_type = str(answer_type)
        self.response_model = str(response_model)
        self.body = body
        self.raw_body = raw_body
        self.error = error
        self.delay_seconds = float(delay_seconds)
        self.probabilities = dict(probabilities or {})
        self.extra_answer_fields = dict(extra_answer_fields or {})
        self.drop_answers = bool(drop_answers)
        self.extra_answer_key = extra_answer_key
        self.usage = dict(usage or {"input_tokens": 111, "output_tokens": 7})
        # Recorded evidence: payload bytes and header NAMES only.  Credential
        # values are never retained by this fixture.
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> SyntheticJevHttpResponse:
        return self.post(url=url, payload=payload, headers=headers, timeout=timeout)

    def post(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> SyntheticJevHttpResponse:
        self.calls.append(
            {
                "url": str(url),
                "payload": json.loads(json.dumps(payload)),
                "header_names": sorted(str(name) for name in headers),
                "has_authorization": bool(headers.get("Authorization")),
                "timeout": float(timeout),
            }
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        if self.raw_body is not None:
            raw = self.raw_body.encode("utf-8") if isinstance(self.raw_body, str) else self.raw_body
            return SyntheticJevHttpResponse(self.status, raw, {"Content-Type": "application/json"})
        if self.body is not None:
            return SyntheticJevHttpResponse(
                self.status, json.dumps(self.body).encode("utf-8"), {"Content-Type": "application/json"}
            )
        return SyntheticJevHttpResponse(
            self.status,
            json.dumps(self._default_body(payload)).encode("utf-8"),
            {"Content-Type": "application/json"},
        )

    def _default_body(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        questions = payload.get("questions") if isinstance(payload, Mapping) else None
        if not self.drop_answers and isinstance(questions, Mapping):
            for question_id, question in questions.items():
                criteria = question.get("criteria") if isinstance(question, Mapping) else None
                keys = list(criteria) if isinstance(criteria, Mapping) else []
                candidates = [key for key in keys if key != ABSTAIN_OPTION]
                default = candidates[0] if candidates else (keys[0] if keys else "")
                chosen = self.oracle.get(str(question_id), default)
                if self.answer_type == "choice":
                    answer: dict[str, Any] = {"type": "choice", "choice": chosen}
                    if self.probabilities:
                        answer["probabilities"] = dict(self.probabilities)
                    answer.update(self.extra_answer_fields)
                else:
                    answer = {"type": self.answer_type, "noul": 0.5}
                    answer.update(self.extra_answer_fields)
                answers[str(question_id)] = answer
        if self.extra_answer_key:
            answers.setdefault(self.extra_answer_key, {"type": "choice", "choice": "c01"})
        return {"model": self.response_model, "answers": answers, "usage": dict(self.usage)}


def question_ids(payload: Mapping[str, Any]) -> Sequence[str]:
    """Sorted question ids present in a recorded request payload."""
    questions = payload.get("questions")
    if not isinstance(questions, Mapping):
        return ()
    return tuple(sorted(str(key) for key in questions))
