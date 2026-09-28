"""Optional, default-off TypeSafe Jev rerank and advisory ordering.

Catalog experiment ``R20`` (``requirements_compiler.py`` candidate seam) is
*"Existing deterministic numeric/text/date candidates and contract -> Score
candidates within each slot; product revalidates"*.

This module is the bounded adapter for that seam.  It never authors evidence:
the wire request carries code-owned option ids (``c01``...) plus the product's
own candidate text, and the only value read back from the provider is one of
those option ids.  Every evidence field (exact ref, span, quote, value, unit,
key, date) is taken from the product's candidate object, so an answer cannot
inject a ref, a quote or a value that the deterministic pipeline did not
already construct.

Four modes, resolved in :func:`resolve_jev_options`:

``off``
    Default.  Zero transport calls and no result mutation.  An unknown mode also
    resolves here.
``shadow``
    Measure only.  The provider choice is recorded; nothing is delivered.
``applied``
    Narrowing, retained unchanged from the reviewed adapter.  Promotion still
    requires :func:`semantic_uniqueness_proof`, so an ambiguous pool -- the only
    pool a rerank can act on -- is refused before dispatch and can never
    manufacture the uniqueness the deterministic pipeline declined to assert.
``advisory``
    Presentation ordering only.  The provider choice moves its candidate to the
    front of the delivered advisory block; *every* admitted candidate stays in
    the block, in the same deterministic pool the selectors saw.  Ordering is
    never narrowing: no ref is dropped, no quote is replaced by provider prose,
    no answer sufficiency is declared, no requirement is closed, no value is
    computed and no coverage is established.  The deterministic verdict, state
    and reason code are untouched, and competing candidates stay explicitly
    labelled unresolved.

Fixed boundaries:

* **Default off.**  ``resolve_jev_options`` returns ``enabled=False`` unless
  ``LCM_JEV_CANDIDATE_RERANK_ENABLED`` is true, and an unknown mode resolves to
  ``off``.  Disabled means zero transport calls and no result mutation.
* **Code-owned scope.**  Only the current question, the code-derived answer
  contract and the bounded candidate windows leave the process.  Ref shapes that
  are not ``lcm:<store>:<start>-<end>``, empty quotes, secret-like content and
  provider credentials never leave; the credential is read from
  ``TYPESAFE_API_KEY`` at call time and is never stored in a record.
* **Code-owned budgets.**  Slot count, candidates per slot, characters per
  candidate, whole-state characters, the advisory pool size, one call and the
  wall-clock deadline are clamped in :meth:`JevLimits.from_mapping`.
* **Verified wire option index.**  The advisory resolves a returned option id
  through the exact ``option_index`` of the request that was sent, never through
  a numbering recomputed over a different pool.  See
  :func:`resolve_advisory_choice`.
* **Baseline retained on every failure.**  Timeout, HTTP error, malformed body,
  unexpected answer type, unknown option id, answer-set mismatch and explicit
  abstention all return a record the caller treats as "no change".  Unknown cost
  stays ``None``, never zero.
* **Product revalidates.**  Every emitted advisory ref, span and quote is
  revalidated against the source through the caller's own normalizer before the
  request is built, and only revalidated entries are rendered.
* **Refuse before paying.**  An incomplete, clipped, oversized or
  over-budget advisory pool is refused *before* dispatch, because the delivered
  block could not have been complete either way.
* **No manufactured uniqueness.**  :func:`semantic_uniqueness_proof` and the
  ``applied`` preflight are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

JEV_RERANK_VERSION = "jev-candidate-rerank-v1"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_DEFAULT_MODEL = "jev-1.13.0"
JEV_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_ABSTAIN_OPTION = "n0"
JEV_ADVISORY_MODE = "advisory"
JEV_MODES = frozenset({"off", "shadow", "applied", JEV_ADVISORY_MODE})

#: Envelope tag of the delivered advisory block.  The block carries only
#: product-owned refs and quotes; no provider-authored text is ever rendered.
JEV_ADVISORY_TAG = "lcm-advisory-order"

#: Hard wire bounds.  A response body beyond ``JEV_MAX_RESPONSE_BYTES`` and an
#: HTTP error body beyond ``JEV_MAX_ERROR_BODY_BYTES`` fail the attempt closed
#: instead of being read into memory unbounded.
JEV_MAX_RESPONSE_BYTES = 65_536
JEV_MAX_ERROR_BODY_BYTES = 8_192
#: Name of the single bounded worker thread a rerank attempt may own.
JEV_WORKER_NAME = "lcm-jev-rerank"
#: Hard cap on live rerank worker threads.  A hung transport can only ever hold
#: this many; further attempts fail closed instead of leaking another daemon
#: thread per turn, and the slot is released as soon as the worker returns.
JEV_MAX_OUTSTANDING_WORKERS = 1

#: Reasons for which the deterministic compiler produced individually valid
#: candidates but could not choose one.  A reranker or an advisory ordering can
#: only act there.
RERANK_ELIGIBLE_REASONS = frozenset(
    {"ambiguous_scalar_candidates", "ambiguous_fact_candidates"}
)
RERANK_ELIGIBLE_PREFIX = "ambiguous_"
#: Ambiguity that is a subset/cardinality decision (R15 scope), not a per-slot
#: choice.  Excluded rather than approximated.
RERANK_EXCLUDED_REASONS = frozenset({"ambiguous_unanchored_operands"})

_EXACT_REF_RE = re.compile(r"^lcm:(?P<store>[1-9]\d*):(?P<start>\d+)-(?P<end>\d+)$")
# Mirrors the selective selector's egress guard: the same credential-like shapes
# block this seam too.
_SECRET_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._~-]{8,}|\b(?:sk|pa)-[A-Za-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+)",
    re.IGNORECASE,
)


class JevScopeError(ValueError):
    """Raised when a candidate window must not be sent to the provider."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = str(reason_code)


class JevTransportError(RuntimeError):
    """Raised for a local transport failure or an exceeded deadline."""


@dataclass(frozen=True)
class JevLimits:
    """Code-owned request scope and budgets for one rerank attempt."""

    max_slots: int = 4
    max_candidates_per_slot: int = 8
    max_candidate_chars: int = 2000
    max_question_chars: int = 600
    max_state_chars: int = 24000
    #: Ceiling on the advisory pool, and therefore on the number of source
    #: revalidations: one bounded pass revalidates each admitted candidate
    #: exactly once.  A larger pool is refused, never truncated, because a
    #: truncated pool could not be delivered whole.
    max_advisory_candidates: int = 8
    max_calls: int = 1
    timeout_seconds: float = 4.0
    model: str = JEV_DEFAULT_MODEL

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Any) -> "JevLimits":
        """Normalize *any* caller value -- mapping **or** instance -- to the bounds.

        A ``JevLimits`` instance is re-validated field by field rather than
        trusted: a caller-built instance could otherwise widen ``max_calls``,
        ``timeout_seconds`` or the character budgets past the code-owned bounds.
        Every field must be a strict, finite type; anything else fails closed to
        the default for that field.
        """
        defaults = cls()
        if isinstance(raw, JevLimits):
            values: Mapping[str, Any] = raw.as_dict()
        elif isinstance(raw, Mapping):
            values = raw
        else:
            values = {}
        model = values.get("model")
        model = model.strip() if isinstance(model, str) else ""
        return cls(
            max_slots=_bounded(values, "max_slots", defaults.max_slots, 1, 8),
            max_candidates_per_slot=_bounded(
                values, "max_candidates_per_slot", defaults.max_candidates_per_slot, 1, 8
            ),
            max_candidate_chars=_bounded(
                values, "max_candidate_chars", defaults.max_candidate_chars, 32, 2000
            ),
            max_question_chars=_bounded(
                values, "max_question_chars", defaults.max_question_chars, 32, 2000
            ),
            max_state_chars=_bounded(
                values, "max_state_chars", defaults.max_state_chars, 512, 48000
            ),
            max_advisory_candidates=_bounded(
                values,
                "max_advisory_candidates",
                defaults.max_advisory_candidates,
                1,
                16,
            ),
            max_calls=_bounded(values, "max_calls", defaults.max_calls, 1, 1),
            timeout_seconds=_bounded_float(
                values, "timeout_seconds", defaults.timeout_seconds, 0.1, 8.0
            ),
            model=(model[:120] or defaults.model),
        )


def _bounded(values: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    """Clamp one strictly-typed integer field; anything else fails to default."""
    number = values.get(key, default)
    # bool is an int subclass; it is not a valid budget.
    if isinstance(number, bool) or not isinstance(number, int):
        return default
    return max(minimum, min(maximum, number))


def _bounded_float(
    values: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    """Clamp one strictly-typed finite float field; anything else fails to default."""
    number = values.get(key, default)
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return default
    number = float(number)
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(minimum, min(maximum, number))


def resolve_jev_options(
    config: Any,
    *,
    transport: Callable[..., Any] | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the default-off flag/mode into one bounded options mapping.

    ``config`` is the plugin :class:`~hermes_lcm.config.LCMConfig`.  The literal
    ``transport`` argument is a test/experiment seam for the deterministic
    synthetic transport; production resolves
    :func:`default_http_transport`.
    """
    enabled = bool(getattr(config, "jev_candidate_rerank_enabled", False))
    raw_mode = str(getattr(config, "jev_candidate_rerank_mode", "") or "").strip().casefold()
    if not enabled:
        mode = "off"
    elif not raw_mode:
        # Enabled with no explicit mode: measure only, never deliver.
        mode = "shadow"
    elif raw_mode in JEV_MODES:
        mode = raw_mode
    else:
        # An unrecognized mode fails closed, like the pre-answer master mode.
        mode = "off"
    return {
        "enabled": mode != "off",
        "mode": mode,
        "model": str(getattr(config, "jev_candidate_rerank_model", "") or "").strip()
        or JEV_DEFAULT_MODEL,
        "endpoint": JEV_ENDPOINT,
        "api_key_env": JEV_API_KEY_ENV,
        "limits": JevLimits.from_mapping(limits).as_dict(),
        "transport": transport,
    }


def eligible_rerank_reason(reason_code: Any) -> bool:
    """True only for code-owned ambiguity reasons a per-slot choice can fix."""
    reason = str(reason_code or "")
    if reason in RERANK_EXCLUDED_REASONS:
        return False
    if reason in RERANK_ELIGIBLE_REASONS:
        return True
    return reason.startswith(RERANK_ELIGIBLE_PREFIX) and reason not in RERANK_EXCLUDED_REASONS


def _semantic_key(candidate: Mapping[str, Any], slot: Any) -> tuple[Any, ...]:
    """Conservative raw-value key used to count distinct alternatives.

    Deliberately raw (no unit normalization): a narrower key can only keep more
    candidates, and the unchanged deterministic selectors still decide.  It is
    never authoritative for the answer.
    """
    value = candidate.get("value")
    value_type = str(getattr(slot, "value_type", "") or "")
    if value_type == "date":
        return (str(candidate.get("key") or ""), str(candidate.get("date") or ""))
    if value_type == "number":
        return (
            value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            str(candidate.get("unit") or ""),
        )
    return (" ".join(str(value or "").casefold().split()),)


def _exact_ref_ok(candidate: Mapping[str, Any]) -> bool:
    return bool(_EXACT_REF_RE.fullmatch(str(candidate.get("exact_ref") or "")))


def _slot_names(contract: Any) -> tuple[str, ...]:
    return tuple(
        str(slot.name)
        for slot in (getattr(contract, "slots", ()) or ())
        if getattr(slot, "name", None)
    )


def build_slot_options(
    candidates: Sequence[Mapping[str, Any]],
    contract: Any,
    *,
    limits: JevLimits | Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Group the product's own candidates per contract slot, with omissions.

    Returns ``({slot_name: [candidate, ...]}, metrics)``.  Candidates that fail
    the exact-ref shape or carry no quote are omitted and counted so a later
    judge error is distinguishable from a candidate omission.
    """
    resolved = JevLimits.from_mapping(limits)
    groups: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {"omitted_candidates": 0, "omission_reasons": {}}

    def omit(reason: str, count: int = 1) -> None:
        metrics["omitted_candidates"] += count
        reasons = metrics["omission_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + count

    for slot in getattr(contract, "slots", ()) or ():
        group: list[dict[str, Any]] = []
        for candidate in candidates:
            if slot.name not in (candidate.get("slot_names") or []):
                continue
            if not _exact_ref_ok(candidate):
                omit("invalid_exact_ref")
                continue
            if not str(candidate.get("quote") or "").strip():
                omit("empty_quote")
                continue
            group.append(dict(candidate))
        if len(group) > resolved.max_candidates_per_slot:
            omit("slot_candidate_cap", len(group) - resolved.max_candidates_per_slot)
            group = group[: resolved.max_candidates_per_slot]
        if group:
            groups[slot.name] = group
    return groups, metrics


def advisory_pool_status(
    candidates: Sequence[Mapping[str, Any]],
    contract: Any,
    *,
    limits: JevLimits | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether the *whole* local pool can be presented to the provider.

    The advisory may present the complete pool or nothing at all, so every
    candidate that cannot be presented -- no contract slot, malformed exact ref,
    empty quote -- rejects the entire advisory, as does a pool larger than the
    code-owned ceiling.  Truncation is never an option: a truncated pool would be
    a model-selected subset wearing the label of the full pool.
    """
    resolved = JevLimits.from_mapping(limits)
    known_slots = _slot_names(contract)
    unassigned = 0
    invalid_exact_ref = 0
    empty_quote = 0
    presentable = 0
    for candidate in candidates:
        if not [name for name in (candidate.get("slot_names") or []) if name in known_slots]:
            unassigned += 1
            continue
        if not _exact_ref_ok(candidate):
            invalid_exact_ref += 1
            continue
        if not str(candidate.get("quote") or "").strip():
            empty_quote += 1
            continue
        presentable += 1

    pool_size = len(candidates)
    if pool_size > resolved.max_advisory_candidates:
        reason_code = "advisory_pool_too_large"
    elif pool_size == 0:
        reason_code = "advisory_pool_empty"
    elif unassigned or invalid_exact_ref or empty_quote:
        reason_code = "advisory_pool_incomplete"
    else:
        reason_code = "advisory_pool_complete"
    return {
        "pool_size": pool_size,
        "presentable_candidates": presentable,
        "unassigned_candidates": unassigned,
        "invalid_exact_ref": invalid_exact_ref,
        "empty_quote": empty_quote,
        "max_advisory_candidates": resolved.max_advisory_candidates,
        "complete": reason_code == "advisory_pool_complete",
        "reason_code": reason_code,
    }


def _shrink(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: max(1, limit - 1)] + "…", True


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_context_tokens(value: Any) -> int:
    """The single context-size estimate shared with the compiler.

    ``requirements_compiler._estimate_tokens`` delegates here instead of
    re-deriving the formula, so the advisory preflight and the compiler's
    added-context budget cannot drift into disagreeing about the same string.
    """
    text = value if isinstance(value, str) else ""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def prepare_rerank_request(
    question: Any,
    *,
    contract: Any,
    slot_options: Mapping[str, Sequence[Mapping[str, Any]]],
    limits: JevLimits | Mapping[str, Any] | None = None,
    as_of: Any = None,
    mode: str = "shadow",
) -> dict[str, Any]:
    """Build the typed, bounded request plus the code-owned option index.

    Option ids are positional and code-generated; they are the only values the
    provider can return.  Raises :class:`JevScopeError` when the question or a
    candidate window must not leave the process, and when the question is longer
    than the code-owned bound -- an over-long question is refused rather than
    clipped, because a clipped question would silently drop the negation or
    scoping that the model reads at face value.

    The returned ``option_index`` is the *only* legal way to map an answer back
    to a candidate: it is built in the same loop, in the same order, as the ids
    that were actually sent.

    When the request has exactly one choosable slot and that slot is ``answer``,
    the question text validated above is also the instruction for that slot, so
    the provider reads the question directly instead of resolving the generic
    wording's ``question``/``state`` indirection.  Every other slot keeps the
    generic instruction byte for byte.
    """
    resolved = JevLimits.from_mapping(limits)
    question_text = str(question or "").strip()
    if not question_text:
        raise JevScopeError("empty_question")
    if len(question_text) > resolved.max_question_chars:
        raise JevScopeError("question_too_long")
    _guard_egress(question_text)

    candidates: dict[str, Any] = {}
    criteria_by_question: dict[str, Any] = {}
    slot_options_ids: dict[str, list[str]] = {}
    texts_by_slot: dict[str, dict[str, Mapping[str, Any]]] = {}
    option_index: dict[str, Mapping[str, Any]] = {}
    clipped_ids: list[str] = []
    text_sha256: dict[str, str] = {}
    counter = 0
    # Only the exact single ``answer`` slot uses the question as its instruction:
    # that slot *is* the product's question.  Any other or additional slot keeps
    # the frozen generic wording, so multi-slot pools are unchanged.
    explicit_answer = len(slot_options) == 1 and "answer" in slot_options

    for slot_name in slot_options:
        group = list(slot_options[slot_name])
        ids: list[str] = []
        criteria: dict[str, Any] = {
            JEV_ABSTAIN_OPTION: "no listed candidate is a supported answer"
        }
        for candidate in group:
            counter += 1
            option_id = f"c{counter:02d}"
            full_text = str(candidate.get("quote") or "")
            digest = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            display, was_clipped = _shrink(full_text, resolved.max_candidate_chars)
            _guard_egress(display)
            if was_clipped:
                clipped_ids.append(option_id)
            text_sha256[option_id] = digest
            candidates[option_id] = {
                "text": display,
                "text_sha256": digest,
                "text_clipped": was_clipped,
                "kind": str(getattr(_slot_of(contract, slot_name), "value_type", "") or "text"),
                **(
                    {"value": candidate.get("value")}
                    if candidate.get("value") is not None
                    else {}
                ),
                **({"unit": candidate.get("unit")} if candidate.get("unit") else {}),
                **({"date": candidate.get("date")} if candidate.get("date") else {}),
            }
            criteria[option_id] = f"candidate {option_id}"
            ids.append(option_id)
            option_index[option_id] = candidate
        question_id = f"slot:{slot_name}"
        criteria_by_question[question_id] = {
            "type": "choice",
            "instructions": question_text
            if explicit_answer
            else (
                "From the candidates in `state`, choose the single candidate that "
                "is the supported answer to `question` for this slot. Choose "
                f"\"{JEV_ABSTAIN_OPTION}\" when no listed candidate is supported. "
                "Return only the option key."
            ),
            "criteria": criteria,
        }
        slot_options_ids[slot_name] = ids
        texts_by_slot[slot_name] = {
            option_id: candidate for option_id, candidate in zip(ids, group)
        }

    state = {
        "question": question_text,
        # Always false: an over-long question is refused, never clipped.
        "question_clipped": False,
        "as_of": str(as_of) if as_of not in (None, "") else None,
        "contract": {
            "operation": str(getattr(contract, "operation", "") or ""),
            "requested_unit": getattr(contract, "requested_unit", None),
            "coverage_policy": str(getattr(contract, "coverage_policy", "") or ""),
            "slots": [
                {
                    "name": slot.name,
                    "value_type": str(getattr(slot, "value_type", "") or ""),
                    "anchor": getattr(slot, "anchor", None),
                    "unit": getattr(slot, "unit", None),
                }
                for slot in getattr(contract, "slots", ()) or ()
            ],
        },
        "candidates": {key: candidates[key] for key in sorted(candidates)},
    }
    request = {
        "state": state,
        "model": resolved.model,
        "questions": {
            question_id: criteria_by_question[question_id]
            for question_id in sorted(criteria_by_question)
        },
    }
    state_text = json.dumps(request, ensure_ascii=False, sort_keys=True)
    # Guard the *serialized* request, so `value`, `unit`, `date` and the contract
    # fields are checked too -- not only the question and each candidate quote.
    _guard_egress(state_text)
    if len(state_text) > resolved.max_state_chars:
        raise JevScopeError("state_budget_exceeded")
    return {
        "status": "prepared",
        "reason_code": "bounded_rerank_request",
        "version": JEV_RERANK_VERSION,
        "mode": mode,
        "request": request,
        "request_sha256": _digest(request),
        "slot_options": slot_options_ids,
        "option_index": option_index,
        "candidates_by_slot": texts_by_slot,
        "clipped_candidate_ids": clipped_ids,
        "candidate_text_sha256": text_sha256,
        "limits": resolved.as_dict(),
    }


def build_rerank_request(
    question: Any,
    *,
    contract: Any,
    slot_options: Mapping[str, Sequence[Mapping[str, Any]]],
    limits: JevLimits | Mapping[str, Any] | None = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Return only the wire payload from :func:`prepare_rerank_request`."""
    return prepare_rerank_request(
        question,
        contract=contract,
        slot_options=slot_options,
        limits=limits,
        as_of=as_of,
    )["request"]


def _slot_of(contract: Any, slot_name: str) -> Any:
    for slot in getattr(contract, "slots", ()) or ():
        if slot.name == slot_name:
            return slot
    return None


def _guard_egress(text: str) -> None:
    if _SECRET_RE.search(text):
        raise JevScopeError("secret_like_content")


def apply_slot_choices(
    candidates: Sequence[Mapping[str, Any]],
    contract: Any,
    choices: Mapping[str, Any],
    *,
    limits: JevLimits | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Revalidate provider choices against the product's own candidates.

    The option ids are rebuilt from the same pool the request was built from, so
    an id can only ever resolve to a candidate the deterministic pipeline
    already constructed.  Returns a narrowed pool (possibly unchanged) or
    ``None`` when anything fails revalidation.

    This rebuild numbers *every* group in contract order, while
    :func:`prepare_rerank_request` numbers only the slots it is handed.  The two
    therefore disagree whenever the request pool is a strict subset of the
    groups, and the choice then maps onto the wrong candidate and is refused.
    The advisory path does not use this function: it resolves through the
    request's own ``option_index``.  See :func:`resolve_advisory_choice`.
    """
    if not isinstance(choices, Mapping) or not choices:
        return None
    resolved = JevLimits.from_mapping(limits)
    groups, _ = build_slot_options(candidates, contract, limits=resolved)
    if not groups:
        return None

    option_index: dict[str, Mapping[str, Any]] = {}
    counter = 0
    for slot in getattr(contract, "slots", ()) or ():
        for candidate in groups.get(slot.name, ()):
            counter += 1
            option_index[f"c{counter:02d}"] = candidate

    chosen: dict[str, Mapping[str, Any]] = {}
    for slot_name, option_id in choices.items():
        if slot_name not in groups:
            return None
        candidate = option_index.get(str(option_id))
        if candidate is None or slot_name not in (candidate.get("slot_names") or []):
            return None
        if not _exact_ref_ok(candidate) or not str(candidate.get("quote") or "").strip():
            return None
        chosen[slot_name] = candidate
    if not chosen:
        return None

    narrowed: list[dict[str, Any]] = []
    for candidate in candidates:
        keep = True
        for slot_name, selected in chosen.items():
            if slot_name not in (candidate.get("slot_names") or []):
                continue
            if _raw_group_key(candidate) != _raw_group_key(selected):
                keep = False
                break
        if keep:
            narrowed.append(dict(candidate))
    return narrowed


def _raw_group_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    value = candidate.get("value")
    return (
        value if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value or ""),
        str(candidate.get("unit") or ""),
        str(candidate.get("key") or ""),
        str(candidate.get("date") or ""),
    )


def semantic_uniqueness_proof(
    slot_options: Mapping[str, Sequence[Mapping[str, Any]]],
    choices: Mapping[str, Any],
    *,
    contract: Any,
    option_ids_by_slot: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Decide whether a provider choice may lawfully narrow a candidate slot.

    A narrowing may only be promoted when the *chosen* candidate already covers
    every alternative in its slot -- that is, the slot has a single semantic
    group, so nothing conflicting is being hidden.  Any slot with more than one
    distinct semantic key cannot be promoted: removing the competing candidates
    would let the model manufacture uniqueness that the deterministic pipeline
    refused to assert (frozen criterion C4).

    ``slot_options`` must be the exact per-slot candidate sequence the request
    was built from and ``option_ids_by_slot`` its positional ids.  Keys are the
    conservative raw :func:`_semantic_key` tuples; no unit normalization is
    applied, so the proof never overstates uniqueness.
    """
    proven = True
    distinct_keys_by_slot: dict[str, int] = {}
    suppressed_by_slot: dict[str, int] = {}
    for slot_name, option_id in (choices or {}).items():
        if slot_name not in slot_options:
            return {
                "proven": False,
                "reason_code": "unknown_slot",
                "distinct_keys_by_slot": distinct_keys_by_slot,
                "suppressed_by_slot": suppressed_by_slot,
            }
        group = list(slot_options[slot_name])
        ids = list(option_ids_by_slot.get(slot_name, ()))
        if str(option_id) not in ids or len(ids) != len(group):
            return {
                "proven": False,
                "reason_code": "option_id_mismatch",
                "distinct_keys_by_slot": distinct_keys_by_slot,
                "suppressed_by_slot": suppressed_by_slot,
            }
        slot = _slot_of(contract, slot_name)
        chosen = group[ids.index(str(option_id))]
        chosen_key = _semantic_key(chosen, slot)
        keys = {_semantic_key(candidate, slot) for candidate in group}
        covered = [candidate for candidate in group if _semantic_key(candidate, slot) == chosen_key]
        distinct_keys_by_slot[slot_name] = len(keys)
        suppressed_by_slot[slot_name] = len(group) - len(covered)
        if keys != {chosen_key} or not covered:
            proven = False
    if not distinct_keys_by_slot:
        proven = False
    return {
        "proven": proven,
        "reason_code": "semantic_uniqueness_proven" if proven else "semantic_uniqueness_unproven",
        "distinct_keys_by_slot": distinct_keys_by_slot,
        "suppressed_by_slot": suppressed_by_slot,
    }


def _applied_uniqueness_is_impossible(
    slot_options: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Any,
) -> dict[str, Any] | None:
    """Choice-independent impossibility proof for ``applied`` mode, or ``None``.

    ``choosable`` holds only slots with at least two distinct semantic keys, and
    :func:`semantic_uniqueness_proof` proves a slot only when its chosen
    candidate covers a single key.  When every choosable slot already has more
    than one key, no provider answer can pass that proof, so the paid call cannot
    change the ``applied`` outcome and is refused before any dispatch.

    Returns ``None`` -- the caller then dispatches exactly as before -- whenever
    some slot could still prove uniqueness, so the skip is proved per pool rather
    than assumed.  ``suppressed_by_slot`` stays empty: with no choice there is
    nothing measured to suppress.
    """
    distinct_keys_by_slot = {
        slot_name: len(
            {_semantic_key(candidate, _slot_of(contract, slot_name)) for candidate in group}
        )
        for slot_name, group in slot_options.items()
    }
    if not distinct_keys_by_slot or any(
        count < 2 for count in distinct_keys_by_slot.values()
    ):
        return None
    return {
        "proven": False,
        "reason_code": "semantic_uniqueness_unproven",
        "distinct_keys_by_slot": distinct_keys_by_slot,
        "suppressed_by_slot": {},
    }


def revalidated_pool_entries(
    candidates: Sequence[Mapping[str, Any]],
    *,
    revalidate: Callable[[Mapping[str, Any]], bool] | None,
    limit: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Revalidate every admitted pool candidate against the source, or refuse.

    ``revalidate`` is the caller's own source normalizer (the compiler passes its
    exact-ref/span hydrator), so the check is the same one that admitted the
    candidate in the first place.  The returned entries are the only material the
    advisory will ever render, and the pass is bounded by ``limit``: an oversized
    pool is refused by :func:`advisory_pool_status` before this runs, so a
    pathological pool can never turn into an unbounded source scan.
    """
    bound = max(0, int(limit))
    entries: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if index >= bound:
            return None, "advisory_pool_too_large"
        if revalidate is not None and not revalidate(candidate):
            return None, "advisory_source_revalidation_failed"
        entries.append(dict(candidate))
    if not entries:
        return None, "advisory_pool_empty"
    return entries, "advisory_sources_revalidated"


def resolve_advisory_choice(
    prepared: Mapping[str, Any],
    choices: Mapping[str, Any],
    *,
    contract: Any,
) -> tuple[list[str] | None, str]:
    """Map wire option ids back through the request's *own* option index.

    ``prepared`` is the exact mapping :func:`prepare_rerank_request` returned for
    the request that was sent, so the ids here are the ids the provider saw.  A
    returned id that is not in that index, that resolves to a candidate outside
    its slot, or that fails the ref/quote shape refuses the whole choice.
    """
    if not isinstance(choices, Mapping) or not choices:
        return None, "missing_choice"
    index = prepared.get("option_index")
    if not isinstance(index, Mapping):
        return None, "missing_option_index"
    chosen_refs: list[str] = []
    for slot_name, option_id in choices.items():
        ids = list((prepared.get("slot_options") or {}).get(slot_name, ()))
        if str(option_id) not in ids:
            return None, "unmapped_option_id"
        candidate = index.get(str(option_id))
        if not isinstance(candidate, Mapping):
            return None, "unmapped_option_id"
        if str(slot_name) not in (candidate.get("slot_names") or []):
            return None, "unmapped_option_id"
        if not _exact_ref_ok(candidate) or not str(candidate.get("quote") or "").strip():
            return None, "unmapped_option_id"
        # The chosen candidate must be one of the entries that were revalidated
        # against the source before dispatch; the provider cannot introduce one.
        chosen_refs.append(str(candidate["exact_ref"]))
    if not chosen_refs:
        return None, "missing_choice"
    return chosen_refs, "advisory_choice_resolved"


def order_advisory_pool(
    entries: Sequence[Mapping[str, Any]], chosen_refs: Sequence[str]
) -> list[dict[str, Any]]:
    """Return the complete pool with the chosen candidates moved to the front.

    A permutation, never a filter: the returned list holds exactly the entries
    it was given, in a deterministic order.  Moving a candidate forward cannot
    hide the competing candidates behind it.
    """
    chosen = {str(item) for item in chosen_refs}
    head = [dict(item) for item in entries if str(item.get("exact_ref")) in chosen]
    tail = [dict(item) for item in entries if str(item.get("exact_ref")) not in chosen]
    return [*head, *tail]


def _neutralise_envelope_text(value: str) -> str:
    """Escape tag-shaped text in a rendered quote so it cannot forge the block.

    Stored evidence is untrusted *input to a prompt envelope*: a quote carrying
    the advisory's own opening or closing tag would otherwise end the block --
    or open a second one -- and leave the rest of that quote structurally
    outside the envelope that is supposed to carry the complete pool.  Only the
    rendered text is escaped: the structured record keeps the exact source ref
    and quote, and no candidate is dropped.
    """
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_advisory_order(
    entries: Sequence[Mapping[str, Any]], contract: Any = None
) -> str:
    """Render the ordered pool as product-owned text, or ``""`` for no pool.

    Only the product's own exact ref and quote appear.  Whitespace inside a quote
    is collapsed to keep one candidate per line, and tag-shaped text inside a
    quote is escaped (see :func:`_neutralise_envelope_text`) so a stored quote
    cannot close the envelope early; the rendered string's length is independent
    of the order, which is what lets the budget be checked before the paid call.
    Competing candidates are labelled unresolved so the block cannot be read as
    an answer.
    """
    pool = [dict(item) for item in entries]
    if not pool:
        return ""
    lines = [
        f'<{JEV_ADVISORY_TAG} version="{JEV_RERANK_VERSION}">',
        "Advisory presentation order over the complete admitted candidate pool.",
        "This block is an ordering only. No candidate was removed, no competing",
        "candidate was resolved, no requirement was closed, no value was computed",
        "and no exhaustive coverage was established. The product's own ambiguity",
        "verdict is unchanged and remains authoritative.",
    ]
    for entry in pool:
        quote = " ".join(str(entry.get("quote") or "").split())
        lines.append(f"- [{entry.get('exact_ref')}] {_neutralise_envelope_text(quote)}")
    lines.append(f"</{JEV_ADVISORY_TAG}>")
    return "\n".join(lines)


@dataclass
class JevHttpResponse:
    """Minimal HTTP result passed to the parser."""

    status: int
    body: bytes
    headers: Mapping[str, str]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every 3xx.

    Returning ``None`` makes urllib raise the HTTPError instead of following the
    redirect, so a fixed endpoint URL cannot be redirected to another origin --
    which would otherwise forward the ``Authorization`` header off-host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _bounded_read(stream: Any, limit: int, *, reason_code: str) -> bytes:
    """Read at most ``limit`` bytes or fail closed with ``reason_code``."""
    data = stream.read(limit + 1)
    if not data:
        return b""
    if len(data) > limit:
        raise JevTransportError(reason_code)
    return bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")


#: Built once: urllib's default opener follows redirects.
_JEV_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def default_http_transport(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> JevHttpResponse:
    """POST JSON with urllib, refusing redirects and bounding both bodies.

    One request, one response.  A 3xx is a refusal, never a follow; the success
    body is bounded by :data:`JEV_MAX_RESPONSE_BYTES` and an HTTP error body by
    :data:`JEV_MAX_ERROR_BODY_BYTES`, so neither can be read unbounded.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with _JEV_OPENER.open(request, timeout=timeout) as response:
            status = int(response.status)
            if 300 <= status < 400:
                raise JevTransportError("redirect_refused")
            return JevHttpResponse(
                status=status,
                body=_bounded_read(response, JEV_MAX_RESPONSE_BYTES, reason_code="response_too_large"),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        if 300 <= status < 400:
            raise JevTransportError("redirect_refused") from exc
        return JevHttpResponse(
            status=status,
            body=_bounded_read(exc, JEV_MAX_ERROR_BODY_BYTES, reason_code="error_body_too_large"),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )


#: Rerank worker threads, bounded by :data:`JEV_MAX_OUTSTANDING_WORKERS`.  A
#: hung transport thread is released as soon as it returns; until then a second
#: attempt is refused instead of spawning another daemon thread per turn.
_worker_lock = threading.Lock()
_outstanding_workers: list[threading.Thread] = []


def live_worker_count() -> int:
    """Number of live rerank worker threads (diagnostics and bounded tests)."""
    with _worker_lock:
        _outstanding_workers[:] = [item for item in _outstanding_workers if item.is_alive()]
        return len(_outstanding_workers)


def _call_with_deadline(
    *,
    transport: Callable[..., Any],
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Any:
    """One bounded attempt; a hung transport can never block the turn.

    The whole attempt -- thread start, call and queue hand-off -- is measured
    against ``timeout`` before the call and accounted after it, and the worker
    count is bounded to one so a hung provider cannot accumulate threads.
    """
    budget = max(0.0, float(timeout))
    with _worker_lock:
        live = [item for item in _outstanding_workers if item.is_alive()]
        _outstanding_workers[:] = live
        if len(live) >= JEV_MAX_OUTSTANDING_WORKERS:
            raise JevTransportError("attempt_in_progress")
        box: queue.Queue = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                box.put((True, transport(url=url, payload=payload, headers=headers, timeout=budget)))
            except BaseException as exc:  # noqa: BLE001 - reported as a fallback
                box.put((False, exc))
            finally:
                with _worker_lock:
                    _outstanding_workers[:] = [
                        item for item in _outstanding_workers if item is not threading.current_thread()
                    ]

        worker = threading.Thread(target=run, name=JEV_WORKER_NAME, daemon=True)
        _outstanding_workers.append(worker)
        worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise JevTransportError("deadline_exceeded")
    try:
        ok, value = box.get_nowait()
    except queue.Empty as exc:  # pragma: no cover - defensive
        raise JevTransportError("transport_no_result") from exc
    if not ok:
        if isinstance(value, JevTransportError):
            raise value
        raise JevTransportError(f"transport_error:{type(value).__name__}")
    return value


def parse_rerank_response(
    raw: Any,
    *,
    question_ids: Sequence[str],
    option_ids_by_question: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate one provider body against the request that was sent.

    Accepts the plugin's own :class:`JevHttpResponse` or any object exposing
    ``status`` and ``body`` (structurally compatible with the embedding
    provider's ``HttpResponse``).
    """
    if isinstance(raw, Mapping):
        status = int(raw.get("status", 200))
        body = raw.get("body")
    else:
        status_attr = getattr(raw, "status", None)
        body = getattr(raw, "body", None)
        if status_attr is None:
            return {"status": "fallback", "reason_code": "unexpected_transport_result"}
        try:
            status = int(status_attr)
        except (TypeError, ValueError, OverflowError):
            return {"status": "fallback", "reason_code": "unexpected_transport_result"}
    if status != 200:
        return {"status": "fallback", "reason_code": f"http_{status}"}
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", errors="replace")
    elif isinstance(body, str):
        text = body
    else:  # pragma: no cover - defensive
        return {"status": "fallback", "reason_code": "malformed_body"}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {"status": "fallback", "reason_code": "malformed_body"}
    if not isinstance(payload, Mapping):
        return {"status": "fallback", "reason_code": "malformed_body"}

    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        return {"status": "fallback", "reason_code": "missing_answers"}
    expected = {str(item) for item in question_ids}
    if {str(key) for key in answers} != expected:
        return {"status": "fallback", "reason_code": "answer_set_mismatch"}

    choices: dict[str, str] = {}
    abstained: list[str] = []
    confidence_by_question: dict[str, float] = {}
    probabilities_by_question: dict[str, dict[str, float]] = {}
    for question_id in sorted(expected):
        answer = answers[question_id]
        if not isinstance(answer, Mapping):
            return {"status": "fallback", "reason_code": "malformed_answer"}
        if str(answer.get("type")) != "choice":
            return {"status": "fallback", "reason_code": "unexpected_answer_type"}
        option = str(answer.get("choice") or "")
        allowed = {str(item) for item in option_ids_by_question.get(question_id, ())}
        if option == JEV_ABSTAIN_OPTION:
            abstained.append(question_id)
            continue
        if option not in allowed:
            return {"status": "fallback", "reason_code": "unknown_choice_option"}
        # `probabilities` and `confidence` are recorded as observations the
        # consumer can weigh.  They are validated structurally only: no
        # calibrated acceptance floor is invented here.
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            observed_probabilities = _observed_probabilities(probabilities)
            if observed_probabilities is None:
                return {"status": "fallback", "reason_code": "malformed_probabilities"}
            probabilities_by_question[question_id] = observed_probabilities
        confidence = answer.get("confidence")
        if confidence is not None:
            observed_confidence = _observed_score(confidence)
            if observed_confidence is None:
                return {"status": "fallback", "reason_code": "malformed_confidence"}
            confidence_by_question[question_id] = observed_confidence
        choices[question_id.removeprefix("slot:")] = option

    raw_usage = payload.get("usage")
    usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
    parsed = {
        "status": "abstained" if abstained else "ok",
        "reason_code": "provider_abstained" if abstained else "provider_choice",
        "choices": choices,
        "abstained_questions": abstained,
        "confidence_by_question": confidence_by_question,
        "probabilities_by_question": probabilities_by_question,
        "model": str(payload.get("model") or "")[:120] or None,
        "usage": {
            # Unknown usage stays unknown; it is never reported as zero.
            "input_tokens": _observed_int(usage.get("input_tokens")),
            "output_tokens": _observed_int(usage.get("output_tokens")),
            # Billing is not returned by this endpoint; unknown stays unknown.
            "cost_usd": None,
        },
    }
    return parsed


def _observed_int(value: Any) -> int | None:
    """Return a reported non-negative integer count, or ``None`` when unknown."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _observed_score(value: Any) -> float | None:
    """Return a finite 0..1 provider score, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number if 0.0 <= number <= 1.0 else None


def _observed_probabilities(value: Any) -> dict[str, float] | None:
    """Normalize a provider ``probabilities`` mapping, or fail closed."""
    if not isinstance(value, Mapping):
        return None
    normalised: dict[str, float] = {}
    for key, item in value.items():
        score = _observed_score(item)
        if score is None:
            return None
        normalised[str(key)] = score
    return normalised


def _advisory_state(record: Mapping[str, Any], limits: JevLimits) -> dict[str, Any]:
    return {
        "status": "pending",
        "reason_code": "advisory_pending",
        "pool_size": 0,
        "presentable_candidates": 0,
        "unassigned_candidates": 0,
        "invalid_exact_ref": 0,
        "empty_quote": 0,
        "max_advisory_candidates": limits.max_advisory_candidates,
        "pool_complete": False,
        "revalidated_candidates": 0,
        "context_estimate_tokens": 0,
        "budget_tokens": None,
        "order": [],
        "chosen_exact_refs": [],
        "selected_options": {},
        "context": None,
    }


def _record_base(options: Mapping[str, Any], limits: JevLimits) -> dict[str, Any]:
    return {
        "version": JEV_RERANK_VERSION,
        "mode": str(options.get("mode") or "off"),
        "status": "disabled",
        "reason_code": "feature_disabled",
        # ``applied`` is the answer-promotion authority and stays reserved for a
        # proven narrowing; an advisory ordering never sets it.
        "applied": False,
        # Set by the caller only when the ordered block was really delivered into
        # its consumer context, not merely received from the provider.
        "advisory_applied": False,
        "advisory": None,
        "calls": 0,
        "endpoint": str(options.get("endpoint") or JEV_ENDPOINT),
        "model": str(options.get("model") or limits.model),
        "response_model": None,
        "choices": {},
        "shadow_decision": None,
        "uniqueness": None,
        "confidence_by_question": {},
        "probabilities_by_question": {},
        "slot_options": {},
        "request": None,
        "request_sha256": None,
        "response_sha256": None,
        "candidate_omissions": 0,
        "omission_reasons": {},
        "clipped_candidate_ids": [],
        "candidate_text_sha256": {},
        # Nothing was measured, so usage stays unknown -- never zero.
        "usage": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        "limits": limits.as_dict(),
        "timeout_seconds": float(limits.timeout_seconds),
        "latency_ms": 0.0,
    }


def _dispatch(
    prepared: Mapping[str, Any],
    record: dict[str, Any],
    *,
    options: Mapping[str, Any],
    effective: JevLimits,
) -> Any:
    """One bounded paid attempt, or ``None`` after recording a fallback.

    Shared by the shadow/applied path and the advisory path so both inherit the
    identical deadline, single-call, header-clearing and accounting behavior.
    """
    key_env = str(options.get("api_key_env") or JEV_API_KEY_ENV)
    api_key = os.environ.get(key_env, "")
    if not api_key:
        record["status"] = "fallback"
        record["reason_code"] = "missing_api_key"
        return None
    endpoint = str(options.get("endpoint") or JEV_ENDPOINT)
    transport = options.get("transport") or default_http_transport
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    try:
        raw = _call_with_deadline(
            transport=transport,
            url=endpoint,
            payload=prepared["request"],
            headers=headers,
            timeout=effective.timeout_seconds,
        )
    except JevTransportError as exc:
        record["calls"] = 1
        record["latency_ms"] = round((time.perf_counter() - started) * 1_000.0, 3)
        record["status"] = "fallback"
        record["reason_code"] = str(exc)
        return None
    finally:
        headers.clear()

    record["calls"] = 1
    record["latency_ms"] = round((time.perf_counter() - started) * 1_000.0, 3)
    body = getattr(raw, "body", None)
    if isinstance(body, (bytes, bytearray)):
        record["response_sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    return raw


def _advisory_reject(record: dict[str, Any], reason_code: str, *, status: str = "rejected") -> dict[str, Any]:
    advisory = record.get("advisory")
    if isinstance(advisory, dict):
        advisory["status"] = status
        advisory["reason_code"] = reason_code
    return record


def _rerank_advisory(
    question: Any,
    *,
    contract: Any,
    candidates: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    effective: JevLimits,
    as_of: Any,
    context_budget_tokens: Any,
    revalidate: Callable[[Mapping[str, Any]], bool] | None,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Order the complete revalidated pool for presentation, or change nothing.

    The deterministic verdict is already decided by the time this runs, and
    nothing here can move it: the caller keeps its pool, its reason code and its
    selectors' verdict regardless of the outcome.
    """
    advisory = _advisory_state(record, effective)
    record["advisory"] = advisory

    # Preflight 1: the whole pool must be presentable, and bounded.
    pool = advisory_pool_status(candidates, contract, limits=effective)
    advisory.update(
        {
            "pool_size": pool["pool_size"],
            "presentable_candidates": pool["presentable_candidates"],
            "unassigned_candidates": pool["unassigned_candidates"],
            "invalid_exact_ref": pool["invalid_exact_ref"],
            "empty_quote": pool["empty_quote"],
            "pool_complete": pool["complete"],
        }
    )
    if not pool["complete"]:
        record["reason_code"] = pool["reason_code"]
        record["status"] = "skipped" if pool["reason_code"] == "advisory_pool_empty" else "refused"
        return _advisory_reject(record, pool["reason_code"])

    # Preflight 2: there must be something to order.  A pool whose every slot has
    # a single semantic group has no competing presentation at all.
    groups, metrics = build_slot_options(candidates, contract, limits=effective)
    record["candidate_omissions"] = metrics["omitted_candidates"]
    record["omission_reasons"] = metrics["omission_reasons"]
    if metrics["omitted_candidates"]:
        # ``advisory_pool_status`` already rejected this pool; belt and braces.
        record["status"] = "refused"
        record["reason_code"] = "advisory_pool_incomplete"
        return _advisory_reject(record, "advisory_pool_incomplete")
    choosable = {
        slot_name: group
        for slot_name, group in groups.items()
        if len({_semantic_key(candidate, _slot_of(contract, slot_name)) for candidate in group}) >= 2
    }
    if not choosable:
        record["status"] = "skipped"
        record["reason_code"] = "no_ambiguous_slot"
        return _advisory_reject(record, "no_ambiguous_slot", status="skipped")

    # Preflight 3: every candidate that will be rendered is revalidated against
    # the source, once each, under the code-owned pool bound.
    entries, revalidation_reason = revalidated_pool_entries(
        candidates, revalidate=revalidate, limit=effective.max_advisory_candidates
    )
    if entries is None:
        record["status"] = "refused"
        record["reason_code"] = revalidation_reason
        return _advisory_reject(record, revalidation_reason)
    advisory["revalidated_candidates"] = len(entries)

    # Preflight 4: build the request over the *whole* pool, and refuse a clipped
    # or over-budget block now, before paying for a result we could not deliver.
    try:
        prepared = prepare_rerank_request(
            question,
            contract=contract,
            slot_options=groups,
            limits=effective,
            as_of=as_of,
            mode=record["mode"],
        )
    except JevScopeError as exc:
        record["status"] = "scope_blocked"
        record["reason_code"] = exc.reason_code
        return _advisory_reject(record, exc.reason_code, status="blocked")
    record.update(
        {
            "status": "ok",
            "reason_code": "provider_choice_pending",
            "request": prepared["request"],
            "request_sha256": prepared["request_sha256"],
            "slot_options": prepared["slot_options"],
            "clipped_candidate_ids": prepared["clipped_candidate_ids"],
            "candidate_text_sha256": prepared["candidate_text_sha256"],
        }
    )
    if prepared["clipped_candidate_ids"]:
        record["status"] = "refused"
        record["reason_code"] = "advisory_pool_clipped"
        return _advisory_reject(record, "advisory_pool_clipped")

    budget = _coerce_budget(context_budget_tokens)
    advisory["budget_tokens"] = budget
    identity_render = render_advisory_order(entries, contract)
    identity_estimate = estimate_context_tokens(identity_render)
    advisory["context_estimate_tokens"] = identity_estimate
    if budget is None:
        # An undeclared bound is not an unlimited bound.
        record["status"] = "refused"
        record["reason_code"] = "advisory_context_budget_undeclared"
        return _advisory_reject(record, "advisory_context_budget_undeclared")
    if identity_estimate > budget:
        record["status"] = "refused"
        record["reason_code"] = "advisory_context_budget_exceeded"
        return _advisory_reject(record, "advisory_context_budget_exceeded")

    raw = _dispatch(prepared, record, options=options, effective=effective)
    if raw is None:
        return _advisory_reject(record, record["reason_code"])

    parsed = parse_rerank_response(
        raw,
        question_ids=tuple(f"slot:{name}" for name in prepared["slot_options"]),
        option_ids_by_question={
            f"slot:{name}": ids for name, ids in prepared["slot_options"].items()
        },
    )
    if parsed["status"] == "fallback":
        record["status"] = "fallback"
        record["reason_code"] = parsed["reason_code"]
        return _advisory_reject(record, parsed["reason_code"])
    record["response_model"] = parsed["model"]
    record["usage"] = parsed["usage"]
    record["confidence_by_question"] = parsed["confidence_by_question"]
    record["probabilities_by_question"] = parsed["probabilities_by_question"]
    if parsed["status"] == "abstained":
        record["status"] = "abstained"
        record["reason_code"] = "abstained"
        return _advisory_reject(record, "abstained", status="abstained")

    record["choices"] = parsed["choices"]
    chosen_refs, resolution_reason = resolve_advisory_choice(
        prepared, parsed["choices"], contract=contract
    )
    if chosen_refs is None:
        record["status"] = "fallback"
        record["reason_code"] = resolution_reason
        return _advisory_reject(record, resolution_reason)

    # A returned id may only ever move a candidate that was revalidated before
    # dispatch into the head of the block; the block itself is the whole pool.
    revalidated_refs = {str(item.get("exact_ref")) for item in entries}
    if any(ref not in revalidated_refs for ref in chosen_refs):
        record["status"] = "fallback"
        record["reason_code"] = "unrevalidated_choice"
        return _advisory_reject(record, "unrevalidated_choice")

    ordered = order_advisory_pool(entries, chosen_refs)
    if len(ordered) != len(entries):
        record["status"] = "fallback"
        record["reason_code"] = "advisory_pool_changed"
        return _advisory_reject(record, "advisory_pool_changed")
    try:
        context = render_advisory_order(ordered, contract)
        _guard_egress(context)
    except JevScopeError as exc:
        record["status"] = "refused"
        record["reason_code"] = exc.reason_code
        return _advisory_reject(record, exc.reason_code)
    if estimate_context_tokens(context) > budget:
        # Unreachable while the render is order-independent; kept so a future
        # per-position rendering cannot quietly overspend the budget.
        record["status"] = "refused"
        record["reason_code"] = "advisory_context_budget_exceeded"
        return _advisory_reject(record, "advisory_context_budget_exceeded")

    advisory.update(
        {
            "status": "ready",
            "reason_code": "advisory_order_ready",
            "order": [str(item.get("exact_ref")) for item in ordered],
            "chosen_exact_refs": list(chosen_refs),
            "selected_options": {str(key): str(value) for key, value in parsed["choices"].items()},
            "context": context,
        }
    )
    record["status"] = "ok"
    record["reason_code"] = "provider_choice"
    return record


def _coerce_budget(value: Any) -> int | None:
    """Return a usable positive token budget, or ``None`` when undeclared."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def rerank_candidates(
    question: Any,
    *,
    contract: Any,
    candidates: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None,
    limits: JevLimits | Mapping[str, Any] | None = None,
    reason_code: Any = "",
    as_of: Any = None,
    advisory_context_budget_tokens: Any = None,
    revalidate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """One bounded attempt over the deterministic candidate pool.

    Never raises for provider problems and never mutates the pool.  ``status`` is
    one of ``disabled`` / ``skipped`` / ``scope_blocked`` / ``ok`` /
    ``refused`` / ``abstained`` / ``fallback``; only ``ok`` carries a usable
    ``choices`` map.  In ``applied`` mode ``ok`` is reachable only when
    :func:`semantic_uniqueness_proof` proved the chosen candidate already covers
    every alternative in its slot.  In ``advisory`` mode ``ok`` means an ordered
    block was built over the complete revalidated pool and left in
    ``record["advisory"]["context"]`` for the caller to deliver; the caller --
    not this function -- decides whether it was delivered, and sets
    ``advisory_applied`` accordingly.  ``refused`` and every failure mode leave
    the caller's pool untouched.

    ``advisory_context_budget_tokens`` is the caller's added-context bound
    (``RequirementsBudgets.max_added_context_tokens``).  An undeclared or
    exceeded bound refuses the advisory before dispatch.

    ``revalidate`` is the caller's own source normalizer for an exact ref, span
    and quote.  Every advisory candidate is checked through it before the request
    is built.
    """
    options = dict(options or {})
    effective = JevLimits.from_mapping(limits if limits is not None else options.get("limits"))
    record = _record_base(options, effective)
    if not options.get("enabled"):
        return record

    record["status"] = "skipped"
    record["reason_code"] = "reason_not_rerank_eligible"
    if not eligible_rerank_reason(reason_code):
        return record

    if record["mode"] == JEV_ADVISORY_MODE:
        return _rerank_advisory(
            question,
            contract=contract,
            candidates=candidates,
            options=options,
            effective=effective,
            as_of=as_of,
            context_budget_tokens=advisory_context_budget_tokens,
            revalidate=revalidate,
            record=record,
        )

    groups, metrics = build_slot_options(candidates, contract, limits=effective)
    record["candidate_omissions"] = metrics["omitted_candidates"]
    record["omission_reasons"] = metrics["omission_reasons"]
    choosable = {
        slot_name: group
        for slot_name, group in groups.items()
        if len({_semantic_key(candidate, _slot_of(contract, slot_name)) for candidate in group}) >= 2
    }
    record["reason_code"] = "no_ambiguous_slot"
    if not choosable:
        return record

    try:
        prepared = prepare_rerank_request(
            question,
            contract=contract,
            slot_options=choosable,
            limits=effective,
            as_of=as_of,
            mode=record["mode"],
        )
    except JevScopeError as exc:
        record["status"] = "scope_blocked"
        record["reason_code"] = exc.reason_code
        return record

    record.update(
        {
            "status": "ok",
            "reason_code": "provider_choice_pending",
            "request": prepared["request"],
            "request_sha256": prepared["request_sha256"],
            "slot_options": prepared["slot_options"],
            "clipped_candidate_ids": prepared["clipped_candidate_ids"],
            "candidate_text_sha256": prepared["candidate_text_sha256"],
        }
    )

    if record["mode"] == "applied":
        impossible = _applied_uniqueness_is_impossible(choosable, contract)
        if impossible is not None:
            # Nothing the provider answers could be promoted from this pool, so
            # the request is not worth sending.  The refusal reason is the same
            # one the post-response guard would have produced.
            record["status"] = "refused"
            record["reason_code"] = "semantic_uniqueness_unproven"
            record["uniqueness"] = impossible
            return record

    raw = _dispatch(prepared, record, options=options, effective=effective)
    if raw is None:
        return record

    parsed = parse_rerank_response(
        raw,
        question_ids=tuple(f"slot:{name}" for name in prepared["slot_options"]),
        option_ids_by_question={
            f"slot:{name}": ids for name, ids in prepared["slot_options"].items()
        },
    )
    if parsed["status"] == "fallback":
        record["status"] = "fallback"
        record["reason_code"] = parsed["reason_code"]
        return record
    record["response_model"] = parsed["model"]
    record["usage"] = parsed["usage"]
    record["confidence_by_question"] = parsed["confidence_by_question"]
    record["probabilities_by_question"] = parsed["probabilities_by_question"]
    if parsed["status"] == "abstained":
        record["status"] = "abstained"
        record["reason_code"] = "abstained"
        return record
    record["choices"] = parsed["choices"]
    record["status"] = "ok"
    record["reason_code"] = "provider_choice"
    if record["mode"] == "applied":
        # The choice is kept as a shadow decision, but promotion requires a
        # semantic-uniqueness proof the ambiguous path cannot supply.  Refusing
        # here is what keeps a model pick from manufacturing completeness.
        record["shadow_decision"] = dict(parsed["choices"])
        proof = semantic_uniqueness_proof(
            {
                slot_name: list(entries.values())
                for slot_name, entries in prepared["candidates_by_slot"].items()
            },
            parsed["choices"],
            contract=contract,
            option_ids_by_slot=prepared["slot_options"],
        )
        record["uniqueness"] = proof
        if not proof["proven"]:
            record["status"] = "refused"
            record["reason_code"] = "semantic_uniqueness_unproven"
            record["applied"] = False
    return record
