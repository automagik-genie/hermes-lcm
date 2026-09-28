"""Provider-free contract tests for the optional TypeSafe Jev candidate seam.

Every test here injects a deterministic synthetic transport.  No network call,
no provider credential and no live Jev decision is exercised; the assertions are
about request shape, code-owned bounds, product revalidation, abstention and
baseline preservation.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.evidence_compiler import compile_preanswer_evidence
from hermes_lcm.jev_candidates import (
    JEV_ABSTAIN_OPTION,
    JEV_ADVISORY_MODE,
    JEV_DEFAULT_MODEL,
    JEV_ENDPOINT,
    JEV_MODES,
    JevLimits,
    JevScopeError,
    JevTransportError,
    apply_slot_choices,
    build_rerank_request,
    build_slot_options,
    default_http_transport,
    eligible_rerank_reason,
    live_worker_count,
    prepare_rerank_request,
    resolve_jev_options,
    semantic_uniqueness_proof,
)
from hermes_lcm.store import MessageStore
from tests.fixtures.jev_synthetic_transport import SyntheticJevTransport
from tests.fixtures import jev_synthetic_transport

COMPETING_QUESTION = "How many points do I need to redeem the reward?"

# The fields a provider may never move.  Missing and null are distinct here, so
# an absent key is compared as absent rather than silently equal to ``None``.
DETERMINISTIC_FIELDS = (
    "status",
    "state",
    "reason_code",
    "contract",
    "baseline",
    "closed_requirements",
    "open_requirements",
    "evidence",
    "novel_exact_refs",
    "direct_fact",
    "computation",
    "computation_sha256",
    "finite_coverage",
    "coverage_certificate",
)


def test_synthetic_fixture_abstain_sentinel_matches_the_adapter():
    assert jev_synthetic_transport.ABSTAIN_OPTION == JEV_ABSTAIN_OPTION


@pytest.fixture(autouse=True)
def _typesafe_key(monkeypatch):
    """Present a dummy credential so the real transport path is reachable.

    No call leaves the process: every test injects a synthetic transport.
    """
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-not-a-credential")


@pytest.fixture(autouse=True)
def _drain_bounded_worker():
    """Wait for the bounded rerank worker before the next test.

    A timed-out attempt keeps its single worker thread until the transport
    returns; the cap is deliberate (see ``test_a_hung_attempt_bounds_the_next``),
    so the suite drains it rather than letting one test's hang become the next
    test's reason code.
    """
    yield
    deadline = time.monotonic() + 5.0
    while live_worker_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert live_worker_count() == 0


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    return SimpleNamespace(
        _config=config,
        _store=store,
        _assertions=None,
        _session_occurrence_dates={},
    )


def _append(engine, content: str, *, session_id: str = "session-a", role: str = "user"):
    store_id = engine._store.append(session_id, {"role": role, "content": content})
    return {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "quote": content,
    }


def _competing_scalars(engine):
    first = _append(engine, "You need 15 points to redeem the reward.")
    second = _append(
        engine, "You need 20 points to redeem the reward.", session_id="session-b"
    )
    return [first, second]


def _compile(engine, question, refs, **kwargs):
    return compile_preanswer_evidence(
        question,
        engine=engine,
        baseline_refs=refs,
        enabled=True,
        budgets={"max_retrieval_calls": 0},
        **kwargs,
    )


def _stable(result) -> dict:
    """Result minus wall-clock-only fields, for byte-comparison assertions."""
    payload = json.loads(json.dumps(result, default=str))
    payload["metrics"]["latency_ms"] = 0.0
    payload["trace"]["digest_sha256"] = None
    return payload


def _deterministic(result) -> dict:
    return {key: _stable(result).get(key) for key in DETERMINISTIC_FIELDS}


def _without_jev(result) -> dict:
    payload = _stable(result)
    payload.pop("jev", None)
    return payload


def _jev_options(transport, **overrides):
    options = resolve_jev_options(
        LCMConfig(
            jev_candidate_rerank_enabled=True,
            jev_candidate_rerank_mode=overrides.pop("mode", "advisory"),
        ),
        transport=transport,
    )
    options.update(overrides)
    return options


def _single_slot_contract(value_type: str = "text"):
    return SimpleNamespace(slots=[SimpleNamespace(name="answer", value_type=value_type)])


# --------------------------------------------------------------------------
# Option resolution, code-owned scope and bounds
# --------------------------------------------------------------------------


def test_advisory_is_a_declared_mode_and_disabled_config_stays_off():
    assert JEV_ADVISORY_MODE == "advisory"
    assert JEV_MODES == frozenset({"off", "shadow", "applied", "advisory"})

    disabled = resolve_jev_options(LCMConfig())
    assert disabled["enabled"] is False
    assert disabled["mode"] == "off"
    assert disabled["model"] == JEV_DEFAULT_MODEL
    assert disabled["endpoint"] == JEV_ENDPOINT
    assert disabled["api_key_env"] == "TYPESAFE_API_KEY"


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        ("", "shadow"),
        ("off", "off"),
        ("shadow", "shadow"),
        ("applied", "applied"),
        ("advisory", "advisory"),
        ("ADVISORY", "advisory"),
        (" advisory ", "advisory"),
        ("bogus", "off"),
        ("advisory2", "off"),
    ],
)
def test_enabled_mode_resolution_fails_closed_on_unknown_values(raw_mode, expected):
    options = resolve_jev_options(
        LCMConfig(jev_candidate_rerank_enabled=True, jev_candidate_rerank_mode=raw_mode)
    )
    assert options["mode"] == expected
    assert options["enabled"] is (expected != "off")


def test_unknown_mode_is_disabled_equivalence_with_no_provider_call(tmp_path):
    transport = SyntheticJevTransport()
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        absent = _compile(engine, COMPETING_QUESTION, refs)
        unknown = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=resolve_jev_options(
                LCMConfig(
                    jev_candidate_rerank_enabled=True,
                    jev_candidate_rerank_mode="definitely-not-a-mode",
                ),
                transport=transport,
            ),
        )
    finally:
        engine._store.close()
    assert transport.calls == []
    # An unrecognized mode resolves to ``off``, and ``off`` means the seam is not
    # even entered: no record, no context, byte-identical to an absent option.
    assert "jev" not in unknown
    assert _stable(unknown) == _stable(absent)


def test_resolved_disabled_options_report_a_disabled_record(tmp_path):
    from hermes_lcm.jev_candidates import rerank_candidates

    transport = SyntheticJevTransport()
    record = rerank_candidates(
        COMPETING_QUESTION,
        contract=_single_slot_contract("number"),
        candidates=[],
        options=resolve_jev_options(LCMConfig(), transport=transport),
        reason_code="ambiguous_scalar_candidates",
    )
    assert record["status"] == "disabled"
    assert record["reason_code"] == "feature_disabled"
    assert record["mode"] == "off"
    assert record["calls"] == 0
    assert record["applied"] is False
    assert record["advisory_applied"] is False
    assert transport.calls == []


def test_limits_are_clamped_and_revalidated_field_by_field():
    widened = JevLimits(
        max_slots=99,
        max_candidates_per_slot=99,
        max_candidate_chars=10**6,
        max_question_chars=-5,
        max_state_chars=10**9,
        max_advisory_candidates=99,
        max_calls=7,
        timeout_seconds=1000.0,
        model="",
    )
    resolved = JevLimits.from_mapping(widened)
    assert resolved.max_slots == 8
    assert resolved.max_candidates_per_slot == 8
    assert resolved.max_candidate_chars == 2000
    assert resolved.max_question_chars == 32
    assert resolved.max_state_chars == 48000
    assert resolved.max_advisory_candidates == 16
    assert resolved.max_calls == 1
    assert resolved.timeout_seconds == 8.0
    assert resolved.model == JEV_DEFAULT_MODEL

    for bad in ({}, None, 42, {"timeout_seconds": float("nan")}, {"max_calls": True}):
        fallback = JevLimits.from_mapping(bad)
        assert fallback == JevLimits()
        assert fallback.max_calls == 1
        assert fallback.max_advisory_candidates == 8


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("ambiguous_scalar_candidates", True),
        ("ambiguous_fact_candidates", True),
        ("ambiguous_answer", True),
        ("ambiguous_unanchored_operands", False),
        ("scalar_not_found", False),
        ("", False),
        (None, False),
    ],
)
def test_only_code_owned_ambiguous_reasons_are_eligible(reason, expected):
    assert eligible_rerank_reason(reason) is expected


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def test_request_carries_question_semantics_and_only_code_owned_candidate_ids(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        result = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(SyntheticJevTransport(), mode="shadow"),
        )
    finally:
        engine._store.close()

    record = result["jev"]
    assert record["status"] == "ok"
    assert record["slot_options"]["answer"] == ["c01", "c02"]
    request = record["request"]
    assert request["state"]["question"] == COMPETING_QUESTION
    assert request["state"]["question_clipped"] is False
    assert request["state"]["contract"]["operation"] == "scalar"
    assert request["state"]["contract"]["requested_unit"] == "point"
    assert request["model"] == JEV_DEFAULT_MODEL
    assert list(request["state"]["candidates"]) == ["c01", "c02"]
    assert request["state"]["candidates"]["c01"]["text"] == "15 points"
    assert request["state"]["candidates"]["c02"]["text"] == "20 points"
    # The provider can only answer with a code-owned id or the abstain sentinel.
    criteria = request["questions"]["slot:answer"]["criteria"]
    assert set(criteria) == {JEV_ABSTAIN_OPTION, "c01", "c02"}
    assert request["questions"]["slot:answer"]["type"] == "choice"
    assert record["request_sha256"]
    assert record["mode"] == "shadow"


def test_wire_request_is_serializable_and_deterministic(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        options = _jev_options(SyntheticJevTransport(), mode="shadow")
        first = _compile(engine, COMPETING_QUESTION, refs, jev=options)["jev"]
        second = _compile(engine, COMPETING_QUESTION, refs, jev=options)["jev"]
    finally:
        engine._store.close()
    assert first["request"] == second["request"]
    assert first["request_sha256"] == second["request_sha256"]
    json.dumps(first["request"], ensure_ascii=False, sort_keys=True)


def test_build_rerank_request_returns_only_the_wire_payload(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        from hermes_lcm.answer_contract import compile_answer_contract
        from hermes_lcm.requirements_compiler import _candidate_inventory, _normalize_ref

        contract = compile_answer_contract(COMPETING_QUESTION, None).contract
        hydrated = [
            _normalize_ref(item, engine=engine, origin="baseline") for item in refs
        ]
        candidates = _candidate_inventory(hydrated, contract, limit=48)
        groups, metrics = build_slot_options(candidates, contract)
    finally:
        engine._store.close()
    assert metrics == {"omitted_candidates": 0, "omission_reasons": {}}
    payload = build_rerank_request(
        COMPETING_QUESTION, contract=contract, slot_options=groups
    )
    assert set(payload) == {"state", "model", "questions"}


def test_question_is_refused_not_clipped_when_over_the_bound(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        result = _compile(
            engine,
            COMPETING_QUESTION + " " + ("context " * 200),
            refs,
            jev=_jev_options(transport, mode="shadow"),
        )
    finally:
        engine._store.close()
    assert result["jev"]["status"] == "scope_blocked"
    assert result["jev"]["reason_code"] == "question_too_long"
    assert transport.calls == []


def test_empty_question_is_refused_before_dispatch():
    group = [
        {
            "exact_ref": "lcm:1:0-4",
            "quote": "text",
            "value": "a",
            "slot_names": ["answer"],
        }
    ]
    with pytest.raises(JevScopeError) as excinfo:
        prepare_rerank_request(
            "   ", contract=_single_slot_contract(), slot_options={"answer": group}
        )
    assert excinfo.value.reason_code == "empty_question"


# --------------------------------------------------------------------------
# Provider-free egress scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "my api_key: abcdef1234",
        "Authorization: Bearer abcdefghijklmnop",
        "use sk-abcdefghijklmnop please",
        "password = hunter2hunter2",
    ],
)
def test_secret_like_question_never_leaves_the_process(secret, tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        result = _compile(
            engine,
            f"How many points do I need to redeem the reward? {secret}",
            refs,
            jev=_jev_options(transport, mode="advisory"),
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "scope_blocked"
    assert record["reason_code"] == "secret_like_content"
    assert record["request"] is None
    assert transport.calls == []


def test_secret_like_candidate_quote_is_refused_at_the_request_boundary():
    group = [
        {
            "exact_ref": "lcm:1:0-24",
            "quote": "token: abcdef1234 leaked",
            "value": "leaked",
            "slot_names": ["answer"],
        }
    ]
    with pytest.raises(JevScopeError) as excinfo:
        prepare_rerank_request(
            "Which value is correct?",
            contract=_single_slot_contract(),
            slot_options={"answer": group},
        )
    assert excinfo.value.reason_code == "secret_like_content"


def test_secret_like_row_never_becomes_an_admitted_candidate(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    secret_like = _append(
        engine, "You need 30 points: token=abcdefgh1234 to redeem the reward."
    )
    transport = SyntheticJevTransport()
    try:
        result = _compile(
            engine,
            COMPETING_QUESTION,
            [*refs, secret_like],
            jev=_jev_options(transport, mode="advisory"),
        )
    finally:
        engine._store.close()
    assert "abcdefgh1234" not in json.dumps(result, default=str)
    assert result["jev"]["status"] in {"ok", "refused", "skipped"}
    assert all(
        "abcdefgh1234" not in str(item.get("quote") or "")
        for item in result["evidence"]
    )
    if isinstance(result.get("context"), str):
        assert "abcdefgh1234" not in result["context"]


def test_state_budget_exceeded_blocks_the_egress(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        result = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(transport, mode="shadow", limits={"max_state_chars": 512}),
        )
    finally:
        engine._store.close()
    assert result["jev"]["status"] == "scope_blocked"
    assert result["jev"]["reason_code"] == "state_budget_exceeded"
    assert transport.calls == []


def test_credential_is_read_at_call_time_and_never_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        missing = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(transport, mode="shadow")
        )
    finally:
        engine._store.close()
    assert missing["jev"]["status"] == "fallback"
    assert missing["jev"]["reason_code"] == "missing_api_key"
    assert missing["jev"]["calls"] == 0
    assert transport.calls == []
    assert "synthetic-not-a-credential" not in json.dumps(missing["jev"], default=str)


# --------------------------------------------------------------------------
# Transport bounds and failure paths
# --------------------------------------------------------------------------


def test_redirects_are_refused_by_the_real_transport_opener():
    from hermes_lcm.jev_candidates import _NoRedirectHandler

    handler = _NoRedirectHandler()
    assert handler.redirect_request(None, None, 302, "", {}, "http://elsewhere") is None


def test_synthetic_transport_deadline_is_enforced(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport(delay_seconds=0.4)
    try:
        result = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(transport, mode="shadow", limits={"timeout_seconds": 0.1}),
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "fallback"
    assert record["reason_code"] == "deadline_exceeded"
    assert record["calls"] == 1
    assert record["usage"]["input_tokens"] is None
    assert result["context"] is None


@pytest.mark.parametrize(
    ("transport_kwargs", "expected"),
    [
        ({"error": OSError("connection refused")}, "transport_error:OSError"),
        ({"status": 500, "raw_body": "boom"}, "http_500"),
        ({"status": 401, "raw_body": "denied"}, "http_401"),
        ({"raw_body": "{not json"}, "malformed_body"),
        ({"body": []}, "malformed_body"),
        ({"body": {"model": "m"}}, "missing_answers"),
        ({"body": {"answers": {}}}, "answer_set_mismatch"),
        (
            {"body": {"answers": {"slot:answer": {"type": "choice", "choice": "c99"}}}},
            "unknown_choice_option",
        ),
        (
            {"body": {"answers": {"slot:answer": {"type": "noul", "noul": 0.9}}}},
            "unexpected_answer_type",
        ),
        ({"body": {"answers": {"slot:answer": "c01"}}}, "malformed_answer"),
        ({"extra_answer_key": "slot:extra"}, "answer_set_mismatch"),
    ],
)
def test_provider_failures_preserve_the_baseline(tmp_path, transport_kwargs, expected):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport(**transport_kwargs)
    try:
        baseline = _compile(engine, COMPETING_QUESTION, refs)
        result = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(transport, mode="advisory")
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "fallback"
    assert record["reason_code"] == expected
    assert record["applied"] is False
    assert record["advisory_applied"] is False
    assert result["context"] is None
    assert _deterministic(result) == _deterministic(baseline)


def test_provider_abstention_preserves_the_baseline(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport(oracle={"slot:answer": JEV_ABSTAIN_OPTION})
    try:
        baseline = _compile(engine, COMPETING_QUESTION, refs)
        result = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(transport, mode="advisory")
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "abstained"
    assert record["reason_code"] == "abstained"
    assert record["choices"] == {}
    assert record["advisory_applied"] is False
    assert result["context"] is None
    assert _deterministic(result) == _deterministic(baseline)


def test_confidence_and_probabilities_are_structural_only(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        bad_confidence = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(
                SyntheticJevTransport(extra_answer_fields={"confidence": "high"}),
                mode="shadow",
            ),
        )
        bad_probabilities = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(
                SyntheticJevTransport(probabilities={"c01": 0.5, "c02": "high"}),
                mode="shadow",
            ),
        )
        good = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(
                SyntheticJevTransport(
                    extra_answer_fields={"confidence": 0.75},
                    probabilities={"c01": 0.25, "c02": 0.75},
                    usage={"input_tokens": 10, "output_tokens": 2},
                ),
                mode="shadow",
            ),
        )
    finally:
        engine._store.close()
    assert bad_confidence["jev"]["reason_code"] == "malformed_confidence"
    assert bad_probabilities["jev"]["reason_code"] == "malformed_probabilities"
    assert good["jev"]["confidence_by_question"] == {"slot:answer": 0.75}
    assert good["jev"]["probabilities_by_question"] == {
        "slot:answer": {"c01": 0.25, "c02": 0.75}
    }
    assert good["jev"]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": None,
    }


def test_default_http_transport_bounds_the_success_body(monkeypatch):
    from hermes_lcm.jev_candidates import JEV_MAX_RESPONSE_BYTES

    captured: dict = {}

    class _Response:
        status = 200
        headers: dict = {}

        def read(self, limit):
            captured["limit"] = limit
            return b"x" * (JEV_MAX_RESPONSE_BYTES + 1)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Opener:
        def open(self, request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return _Response()

    monkeypatch.setattr("hermes_lcm.jev_candidates._JEV_OPENER", _Opener())
    with pytest.raises(JevTransportError) as excinfo:
        default_http_transport(
            url=JEV_ENDPOINT,
            payload={"a": 1},
            headers={"Authorization": "Bearer x"},
            timeout=1.0,
        )
    assert str(excinfo.value) == "response_too_large"
    assert captured["limit"] == JEV_MAX_RESPONSE_BYTES + 1
    assert captured["url"] == JEV_ENDPOINT


# --------------------------------------------------------------------------
# Deterministic authority is preserved
# --------------------------------------------------------------------------


def test_semantic_uniqueness_proof_refuses_an_ambiguous_pool(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        from hermes_lcm.answer_contract import compile_answer_contract
        from hermes_lcm.requirements_compiler import _candidate_inventory, _normalize_ref

        contract = compile_answer_contract(COMPETING_QUESTION, None).contract
        hydrated = [
            _normalize_ref(item, engine=engine, origin="baseline") for item in refs
        ]
        candidates = _candidate_inventory(hydrated, contract, limit=48)
        groups, _ = build_slot_options(candidates, contract)
        proof = semantic_uniqueness_proof(
            groups,
            {"answer": "c01"},
            contract=contract,
            option_ids_by_slot={"answer": ["c01", "c02"]},
        )
    finally:
        engine._store.close()
    assert proof["proven"] is False
    assert proof["reason_code"] == "semantic_uniqueness_unproven"
    assert proof["distinct_keys_by_slot"] == {"answer": 2}
    assert proof["suppressed_by_slot"] == {"answer": 1}


def test_applied_mode_preflight_still_refuses_before_any_dispatch(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        baseline = _compile(engine, COMPETING_QUESTION, refs)
        result = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(transport, mode="applied")
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "refused"
    assert record["reason_code"] == "semantic_uniqueness_unproven"
    assert record["calls"] == 0
    assert record["shadow_decision"] is None
    assert record["applied"] is False
    assert transport.calls == []
    assert _without_jev(result) == _stable(baseline)


def test_shadow_mode_records_the_choice_without_delivering_it(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport(oracle={"slot:answer": "c02"})
    try:
        baseline = _compile(engine, COMPETING_QUESTION, refs)
        result = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(transport, mode="shadow")
        )
    finally:
        engine._store.close()
    record = result["jev"]
    assert record["status"] == "ok"
    assert record["choices"] == {"answer": "c02"}
    assert record["applied"] is False
    assert record["advisory_applied"] is False
    assert result["context"] is None
    assert _deterministic(result) == _deterministic(baseline)
    assert len(transport.calls) == 1
    assert transport.calls[0]["url"] == JEV_ENDPOINT
    assert transport.calls[0]["has_authorization"] is True
    assert transport.calls[0]["header_names"] == ["Authorization", "Content-Type"]


def test_ineligible_reason_skips_without_a_call(tmp_path):
    engine = _engine(tmp_path)
    ref = _append(engine, "Please redeem the reward whenever it suits you.")
    transport = SyntheticJevTransport()
    try:
        baseline = _compile(engine, COMPETING_QUESTION, [ref])
        result = _compile(
            engine, COMPETING_QUESTION, [ref], jev=_jev_options(transport, mode="advisory")
        )
    finally:
        engine._store.close()
    assert baseline["reason_code"] == "scalar_not_found"
    record = result["jev"]
    assert record["status"] == "skipped"
    assert record["reason_code"] == "reason_not_rerank_eligible"
    assert transport.calls == []
    assert _deterministic(result) == _deterministic(baseline)


def test_single_semantic_group_slot_skips_without_a_call(tmp_path):
    from hermes_lcm.jev_candidates import rerank_candidates

    transport = SyntheticJevTransport()
    contract = _single_slot_contract("number")
    pool = [
        {
            "exact_ref": "lcm:1:0-6",
            "quote": "15 pts",
            "value": 15,
            "unit": "point",
            "slot_names": ["answer"],
        },
        {
            "exact_ref": "lcm:2:0-6",
            "quote": "15 pts",
            "value": 15,
            "unit": "point",
            "slot_names": ["answer"],
        },
    ]
    record = rerank_candidates(
        COMPETING_QUESTION,
        contract=contract,
        candidates=pool,
        options=_jev_options(transport, mode="advisory"),
        reason_code="ambiguous_scalar_candidates",
    )
    assert record["status"] == "skipped"
    assert record["reason_code"] == "no_ambiguous_slot"
    assert transport.calls == []


def test_options_disabled_performs_no_work(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    transport = SyntheticJevTransport()
    try:
        absent = _compile(engine, COMPETING_QUESTION, refs)
        off = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=resolve_jev_options(LCMConfig(), transport=transport),
        )
    finally:
        engine._store.close()
    assert transport.calls == []
    assert off["context"] is None
    assert "jev" not in off
    assert _stable(off) == _stable(absent)


def test_a_hung_attempt_bounds_the_next_attempt(tmp_path):
    """The reviewed single-worker cap is retained: a hang cannot accumulate.

    A timed-out attempt leaves its worker alive until the transport returns, and
    the next attempt is refused rather than spawning another daemon thread.  The
    refusal is a fallback with zero extra calls, so the caller still sees the
    baseline.
    """
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    slow = SyntheticJevTransport(delay_seconds=0.5)
    second = SyntheticJevTransport()
    try:
        first = _compile(
            engine,
            COMPETING_QUESTION,
            refs,
            jev=_jev_options(slow, mode="shadow", limits={"timeout_seconds": 0.1}),
        )
        assert first["jev"]["reason_code"] == "deadline_exceeded"
        assert live_worker_count() == 1

        blocked = _compile(
            engine, COMPETING_QUESTION, refs, jev=_jev_options(second, mode="shadow")
        )
    finally:
        engine._store.close()
    assert blocked["jev"]["status"] == "fallback"
    assert blocked["jev"]["reason_code"] == "attempt_in_progress"
    # Reviewed dispatch accounting: the refused attempt is counted as one attempt
    # (the recorded ``calls`` field), while no transport of its own was invoked.
    assert blocked["jev"]["calls"] == 1
    assert second.calls == []
    assert blocked["context"] is None


def test_absent_jev_options_change_nothing(tmp_path):
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        baseline = _compile(engine, COMPETING_QUESTION, refs)
        explicit_none = _compile(engine, COMPETING_QUESTION, refs, jev=None)
    finally:
        engine._store.close()
    assert "jev" not in baseline
    assert "jev" not in explicit_none
    assert _stable(baseline) == _stable(explicit_none)


# --------------------------------------------------------------------------
# Option-id mapping regression
# --------------------------------------------------------------------------


def test_a_choice_numbered_against_the_request_pool_cannot_be_replayed_over_all_groups(
    tmp_path,
):
    """Pin the reviewed numbering mismatch the advisory path must not reuse.

    ``prepare_rerank_request`` numbers the slots it is handed, while
    ``apply_slot_choices`` rebuilds the numbering over every group in contract
    order.  When the request pool is a strict subset of the groups the two
    numberings disagree, so a wire choice replays onto the wrong candidate and
    revalidation fails closed.  The advisory resolves its choice through the
    request's own index instead.
    """
    engine = _engine(tmp_path)
    refs = _competing_scalars(engine)
    try:
        from hermes_lcm.answer_contract import compile_answer_contract
        from hermes_lcm.requirements_compiler import _candidate_inventory, _normalize_ref

        contract = compile_answer_contract(COMPETING_QUESTION, None).contract
        hydrated = [
            _normalize_ref(item, engine=engine, origin="baseline") for item in refs
        ]
        candidates = _candidate_inventory(hydrated, contract, limit=48)
    finally:
        engine._store.close()

    # A two-slot contract whose leading slot has one semantic group, so the
    # ambiguity-only request pool contains the trailing ``answer`` slot alone.
    multi_contract = SimpleNamespace(
        slots=[
            SimpleNamespace(name="entity", value_type="text"),
            SimpleNamespace(name="answer", value_type="number"),
        ]
    )
    pool = [
        {
            "exact_ref": "lcm:1:0-21",
            "quote": "one entity reference",
            "value": "same",
            "slot_names": ["entity"],
        },
        {
            "exact_ref": "lcm:2:0-25",
            "quote": "another entity reference",
            "value": "same",
            "slot_names": ["entity"],
        },
        {
            "exact_ref": candidates[0]["exact_ref"],
            "quote": candidates[0]["quote"],
            "value": candidates[0]["value"],
            "unit": candidates[0].get("unit"),
            "slot_names": ["answer"],
        },
        {
            "exact_ref": candidates[1]["exact_ref"],
            "quote": candidates[1]["quote"],
            "value": candidates[1]["value"],
            "unit": candidates[1].get("unit"),
            "slot_names": ["answer"],
        },
    ]
    groups, _ = build_slot_options(pool, multi_contract)
    choosable = {
        name: group
        for name, group in groups.items()
        if len({str(item.get("value")) for item in group}) >= 2
    }
    assert list(groups) == ["entity", "answer"]
    assert list(choosable) == ["answer"]

    prepared = prepare_rerank_request(
        COMPETING_QUESTION, contract=multi_contract, slot_options=choosable
    )
    # The request numbers only the pool it sends: ``answer`` starts at c01.
    assert prepared["slot_options"] == {"answer": ["c01", "c02"]}

    # apply_slot_choices renumbers from the full group set, so c02 there is the
    # second *entity* candidate and revalidation refuses the whole choice.
    assert apply_slot_choices(pool, multi_contract, {"answer": "c02"}) is None
    assert apply_slot_choices(pool, multi_contract, {"answer": "c04"}) is not None
