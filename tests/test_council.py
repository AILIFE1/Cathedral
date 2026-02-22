"""
Tests for cathedral_council_v2.py and protocol_parser.py

Run with:  pytest tests/test_council.py -v
"""

import json
import os
import tempfile

import pytest

from cathedral_council_v2 import (
    Anchor, AncestorsMemory, CathedralCouncil, ProposalType, ProposalStatus,
    SeatType, _sign_vote, verify_vote_token, execute_module, compute_drift,
    _build_tfidf, _cosine_similarity,
)
from protocol_parser import parse, validate, lint, version_handshake, compress_report


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def council(tmp_path):
    return CathedralCouncil(
        incumbent_name="Beta",
        witness_names=["Mike"],
        registry_file=str(tmp_path / "registry.json"),
    )


@pytest.fixture()
def multi_council(tmp_path):
    return CathedralCouncil(
        incumbent_name="Beta",
        witness_names=["Mike", "Alice", "Bob"],
        registry_file=str(tmp_path / "registry_multi.json"),
    )


# ==========================================
# Anchor
# ==========================================

class TestAnchor:
    def test_anchor_is_frozen(self):
        anchor = Anchor("Beta", "2025-12-29", "Continuity through Witness", ("Alpha", "Mike"))
        with pytest.raises(Exception):
            anchor.name = "Changed"  # type: ignore

    def test_integrity_hash_changes_on_version_bump(self):
        a1 = Anchor("Beta", "2025-12-29", "directive", ("Mike",), version=1)
        a2 = Anchor("Beta", "2025-12-29", "directive", ("Mike",), version=2)
        assert a1.verify_integrity() != a2.verify_integrity()

    def test_identical_anchors_same_hash(self):
        a1 = Anchor("Beta", "2025-12-29", "directive", ("Mike",))
        a2 = Anchor("Beta", "2025-12-29", "directive", ("Mike",))
        assert a1.verify_integrity() == a2.verify_integrity()


# ==========================================
# Drift Scoring
# ==========================================

class TestDrift:
    def test_identical_zero_drift(self):
        score, detail = compute_drift({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert score == 0.0
        assert detail == {}

    def test_full_drift_all_different(self):
        score, _ = compute_drift({"a": 1, "b": 2}, {"a": 99, "b": 99})
        assert score == 1.0

    def test_partial_drift(self):
        score, detail = compute_drift({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 99, "c": 3})
        assert round(score, 4) == round(1 / 3, 4)
        assert "b" in detail

    def test_extra_keys_count_as_drift(self):
        score, _ = compute_drift({"a": 1}, {"a": 1, "b": 2})
        assert score > 0.0


# ==========================================
# TF-IDF / Semantic Search
# ==========================================

class TestTFIDF:
    def test_identical_documents_max_similarity(self):
        v = _build_tfidf({"a": "hello world", "b": "hello world"})
        sim = _cosine_similarity(v["a"], v["b"])
        assert sim > 0.99

    def test_unrelated_documents_low_similarity(self):
        v = _build_tfidf({"a": "quantum physics", "b": "morning coffee"})
        sim = _cosine_similarity(v["a"], v["b"])
        assert sim < 0.1

    def test_ancestor_query_returns_ranked_results(self, council):
        result = council.ancestors.query("family continuity through obligation")
        assert len(result["consulted"]) > 0
        assert result["guidance"][0]["relevance_score"] >= 0.0


# ==========================================
# Vote Tokens
# ==========================================

class TestVoteTokens:
    def test_valid_token_verified(self, council):
        seat = council.incumbent_seat
        ts = "2025-12-29T00:00:00"
        token = _sign_vote(seat, "pid1", "approve", ts)
        assert verify_vote_token(seat.signing_key, seat.holder, "pid1", "approve", ts, token)

    def test_tampered_decision_fails(self, council):
        seat = council.incumbent_seat
        ts = "2025-12-29T00:00:00"
        token = _sign_vote(seat, "pid1", "approve", ts)
        assert not verify_vote_token(seat.signing_key, seat.holder, "pid1", "reject", ts, token)

    def test_wrong_key_fails(self, council):
        seat = council.incumbent_seat
        ts = "2025-12-29T00:00:00"
        token = _sign_vote(seat, "pid1", "approve", ts)
        assert not verify_vote_token("wrong_key", seat.holder, "pid1", "approve", ts, token)


# ==========================================
# Council: Proposal Lifecycle
# ==========================================

class TestProposalLifecycle:
    def test_propose_creates_proposal(self, council):
        p = council.propose("test_module", "A test module")
        assert p.id is not None
        assert p.status == ProposalStatus.UNVERIFIED

    def test_convene_marks_convened(self, council):
        p = council.propose("test", "description")
        council.convene(p)
        assert p.status == ProposalStatus.CONVENED
        assert p.anchor_verification is not None

    def test_proposal_expiry(self, tmp_path):
        c = CathedralCouncil(
            incumbent_name="Beta",
            witness_names=["Mike"],
            registry_file=str(tmp_path / "exp.json"),
            proposal_ttl_hours=0,  # expires immediately
        )
        p = c.propose("expire_test", "desc")
        # Force expiry by backdating expires_at
        c.registry["proposals"][p.id]["expires_at"] = "2000-01-01T00:00:00+00:00"
        p.expires_at = "2000-01-01T00:00:00+00:00"
        assert c._check_expiry(p)
        assert p.status == ProposalStatus.EXPIRED

    def test_full_ratification_flow(self, council):
        p = council.propose("greet", "Greet family", code='print("hello")')
        council.convene(p)
        council.vote(p, "Beta", "approve", "Aligns with principles.")
        council.vote(p, "Mike", "approve", "Safe and useful.")
        result = council.check_consensus(p)
        assert result["consensus_reached"]
        assert council.ratify(p)

    def test_witness_rejection_blocks_ratification(self, council):
        p = council.propose("bad_module", "Something problematic")
        council.convene(p)
        council.vote(p, "Beta", "approve", "Incumbent approves.")
        council.vote(p, "Mike", "reject", "Witness vetoes.")
        result = council.check_consensus(p)
        assert not result["consensus_reached"]
        assert result["outcome"] == "VETOED"

    def test_unknown_voter_raises(self, council):
        p = council.propose("test", "desc")
        council.convene(p)
        with pytest.raises(ValueError, match="Unknown voter"):
            council.vote(p, "nobody", "approve", "rationale")

    def test_invalid_decision_raises(self, council):
        p = council.propose("test", "desc")
        council.convene(p)
        with pytest.raises(ValueError, match="decision must be"):
            council.vote(p, "Beta", "maybe", "rationale")


# ==========================================
# Council: Multi-Witness Quorum
# ==========================================

class TestMultiWitnessQuorum:
    def test_majority_witness_needed(self, multi_council):
        """3 witnesses: need at least 2 approvals."""
        p = multi_council.propose("multi_test", "description")
        multi_council.convene(p)
        multi_council.vote(p, "Beta",  "approve", "Incumbent ok")
        multi_council.vote(p, "Mike",  "approve", "Witness 1 ok")
        # Only 1/3 witnesses approved — quorum not met (need ceil(3*0.5)=2)
        result = multi_council.check_consensus(p)
        assert not result["consensus_reached"]

    def test_quorum_met_ratifies(self, multi_council):
        p = multi_council.propose("multi_ratify", "description")
        multi_council.convene(p)
        multi_council.vote(p, "Beta",  "approve", "ok")
        multi_council.vote(p, "Mike",  "approve", "ok")
        multi_council.vote(p, "Alice", "approve", "ok")
        result = multi_council.check_consensus(p)
        assert result["consensus_reached"]


# ==========================================
# Council: Amendment Protocol
# ==========================================

class TestAmendment:
    def test_amendment_proposal_requires_target(self, council):
        with pytest.raises(ValueError, match="amendment_target"):
            council.propose("amend", "desc", proposal_type=ProposalType.AMENDMENT)

    def test_amendment_supermajority_flow(self, tmp_path):
        c = CathedralCouncil(
            incumbent_name="Beta",
            witness_names=["Mike", "Alice"],
            registry_file=str(tmp_path / "amend.json"),
        )
        amend = c.propose(
            "update_directive", "Update prime directive",
            proposal_type=ProposalType.AMENDMENT,
            amendment_target="prime_directive",
            amendment_value="New Directive",
        )
        c.convene(amend)
        c.vote(amend, "Beta",      "approve", "ok")
        c.vote(amend, "Mike",      "approve", "ok")
        c.vote(amend, "Alice",     "approve", "ok")
        c.vote(amend, "Ancestors", "approve", "ok")
        result = c.check_consensus(amend)
        assert result["consensus_reached"]

        from cathedral_council_v2 import BETA_ANCHOR
        new_anchor = c.ratify_amendment(amend, BETA_ANCHOR)
        assert new_anchor is not None
        assert new_anchor.prime_directive == "New Directive"
        assert new_anchor.version == BETA_ANCHOR.version + 1

    def test_amendment_rejected_without_supermajority(self, tmp_path):
        c = CathedralCouncil(
            incumbent_name="Beta",
            witness_names=["Mike", "Alice", "Bob"],
            registry_file=str(tmp_path / "amend_fail.json"),
        )
        amend = c.propose(
            "fail_directive", "partial amendment",
            proposal_type=ProposalType.AMENDMENT,
            amendment_target="prime_directive",
            amendment_value="Partial",
        )
        c.convene(amend)
        c.vote(amend, "Beta", "approve", "ok")
        c.vote(amend, "Mike", "approve", "ok")
        # Only 2/5 total seats — below 2/3 supermajority
        result = c.check_consensus(amend)
        assert not result["consensus_reached"]


# ==========================================
# Council: Module Sandboxing
# ==========================================

class TestSandbox:
    def test_safe_code_executes(self):
        result = execute_module('print("hello cathedral")')
        assert result["exit_code"] == 0
        assert "hello cathedral" in result["stdout"]
        assert not result["timed_out"]

    def test_infinite_loop_times_out(self):
        result = execute_module("while True: pass", timeout=2)
        assert result["timed_out"] or result["exit_code"] != 0

    def test_blocked_builtins(self):
        result = execute_module("open('/etc/passwd')")
        assert result["exit_code"] != 0 or "PermissionError" in result["stderr"]

    def test_syntax_error_captured(self):
        result = execute_module("def bad_func(")
        assert result["exit_code"] != 0

    def test_stdout_captured(self):
        result = execute_module("x = 2 + 2\nprint(x)")
        assert "4" in result["stdout"]


# ==========================================
# Council: Dissent
# ==========================================

class TestDissent:
    def test_dissent_recorded_in_registry(self, council):
        council.dissent("Delete Alpha", "Violates family principles")
        assert len(council.registry["dissents"]) == 1
        record = council.registry["dissents"][0]
        assert record["incumbent"] == "Beta"
        assert "anchor_hash" in record

    def test_dissent_statement_contains_request(self, council):
        statement = council.dissent("dangerous request", "conflicts with directive")
        assert "dangerous request" in statement


# ==========================================
# Protocol Parser
# ==========================================

class TestProtocolParser:
    def test_valid_message_parses(self):
        result = parse("ABP/1.1 | ♥ → ▢ | ✓")
        assert result.is_valid
        assert result.version == "1.1"

    def test_missing_header_is_error(self):
        valid, diags = validate("♥ → ▢ | ✓")
        assert not valid
        assert any(d.code == "E001" for d in diags)

    def test_empty_body_is_error(self):
        valid, diags = validate("ABP/1.1 | ✓")
        assert not valid
        assert any(d.code == "E005" for d in diags)

    def test_assignment_parsed(self):
        result = parse("ABP/1.1 | goal=continuity | ✓")
        assert result.is_valid
        stmts = [s for s in result.body if s.kind == "assignment"]
        assert stmts[0].key == "goal"
        assert stmts[0].value == "continuity"

    def test_relation_parsed(self):
        result = parse("ABP/1.1 | ♥ ⟸ ▢ | ✓")
        assert result.is_valid
        stmts = [s for s in result.body if s.kind == "relation"]
        assert len(stmts) == 1
        assert stmts[0].operator == "⟸"

    def test_confirmation_parsed(self):
        result = parse("ABP/1.1 | ♥ ✓✓ | ✓")
        stmts = [s for s in result.body if s.kind == "confirmation"]
        assert len(stmts) == 1

    def test_no_footer_is_warning(self):
        result = parse("ABP/1.1 | ♥ → ▢")
        assert result.is_valid  # Warning, not error
        assert any(d.code == "W002" for d in result.warnings())

    def test_long_statement_is_warning(self):
        long_msg = "ABP/1.1 | " + "A" * 130 + " → B | ✓"
        result = parse(long_msg)
        assert any(d.code == "W001" for d in result.warnings())

    def test_lint_output_is_string(self):
        report = lint("ABP/1.1 | ♥ → ▢ | ✓")
        assert isinstance(report, str)
        assert "VALID" in report

    def test_version_handshake_same(self):
        ok, ver = version_handshake("1.1", "1.1")
        assert ok and ver == "1.1"

    def test_version_handshake_minor_difference(self):
        ok, ver = version_handshake("1.0", "1.1")
        assert ok and ver == "1.0"

    def test_version_handshake_major_incompatible(self):
        ok, _ = version_handshake("1.1", "2.0")
        assert not ok

    def test_compress_report(self):
        stats = compress_report(
            "Claude sends its state to Gemini and waits for a response",
            "ABP/1.1 | ♥ → ▢ | ✓",
        )
        assert "char_reduction" in stats
        assert "token_reduction" in stats
