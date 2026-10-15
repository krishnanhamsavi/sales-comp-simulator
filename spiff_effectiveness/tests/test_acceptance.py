"""PRD acceptance criteria for the SPIFF effectiveness module.

The three 100-run simulation tests are marked `simulation` (about 4 minutes
together). Run everything with `pytest`; skip them locally with
`pytest -m "not simulation"`.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from spiff_effectiveness import validate
from spiff_effectiveness.config import DEFAULT, SEED, MeasureConfig
from spiff_effectiveness.data.loader import from_tables
from spiff_effectiveness.generate import programs as G
from spiff_effectiveness.measure import cohorts, did
from spiff_effectiveness.report import assemble, narrate

PKG = Path(__file__).resolve().parents[1]
RUNS = 100


@pytest.fixture(scope="session")
def default_data():
    return from_tables(G.build(seed=SEED))


@pytest.fixture(scope="session")
def effect_runs():
    return validate.simulate(RUNS)


@pytest.fixture(scope="session")
def null_runs():
    return validate.simulate(RUNS, true_effect=0.0)


# --- simulation criteria -----------------------------------------------------
@pytest.mark.simulation
def test_recovery_ci_contains_true_effect(effect_runs):
    covered = sum(r.target_covered for r in effect_runs)
    assert covered >= 90, f"target-product CI covered the planted effect in {covered}/{RUNS} runs"
    assert sum(r.total_covered for r in effect_runs) >= 90


@pytest.mark.simulation
def test_null_effect_not_significant(null_runs):
    quiet = sum(not r.total_significant for r in null_runs)
    assert quiet >= 90, f"only {quiet}/{RUNS} null runs were non-significant"


@pytest.mark.simulation
def test_placebo_window_not_significant(effect_runs):
    quiet = sum(not r.placebo_significant for r in effect_runs)
    assert quiet >= 95, f"placebo was non-significant in only {quiet}/{RUNS} runs"


# --- default-seed criteria ---------------------------------------------------
def test_selection_bias_is_corrected(default_data):
    """The naive participant-vs-non-participant gap must dwarf the DiD estimate."""
    r = assemble.run(default_data, "SP-01", DEFAULT)
    assert r.status == assemble.STATUS_OK
    naive, att = r.naive.treated_vs_untreated, r.estimates.att_total.beta
    assert naive > 5 * abs(att) and naive - att > 5_000


def test_balance_after_matching(default_data):
    r = assemble.run(default_data, "SP-01", DEFAULT)
    worst = max(abs(b.smd_after) for b in r.matching.balance)
    assert r.matching.balance_passed and worst < 0.10


def test_determinism():
    a = assemble.run(from_tables(G.build(seed=SEED)), "SP-01", DEFAULT).to_json()
    b = assemble.run(from_tables(G.build(seed=SEED)), "SP-01", DEFAULT).to_json()
    assert a.encode() == b.encode()


def test_unbalanced_panel_is_handled(default_data):
    """Reps who join or leave mid-panel stay in the analysis with the months they have."""
    months = default_data.bookings.groupby("rep_id").size()
    assert (months < 24).sum() >= 20
    core = assemble.measure(default_data, "SP-01", DEFAULT)
    p = core.cohorts.program
    panel = did.build_panel(default_data, core.match.matched_treated, core.match.matched_controls, p.start, p.end,
                            DEFAULT.pre_periods)
    full = DEFAULT.pre_periods + p.months
    partial = (panel.groupby("rep_id").size() < full).sum()
    assert partial >= 1, "expected at least one matched rep with a partial history in the analysis window"
    assert core.did is not None and core.did.n_reps == 2 * len(core.match.pairs)


def test_no_match_path_returns_structured_result(default_data):
    r = assemble.run(default_data, "SP-03", DEFAULT)
    assert r.status == assemble.STATUS_NO_OVERLAP
    assert r.estimates is None and r.economics is None and r.naive is None
    assert r.matching.matched_pairs < DEFAULT.min_matched
    assert "comparable control" in r.status_reasons[0]


def test_failed_diagnostics_suppress_headline(default_data):
    r = assemble.run(default_data, "SP-02", DEFAULT)
    assert r.status == assemble.STATUS_UNRELIABLE
    assert r.estimates is None and r.economics is None
    payload = narrate.narration_payload(r)
    assert "result" not in payload and "naive_reads_for_contrast" not in payload


def test_min_matched_threshold_is_respected(default_data):
    strict = MeasureConfig(min_matched=10_000)
    assert assemble.run(default_data, "SP-01", strict).status == assemble.STATUS_NO_OVERLAP


# --- provenance guard --------------------------------------------------------
def test_provenance_guard_rejects_fabricated_number(default_data, tmp_path):
    r = assemble.run(default_data, "SP-01", DEFAULT)
    payload = narrate.narration_payload(r)
    honest = f"Payouts were {payload['spend']['payout_cost']}."
    assert narrate.unverified_numbers(honest, payload) == []

    fabricated = honest + " Incremental bookings were $1,234,567, a 42.4% return."
    calls = []

    def mocked(_payload):
        calls.append(1)
        return fabricated

    log = tmp_path / "guard.jsonl"
    memo = narrate.narrate(r, draft_fn=mocked, log_path=log)
    assert memo.source == "template" and len(calls) == narrate.MAX_ATTEMPTS
    assert all("$1,234,567" in rej and "42.4%" in rej for rej in memo.rejected)
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(entries) == narrate.MAX_ATTEMPTS and all(e["outcome"] == "rejected" for e in entries)
    assert entries[0]["unmatched_numbers"] == ["$1,234,567", "42.4%"]


def test_provenance_guard_accepts_regenerated_clean_draft(default_data, tmp_path):
    r = assemble.run(default_data, "SP-01", DEFAULT)
    drafts = iter(["Cost was $999,999.", "Cost was " + narrate.narration_payload(r)["spend"]["payout_cost"] + "."])
    memo = narrate.narrate(r, draft_fn=lambda _: next(drafts), log_path=tmp_path / "g.jsonl")
    assert memo.source == narrate.MODEL and memo.attempts == 2 and memo.rejected == [["$999,999"]]


def test_template_memos_pass_their_own_guard(default_data):
    for pid in ("SP-01", "SP-02", "SP-03"):
        r = assemble.run(default_data, pid, DEFAULT)
        memo = narrate.narrate(r, draft_fn=None, log_path=None)
        assert narrate.unverified_numbers(memo.markdown, narrate.narration_payload(r)) == []


# --- architecture contract ---------------------------------------------------
def test_estimator_never_reads_planted_truth():
    """measure/, report/ and data/ must not import the generator truth or latent skill."""
    forbidden = {"TRUE_EFFECT", "CANNIBALISATION", "latent_skill", "_truth", "generate"}
    for sub in ("measure", "report", "data"):
        for f in (PKG / sub).glob("*.py"):
            tree = ast.parse(f.read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            names |= {n.module.split(".")[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            hits = (names & forbidden) | {s for s in strings if s in forbidden}
            if f.name == "loader.py":
                hits -= {"generate"}          # ensure_data() may build the dataset if it is missing
            assert not hits, f"{f.relative_to(PKG)} references {hits}"


def test_cohort_rules(default_data):
    c = cohorts.define_cohorts(default_data, "SP-01", DEFAULT)
    assert not set(c.treated) & set(c.controls)
    assert not set(c.contaminated) & (set(c.treated) | set(c.controls))
    reps = default_data.reps.set_index("rep_id")
    assert set(reps.loc[list(c.treated) + list(c.controls)].segment) <= {"SMB", "Mid-Market"}
