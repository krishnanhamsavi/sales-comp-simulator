"""Validation against planted ground truth.

    python -m spiff_effectiveness.validate --runs 100

This harness is the only code outside generate/ that reads the planted
effect. It regenerates the panel under many seeds, runs the unmodified
pipeline, and checks the estimator against the answer it was never shown.
Results are written to docs/validation.json for the README.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import CANNIBALISATION, DEFAULT, TRUE_EFFECT, MeasureConfig
from .data.loader import from_tables
from .generate import programs as G
from .measure import did
from .report import assemble

PROGRAM = "SP-01"
SEED0 = 1000


@dataclass(frozen=True)
class Run:
    seed: int
    status: str
    true_att_total: float
    true_att_target: float
    att_total: float
    total_ci: tuple
    att_target: float
    target_ci: tuple
    total_significant: bool
    placebo_passed: bool
    placebo_significant: bool
    balance_passed: bool
    naive_treated_vs_untreated: float

    @property
    def target_covered(self) -> bool:
        return self.target_ci[0] <= self.true_att_target <= self.target_ci[1]

    @property
    def total_covered(self) -> bool:
        return self.total_ci[0] <= self.true_att_total <= self.total_ci[1]


def one_run(seed: int, true_effect: float = TRUE_EFFECT, cfg: MeasureConfig = DEFAULT) -> Run:
    tables = G.build(seed=seed, true_effect=true_effect, cannibalisation=CANNIBALISATION)
    data = from_tables(tables)
    core = assemble.measure(data, PROGRAM, cfg)
    p = core.cohorts.program
    panel = did.build_panel(data, core.match.matched_treated, core.match.matched_controls, p.start, p.end,
                            cfg.pre_periods)
    n = len(panel[(panel.treated == 1) & (panel.post == 1)])
    truth = tables.truth[(tables.truth.program_id == PROGRAM) & tables.truth.rep_id.isin(core.match.matched_treated)]
    d = core.did
    return Run(seed=seed, status=assemble.build(data, core, cfg).status,
               true_att_total=float(truth.lift_total.sum() / n), true_att_target=float(truth.lift_target.sum() / n),
               att_total=d.total.beta, total_ci=(d.total.ci_low, d.total.ci_high),
               att_target=d.target.beta, target_ci=(d.target.ci_low, d.target.ci_high),
               total_significant=d.total.significant, placebo_passed=core.placebo.passed,
               placebo_significant=bool(core.placebo.available and not core.placebo.passed),
               balance_passed=core.match.balance_passed,
               naive_treated_vs_untreated=core.naive.treated_vs_untreated)


def simulate(runs: int, true_effect: float = TRUE_EFFECT, seed0: int = SEED0) -> list[Run]:
    return [one_run(seed0 + i, true_effect) for i in range(runs)]


def summarise(effect_runs: list[Run], null_runs: list[Run]) -> dict:
    n, m = len(effect_runs), len(null_runs)
    return {
        "program": PROGRAM, "runs": n, "planted_effect": TRUE_EFFECT, "planted_cannibalisation": CANNIBALISATION,
        "recovery_target_ci_covers_truth": sum(r.target_covered for r in effect_runs),
        "recovery_total_ci_covers_truth": sum(r.total_covered for r in effect_runs),
        "power_total_significant": sum(r.total_significant for r in effect_runs),
        "power_target_significant": sum(r.target_ci[0] > 0 for r in effect_runs),
        "null_runs": m,
        "null_total_not_significant": sum(not r.total_significant for r in null_runs),
        "placebo_not_significant": sum(not r.placebo_significant for r in effect_runs),
        "balance_passed": sum(r.balance_passed for r in effect_runs),
        "mean_true_att_total": sum(r.true_att_total for r in effect_runs) / n,
        "mean_att_total": sum(r.att_total for r in effect_runs) / n,
        "mean_true_att_target": sum(r.true_att_target for r in effect_runs) / n,
        "mean_att_target": sum(r.att_target for r in effect_runs) / n,
        "mean_naive_treated_vs_untreated": sum(r.naive_treated_vs_untreated for r in effect_runs) / n,
        "per_run": [{"seed": r.seed, "true_att_target": round(r.true_att_target, 2),
                     "att_target": round(r.att_target, 2), "target_ci": [round(x, 2) for x in r.target_ci],
                     "true_att_total": round(r.true_att_total, 2), "att_total": round(r.att_total, 2),
                     "total_ci": [round(x, 2) for x in r.total_ci]} for r in effect_runs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "validation.json"))
    a = ap.parse_args()
    s = summarise(simulate(a.runs), simulate(a.runs, true_effect=0.0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=2))
    print({k: v for k, v in s.items() if k != "per_run"})


if __name__ == "__main__":
    main()
