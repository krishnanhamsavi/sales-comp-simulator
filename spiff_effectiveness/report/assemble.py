"""Builds one frozen ProgramResult: the single source of every reported number.

The Streamlit page and the memo narrator read this object and format it.
Neither computes anything. Headline estimates are withheld (set to None)
whenever the design fails a diagnostic, so no surface can show them by
accident.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, replace

import numpy as np

from ..config import MeasureConfig
from ..data.loader import PanelData
from ..measure import cohorts as co
from ..measure import diagnostics, did, economics, engagement, features, matching

STATUS_OK = "ok"
STATUS_UNRELIABLE = "unreliable"
STATUS_NO_OVERLAP = "insufficient_overlap"


@dataclass(frozen=True)
class ProgramSetup:
    program_id: str
    name: str
    window_start: str
    window_end: str
    months: int
    eligibility_rule: str
    mechanic: str
    rate: float
    budget: float
    target_product: str
    eligible_reps: int
    treated_reps: int
    control_pool: int
    excluded_ineligible: int
    excluded_contaminated: int
    excluded_thin_history: int
    excluded_ramping: int


@dataclass(frozen=True)
class MatchSummary:
    method: str
    caliper_logit: float
    matched_pairs: int
    dropped_treated: int
    balance: tuple
    balance_passed: bool


@dataclass(frozen=True)
class Estimates:
    att_total: did.Estimate
    att_target: did.Estimate
    cannibalisation_rate: did.Estimate
    n_obs: int
    n_reps: int


@dataclass(frozen=True)
class NaiveReads:
    treated_vs_untreated: float     # window mean bookings, treated minus untreated eligible reps
    before_after: float             # treated reps, window mean minus pre-period mean


@dataclass(frozen=True)
class ProgramResult:
    setup: ProgramSetup
    status: str
    status_reasons: tuple
    matching: MatchSummary
    parallel_trends: diagnostics.ParallelTrends | None
    placebo: diagnostics.Placebo | None
    estimates: Estimates | None
    naive: NaiveReads | None
    economics: economics.Economics | None
    engagement: engagement.Engagement
    assumptions: dict

    @property
    def reliable(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        return _clean(dataclasses.asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, 6)
    return x


@dataclass(frozen=True)
class Core:
    """Everything that does not depend on the margin assumption. Cached by the app."""
    cohorts: co.Cohorts
    match: matching.MatchResult
    did: did.DidResult | None
    trends: diagnostics.ParallelTrends | None
    placebo: diagnostics.Placebo | None
    naive: NaiveReads | None
    engagement: engagement.Engagement


def measure(data: PanelData, program_id: str, cfg: MeasureConfig) -> Core:
    c = co.define_cohorts(data, program_id, cfg)
    feats = features.pre_period_features(data, list(c.treated) + list(c.controls), c.program.start, cfg)
    m = matching.match(feats, c.treated, c.controls, cfg)
    eng = engagement.engagement(data, c, cfg)
    if len(m.pairs) < cfg.min_matched:
        return Core(c, m, None, None, None, None, eng)
    panel = did.build_panel(data, m.matched_treated, m.matched_controls, c.program.start, c.program.end,
                            cfg.pre_periods)
    d = did.estimate(panel, cfg)
    trends = diagnostics.event_study(panel, cfg)
    plac = diagnostics.placebo(data, c, cfg)
    naive = NaiveReads(**did.naive_comparisons(data, c.treated, c.controls, c.program.start, c.program.end,
                                               cfg.pre_periods))
    return Core(c, m, d, trends, plac, naive, eng)


def build(data: PanelData, core: Core, cfg: MeasureConfig) -> ProgramResult:
    c, m, p = core.cohorts, core.match, core.cohorts.program
    setup = ProgramSetup(
        program_id=p.program_id, name=p.name, window_start=f"{p.start:%Y-%m}", window_end=f"{p.end:%Y-%m}",
        months=p.months, eligibility_rule=p.eligibility_rule, mechanic=p.mechanic, rate=p.rate, budget=p.budget,
        target_product=p.target_product, eligible_reps=len(c.eligible), treated_reps=len(c.treated),
        control_pool=len(c.controls), excluded_ineligible=c.ineligible, excluded_contaminated=len(c.contaminated),
        excluded_thin_history=len(c.thin_history), excluded_ramping=len(c.ramping))
    msum = MatchSummary(method=m.method, caliper_logit=m.caliper, matched_pairs=len(m.pairs),
                        dropped_treated=len(m.dropped_treated), balance=m.balance, balance_passed=m.balance_passed)
    assumptions = {k: v for k, v in dataclasses.asdict(cfg).items()}

    if core.did is None:
        plural = "" if len(m.pairs) == 1 else "s"
        reasons = (f"Only {len(m.pairs)} treated rep{plural} found a comparable control inside the caliper; "
                   f"at least {cfg.min_matched} are needed for a credible estimate.",)
        return ProgramResult(setup, STATUS_NO_OVERLAP, reasons, msum, None, None, None, None, None,
                             core.engagement, assumptions)

    reasons = []
    if not m.balance_passed:
        bad = [b.feature for b in m.balance if not b.passed]
        reasons.append(f"Matching left {len(bad)} feature(s) imbalanced above {cfg.smd_threshold:.2f} SMD: "
                       + ", ".join(bad) + ".")
    if not core.trends.passed:
        reasons.append(f"Pre-period trends differ between cohorts (joint test p = {core.trends.joint_p:.3f}), "
                       "so the parallel trends assumption is not supported.")
    if core.placebo.available and not core.placebo.passed:
        reasons.append(f"The placebo window {core.placebo.window_start} to {core.placebo.window_end}, where the "
                       f"true effect is zero, returned a significant effect (p = {core.placebo.estimate.p_value:.3f}).")
    if reasons:
        return ProgramResult(setup, STATUS_UNRELIABLE, tuple(reasons), msum, core.trends, core.placebo, None,
                             None, None, core.engagement, assumptions)

    est = Estimates(att_total=core.did.total, att_target=core.did.target,
                    cannibalisation_rate=core.did.cannibalisation, n_obs=core.did.n_obs, n_reps=core.did.n_reps)
    econ = economics.program_economics(data, p, len(c.treated), core.did, cfg)
    return ProgramResult(setup, STATUS_OK, (), msum, core.trends, core.placebo, est, core.naive, econ,
                         core.engagement, assumptions)


def run(data: PanelData, program_id: str, cfg: MeasureConfig) -> ProgramResult:
    return build(data, measure(data, program_id, cfg), cfg)


def with_margin(cfg: MeasureConfig, margin: float) -> MeasureConfig:
    return replace(cfg, gross_margin=margin)
