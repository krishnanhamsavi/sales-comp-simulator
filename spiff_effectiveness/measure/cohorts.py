"""Step 1: treated, eligible-control and excluded reps for one program."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import MeasureConfig
from ..data.loader import PanelData


@dataclass(frozen=True)
class ProgramSpec:
    program_id: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    months: int
    eligibility_rule: str
    mechanic: str
    rate: float
    budget: float
    target_product: str


@dataclass(frozen=True)
class Cohorts:
    program: ProgramSpec
    eligible: tuple            # eligible and active in the window
    treated: tuple             # eligible with >= 1 payout in the window
    controls: tuple            # eligible, untreated, uncontaminated, with enough pre-period history
    ineligible: int            # fail the rule: dropped entirely, never used as controls
    contaminated: tuple        # paid by another program during this program's analysis span
    thin_history: tuple        # fewer than half the pre-period months observed
    ramping: tuple             # hired too recently: new-hire ramp growth would masquerade as an effect


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def program_spec(data: PanelData, program_id: str) -> ProgramSpec:
    r = data.programs.set_index("program_id").loc[program_id]
    return ProgramSpec(program_id=program_id, name=r["name"], start=r.start_period, end=r.end_period,
                       months=months_between(r.start_period, r.end_period) + 1,
                       eligibility_rule=r.eligibility_rule, mechanic=r.mechanic, rate=float(r.rate),
                       budget=float(r.budget), target_product=r.target_product)


def analysis_span(p: ProgramSpec, cfg: MeasureConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest month any step reads (the placebo's own pre-period) to the window end."""
    return p.start - pd.DateOffset(months=cfg.pre_periods + p.months), p.end


def define_cohorts(data: PanelData, program_id: str, cfg: MeasureConfig) -> Cohorts:
    p = program_spec(data, program_id)
    b = data.bookings
    in_window = b[(b.period >= p.start) & (b.period <= p.end)].rep_id.unique()
    rule_ok = set(data.reps.query(p.eligibility_rule).rep_id)
    eligible = sorted(rule_ok & set(in_window))

    pay = data.payouts
    window_pay = pay[(pay.program_id == program_id) & (pay.period >= p.start) & (pay.period <= p.end)]
    treated_all = set(window_pay.rep_id) & set(eligible)

    s0, s1 = analysis_span(p, cfg)
    other = pay[(pay.program_id != program_id) & (pay.period >= s0) & (pay.period <= s1)]
    contaminated = sorted(set(other.rep_id) & set(eligible))

    pre0 = p.start - pd.DateOffset(months=cfg.pre_periods)
    pre_n = b[(b.period >= pre0) & (b.period < p.start)].groupby("rep_id").size()
    thin = sorted(r for r in eligible if pre_n.get(r, 0) < (cfg.pre_periods + 1) // 2)

    # New hires grow fast while they ramp, for reasons unrelated to any SPIFF.
    # Reps whose first month in the panel is after the panel start and less
    # than ramp_months before the pre-period are excluded from both cohorts.
    first = b.groupby("rep_id").period.min()
    panel0 = b.period.min()
    ramp_cut = pre0 - pd.DateOffset(months=cfg.ramp_months)
    ramping = sorted(r for r in eligible if first[r] > panel0 and first[r] > ramp_cut and r not in thin)

    drop = set(contaminated) | set(thin) | set(ramping)
    treated = sorted(treated_all - drop)
    controls = sorted(set(eligible) - treated_all - drop)
    return Cohorts(program=p, eligible=tuple(eligible), treated=tuple(treated), controls=tuple(controls),
                   ineligible=len(data.reps) - len(rule_ok), contaminated=tuple(contaminated),
                   thin_history=tuple(thin), ramping=tuple(ramping))
