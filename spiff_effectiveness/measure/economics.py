"""Program economics: what the program cost and what the effect was worth.

Every causal quantity is carried with its confidence interval. Interval
bounds are the ATT interval pushed through the same linear formula, which is
exact because each metric is linear in the ATT.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import MeasureConfig
from ..data.loader import PanelData
from .cohorts import ProgramSpec
from .did import DidResult


MARGIN_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


@dataclass(frozen=True)
class Range:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class Economics:
    payout_cost: float
    budget: float
    budget_utilisation: float
    treated_reps: int
    months: int
    gross_margin: float               # ASSUMPTION, from config / sidebar
    incremental_bookings: Range
    incremental_gross_profit: Range
    net_return: Range
    roi: Range
    breakeven_att: float
    breakeven_verdict: str            # "cleared" | "not cleared" | "inconclusive"
    margin_sensitivity: tuple         # (gross margin, net return Range) across plausible margins


def _lin(est, scale: float, shift: float = 0.0) -> Range:
    lo, hi = sorted((est.ci_low * scale + shift, est.ci_high * scale + shift))
    return Range(est.beta * scale + shift, lo, hi)


def program_economics(data: PanelData, p: ProgramSpec, treated_reps: int, d: DidResult,
                      cfg: MeasureConfig) -> Economics:
    pay = data.payouts
    cost = float(pay[(pay.program_id == p.program_id) & (pay.period >= p.start)
                     & (pay.period <= p.end)].payout_amount.sum())
    rep_months = treated_reps * p.months
    m = cfg.gross_margin
    breakeven = cost / (rep_months * m) if rep_months and m else float("inf")
    est = d.total
    if est.ci_low > breakeven:
        verdict = "cleared"
    elif est.ci_high < breakeven:
        verdict = "not cleared"
    else:
        verdict = "inconclusive"
    return Economics(
        payout_cost=cost, budget=p.budget, budget_utilisation=cost / p.budget if p.budget else float("nan"),
        treated_reps=treated_reps, months=p.months, gross_margin=m,
        incremental_bookings=_lin(est, rep_months),
        incremental_gross_profit=_lin(est, rep_months * m),
        net_return=_lin(est, rep_months * m, -cost),
        roi=_lin(est, rep_months * m / cost, -1.0) if cost else Range(float("nan"), float("nan"), float("nan")),
        breakeven_att=breakeven, breakeven_verdict=verdict,
        margin_sensitivity=tuple((round(g, 2), _lin(est, rep_months * g, -cost)) for g in MARGIN_GRID))
