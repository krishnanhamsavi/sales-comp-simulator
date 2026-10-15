"""Participation and engagement: who took part and how payouts were spread.

These are descriptive, not causal. They describe the program, not the reps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MeasureConfig
from ..data.loader import PanelData
from .cohorts import Cohorts


@dataclass(frozen=True)
class GroupRow:
    group: str
    value: str
    eligible: int
    participants: int
    participation_rate: float
    payout: float


@dataclass(frozen=True)
class Engagement:
    eligible_reps: int
    participants: int
    participation_rate: float
    payout_concentration: float          # share of payout earned by the top X of participants
    concentration_top: float
    concentration_curve: tuple           # (share of participants, share of payout) points, top earners first
    median_days_to_first_payout: float
    median_payout_per_participant: float
    total_payout: float                  # actual spend in the window, all participants
    budget_utilisation: float
    by_group: tuple                      # GroupRow


def engagement(data: PanelData, c: Cohorts, cfg: MeasureConfig) -> Engagement:
    p = c.program
    pay = data.payouts
    pay = pay[(pay.program_id == p.program_id) & (pay.period >= p.start) & (pay.period <= p.end)]
    spend = float(pay.payout_amount.sum())
    pay = pay[pay.rep_id.isin(set(c.eligible))]
    per_rep = pay.groupby("rep_id").payout_amount.sum().sort_values(ascending=False)
    n_part = len(per_rep)
    total = per_rep.sum()
    k = int(np.ceil(cfg.concentration_top * n_part)) if n_part else 0
    conc = float(per_rep.iloc[:k].sum() / total) if total else 0.0
    cum = np.r_[0.0, per_rep.cumsum().to_numpy() / total] if total else np.array([0.0])
    shares = np.r_[0.0, np.arange(1, n_part + 1) / n_part] if n_part else np.array([0.0])
    step = max(1, n_part // 40)
    curve = tuple((round(float(s), 4), round(float(v), 4)) for s, v in
                  list(zip(shares, cum))[::step] + [(shares[-1], cum[-1])])
    first = pay.groupby("rep_id").period.min()
    days = (first - p.start).dt.days
    reps = data.reps.set_index("rep_id").loc[list(c.eligible)]
    rows = []
    for col in ("segment", "region"):
        for v, grp in reps.groupby(col):
            part = [r for r in grp.index if r in per_rep.index]
            rows.append(GroupRow(group=col, value=str(v), eligible=len(grp), participants=len(part),
                                 participation_rate=len(part) / len(grp) if len(grp) else 0.0,
                                 payout=float(per_rep.reindex(part).sum())))
    return Engagement(
        eligible_reps=len(c.eligible), participants=n_part,
        participation_rate=n_part / len(c.eligible) if c.eligible else 0.0,
        payout_concentration=conc, concentration_top=cfg.concentration_top, concentration_curve=curve,
        median_days_to_first_payout=float(days.median()) if len(days) else float("nan"),
        median_payout_per_participant=float(per_rep.median()) if n_part else 0.0,
        total_payout=spend, budget_utilisation=spend / p.budget if p.budget else float("nan"),
        by_group=tuple(rows))
