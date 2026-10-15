"""Monte Carlo cost engine: what does a plan cost across plausible plan years."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .plans import Plan


def population(seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segs = list(C.SEGMENTS)
    seg = rng.choice(segs, size=C.N_REPS, p=[C.SEGMENTS[s][0] for s in segs])
    df = pd.DataFrame({"rep_id": [f"R{i + 1:03d}" for i in range(C.N_REPS)], "segment": seg})
    df["ote"] = df.segment.map(lambda s: C.SEGMENTS[s][1])
    df["target_incentive"] = df.segment.map(lambda s: C.SEGMENTS[s][1] * C.SEGMENTS[s][2])
    df["quota"] = df.segment.map(lambda s: C.SEGMENTS[s][3]) * rng.uniform(0.9, 1.1, C.N_REPS)
    df["skill"] = rng.normal(0, 1, C.N_REPS)
    return df


def attainment_draws(pop: pd.DataFrame, n_sims: int = C.N_SIMS, shift: float = 0.0,
                     seed: int = C.SEED) -> np.ndarray:
    """(n_sims, n_reps) annual attainment. Persistent skill + yearly noise,
    scaled so each segment's median matches config. `shift` moves every rep's
    attainment by a relative amount (e.g. +0.05 = 5% more bookings)."""
    rng = np.random.default_rng(seed + 1)
    med = pop.segment.map(C.ATTAINMENT_MEDIAN).to_numpy()
    sig = pop.segment.map(C.ATTAINMENT_SIGMA).to_numpy()
    z = (np.sqrt(C.SKILL_SHARE) * pop.skill.to_numpy()[None, :]
         + np.sqrt(1 - C.SKILL_SHARE) * rng.normal(0, 1, (n_sims, len(pop))))
    return med * np.exp(sig * z) * (1 + shift)


@dataclass
class PlanCost:
    plan: Plan
    variable_cost: float          # expected annual variable comp
    variable_cost_p10: float
    variable_cost_p90: float
    bookings: float               # expected annual bookings
    cost_of_sales: float          # (base + variable) / bookings
    variable_rate: float          # variable / bookings
    pct_at_quota: float
    payout_vs_target: float       # variable cost / sum of target incentive
    top_quintile_share: float     # share of variable pay earned by top 20% of reps
    rep_payout_pcts: pd.DataFrame  # per-segment payout distribution

    def row(self) -> dict:
        return {"Plan": self.plan.name, "Expected variable comp": self.variable_cost,
                "P10": self.variable_cost_p10, "P90": self.variable_cost_p90,
                "Payout vs. target": self.payout_vs_target, "Cost of sales": self.cost_of_sales,
                "Variable rate": self.variable_rate, "Reps at quota": self.pct_at_quota,
                "Top 20% share of pay": self.top_quintile_share}


def evaluate(plan: Plan, pop: pd.DataFrame, shift: float = 0.0, n_sims: int = C.N_SIMS) -> PlanCost:
    att = attainment_draws(pop, n_sims, shift)
    ti = pop.target_incentive.to_numpy()
    quota = pop.quota.to_numpy()
    pay = plan.payout_multiple(att) * ti                       # (sims, reps)
    book = att * quota
    base = (pop.ote - pop.target_incentive).sum()
    var_total = pay.sum(axis=1)
    book_total = book.sum(axis=1)
    srt = np.sort(pay, axis=1)[:, ::-1]
    k = int(np.ceil(0.2 * pay.shape[1]))
    top = (srt[:, :k].sum(axis=1) / np.maximum(var_total, 1)).mean()
    pcts = []
    for seg in C.SEGMENTS:
        m = (pop.segment == seg).to_numpy()
        mult = (pay[:, m] / ti[m]).ravel()
        pcts.append({"segment": seg, **{f"p{q}": float(np.percentile(mult, q)) for q in (10, 25, 50, 75, 90)}})
    return PlanCost(
        plan=plan, variable_cost=float(var_total.mean()),
        variable_cost_p10=float(np.percentile(var_total, 10)), variable_cost_p90=float(np.percentile(var_total, 90)),
        bookings=float(book_total.mean()), cost_of_sales=float(((base + var_total) / book_total).mean()),
        variable_rate=float((var_total / book_total).mean()), pct_at_quota=float((att >= 1.0).mean()),
        payout_vs_target=float(var_total.mean() / ti.sum()), top_quintile_share=float(top),
        rep_payout_pcts=pd.DataFrame(pcts))


def sensitivity(plan: Plan, pop: pd.DataFrame, shifts=C.SHIFTS) -> pd.DataFrame:
    """Plan cost under team-wide attainment shifts, and the marginal cost of
    each extra dollar of bookings relative to the no-shift case."""
    base = evaluate(plan, pop, 0.0)
    rows = []
    for s in shifts:
        r = evaluate(plan, pop, s)
        d_book = r.bookings - base.bookings
        rows.append({"Attainment shift": s, "Bookings": r.bookings, "Variable comp": r.variable_cost,
                     "Cost of sales": r.cost_of_sales,
                     "Marginal comp per $1 bookings": (r.variable_cost - base.variable_cost) / d_book
                     if abs(d_book) > 1 else np.nan})
    return pd.DataFrame(rows)
