"""Synthetic rep panel: reps and counterfactual monthly bookings.

Counterfactual means "what each rep would have booked with no SPIFF". The
programs step (programs.py) decides who participates and adds the planted
effect on top. Shapes built in here, and why:

* latent_skill drives baseline level and is never shown to the estimator.
* A shared upward trend and a quarter-end spike affect every rep, so a
  before/after comparison without a control group is wrong.
* Each rep has a persistent growth drift plus a monthly random walk, so
  reps do not move in lockstep and selecting on recent momentum can break
  parallel trends.
* Some reps join or leave mid-panel, so the panel is unbalanced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

SEGMENT_W = {"Enterprise": 0.20, "Mid-Market": 0.35, "SMB": 0.45}
REGION_W = {"AMER": 0.50, "EMEA": 0.30, "APAC": 0.20}
QUOTA = {"Enterprise": 1_600_000, "Mid-Market": 900_000, "SMB": 450_000}
AVG_DEAL = {"Enterprise": 60_000, "Mid-Market": 25_000, "SMB": 9_000}
FOCUS_SHARE = {"Enterprise": 0.30, "Mid-Market": 0.32, "SMB": 0.34}   # baseline share in the target product

SKILL_LEVEL = 0.18          # log-bookings per SD of latent skill
TREND = 0.006               # shared monthly growth
QUARTER_SHAPE = {0: 0.88, 1: 0.97, 2: 1.30}   # month-in-quarter multipliers (Jan/Apr.. , Feb.., Mar..)
DRIFT_SD = 0.003            # persistent rep-specific monthly growth
WALK_SD = 0.012             # monthly random walk on log level
NOISE_SD = 0.08             # idiosyncratic monthly noise on log bookings
JOIN_P, LEAVE_P = 0.10, 0.08


def periods() -> pd.DatetimeIndex:
    return pd.date_range(C.PANEL_START, periods=C.N_PERIODS, freq="MS")


def build_reps(rng: np.random.Generator) -> pd.DataFrame:
    n = C.N_REPS
    seg = rng.choice(list(SEGMENT_W), n, p=list(SEGMENT_W.values()))
    reg = rng.choice(list(REGION_W), n, p=list(REGION_W.values()))
    joins = np.where(rng.random(n) < JOIN_P, rng.integers(1, 19, n), 0)
    leaves = np.where(rng.random(n) < LEAVE_P, rng.integers(10, C.N_PERIODS - 1, n), C.N_PERIODS - 1)
    leaves = np.maximum(leaves, joins + 6)
    tenure = np.where(joins > 0, 1, np.clip(np.round(rng.gamma(2.2, 14, n)), 1, 84)).astype(int)
    quota = np.array([QUOTA[s] for s in seg]) * rng.uniform(0.9, 1.1, n)
    return pd.DataFrame({
        "rep_id": [f"R{i + 1:04d}" for i in range(n)], "segment": seg, "region": reg,
        "tenure_months": tenure, "quota_annual": quota.round(-3),
        "latent_skill": rng.normal(0, 1, n),
        # generator-internal columns, dropped before the reps table is written
        "_join": joins, "_leave": np.minimum(leaves, C.N_PERIODS - 1),
        "_drift": rng.normal(0, DRIFT_SD, n),
        "_focus": np.clip([FOCUS_SHARE[s] for s in seg] + rng.normal(0, 0.05, n), 0.12, 0.6),
    })


def counterfactual_bookings(reps: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Long table of rep-months with counterfactual target / non-target bookings."""
    per = periods()
    T = len(per)
    n = len(reps)
    t = np.arange(T)
    season = np.array([QUARTER_SHAPE[(p.month - 1) % 3] for p in per])
    walk = np.cumsum(rng.normal(0, WALK_SD, (n, T)), axis=1)
    base = (reps.quota_annual.to_numpy() / 12)[:, None] * np.exp(SKILL_LEVEL * reps.latent_skill.to_numpy())[:, None]
    # new hires ramp to full productivity over their first six months
    months_in = t[None, :] - reps._join.to_numpy()[:, None] + np.where(reps._join.to_numpy() > 0, 0, 12)[:, None]
    ramp = np.clip(0.45 + 0.11 * months_in, 0.45, 1.0)
    level = (base * ramp * ((1 + TREND) ** t)[None, :] * season[None, :]
             * np.exp(reps._drift.to_numpy()[:, None] * t[None, :] + walk))
    total = level * np.exp(rng.normal(0, NOISE_SD, (n, T)) - NOISE_SD ** 2 / 2)
    share = np.clip(reps._focus.to_numpy()[:, None] + rng.normal(0, 0.03, (n, T)), 0.05, 0.9)

    active = (t[None, :] >= reps._join.to_numpy()[:, None]) & (t[None, :] <= reps._leave.to_numpy()[:, None])
    ii, tt = np.nonzero(active)
    return pd.DataFrame({
        "rep_id": reps.rep_id.to_numpy()[ii], "t": tt, "period": per[tt],
        "target_cf": total[ii, tt] * share[ii, tt], "other_cf": total[ii, tt] * (1 - share[ii, tt]),
    })
