"""Steps 4 and 6: parallel trends, event study, and the placebo window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..config import MeasureConfig
from ..data.loader import PanelData
from . import did, features, matching
from .cohorts import Cohorts


@dataclass(frozen=True)
class TrendPoint:
    rel_month: int
    treated_mean: float
    control_mean: float


@dataclass(frozen=True)
class EventCoef:
    rel_month: int
    beta: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class ParallelTrends:
    passed: bool
    joint_f: float
    joint_p: float
    n_significant_pre: int      # individual pre-period coefficients with p < alpha (informational)
    coefs: tuple                # EventCoef, reference month -1 omitted
    trend: tuple                # TrendPoint, cohort means by month relative to launch


@dataclass(frozen=True)
class Placebo:
    available: bool
    passed: bool
    window_start: str
    window_end: str
    estimate: did.Estimate | None
    n_pairs: int


def cohort_trends(panel: pd.DataFrame) -> tuple:
    g = panel.groupby(["rel", "treated"]).bookings.mean().unstack()
    return tuple(TrendPoint(int(r), float(g.loc[r, 1.0]), float(g.loc[r, 0.0])) for r in g.index)


def event_study(panel: pd.DataFrame, cfg: MeasureConfig) -> ParallelTrends:
    """y_it = a_i + g_t + sum_{k != -1} d_k (Treated_i x 1[rel_t = k]).

    The PRD asks for every pre-period coefficient to be indistinguishable
    from zero. Testing five coefficients one at a time would fail a valid
    design about 23% of the time at alpha = 0.05, so the verdict uses the
    joint Wald test that all pre-period coefficients are zero. Individual
    coefficients and CIs are still reported."""
    rels = sorted(r for r in panel.rel.unique() if r != -1)
    D = np.column_stack([(panel.treated * (panel.rel == k)).to_numpy(float) for k in rels])
    fit = did.twfe(panel, ["bookings"], D)
    dfr = fit.n_clusters - 1
    tcrit = stats.t.ppf(1 - cfg.alpha / 2, dfr)
    coefs = []
    for i, k in enumerate(rels):
        b, se = fit.coef[0, i], float(np.sqrt(max(fit.cov[i, i], 0)))
        p = float(2 * stats.t.sf(abs(b / se), dfr)) if se > 0 else 1.0
        coefs.append(EventCoef(int(k), float(b), float(b - tcrit * se), float(b + tcrit * se), p))
    pre = [i for i, k in enumerate(rels) if k < -1]
    if pre:
        b = fit.coef[0, pre]
        V = fit.cov[np.ix_(pre, pre)]
        f = float(b @ np.linalg.pinv(V) @ b / len(pre))
        p_joint = float(stats.f.sf(f, len(pre), dfr))
    else:
        f, p_joint = 0.0, 1.0
    n_sig = sum(1 for i in pre if coefs[i].p_value < cfg.alpha)
    return ParallelTrends(passed=p_joint >= cfg.alpha, joint_f=f, joint_p=p_joint, n_significant_pre=n_sig,
                          coefs=tuple(coefs), trend=cohort_trends(panel))


def placebo(data: PanelData, c: Cohorts, cfg: MeasureConfig) -> Placebo:
    """Re-run the whole estimator (features, matching, DiD) with a fake launch
    placed entirely before the real one, where the true effect is zero.

    The fake window is the last stretch of the real pre-period, the same
    length as the program. If participation was driven by performance in
    those same months, the placebo fires even when the real estimate is
    sound, so a failure here is read as "the cohorts were already diverging
    right before launch", which is exactly the risk it exists to catch."""
    p = c.program
    fake_start = p.start - pd.DateOffset(months=p.months)
    fake_end = p.start - pd.DateOffset(months=1)
    label = (f"{fake_start:%Y-%m}", f"{fake_end:%Y-%m}")
    if fake_start - pd.DateOffset(months=cfg.pre_periods) < data.bookings.period.min():
        return Placebo(False, True, *label, None, 0)
    reps = list(c.treated) + list(c.controls)
    f = features.pre_period_features(data, reps, fake_start, cfg)
    f = f.dropna()
    t = tuple(r for r in c.treated if r in f.index)
    k = tuple(r for r in c.controls if r in f.index)
    m = matching.match(f, t, k, cfg)
    if len(m.pairs) < cfg.min_matched:
        return Placebo(False, True, *label, None, len(m.pairs))
    panel = did.build_panel(data, m.matched_treated, m.matched_controls, fake_start, fake_end, cfg.pre_periods)
    est = did.estimate(panel, cfg).total
    return Placebo(True, not est.significant, *label, est, len(m.pairs))
