"""Step 5: two-way fixed effects difference-in-differences.

    y_it = alpha_i + gamma_t + beta * (Treated_i x Post_t) + e_it

Implemented directly in NumPy (rep and period dummies, OLS, cluster-robust
sandwich by rep) rather than through a regression library, so the joint
covariance of estimates across outcomes is available. The cannibalisation
rate's confidence interval needs it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..config import MeasureConfig
from ..data.loader import PanelData


@dataclass(frozen=True)
class Estimate:
    beta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class FitResult:
    coef: np.ndarray            # coefficients of the treatment regressors, per outcome: (n_outcomes, k)
    cov: np.ndarray             # joint covariance of vec(coef): (n_outcomes*k, n_outcomes*k)
    n_obs: int
    n_clusters: int


def build_panel(data: PanelData, treated: tuple, controls: tuple, start: pd.Timestamp, end: pd.Timestamp,
                pre_periods: int) -> pd.DataFrame:
    """Matched rep-months from start - pre_periods to end. Unbalanced: reps keep
    whatever months they were active."""
    b = data.bookings
    pre0 = start - pd.DateOffset(months=pre_periods)
    reps = set(treated) | set(controls)
    df = b[b.rep_id.isin(reps) & (b.period >= pre0) & (b.period <= end)].copy()
    df["treated"] = df.rep_id.isin(set(treated)).astype(float)
    df["post"] = (df.period >= start).astype(float)
    df["rel"] = (df.period.dt.year - start.year) * 12 + df.period.dt.month - start.month
    df["target_bookings"] = df.bookings * df.product_mix_pct_focus
    df["other_bookings"] = df.bookings - df.target_bookings
    return df.sort_values(["rep_id", "period"]).reset_index(drop=True)


def twfe(df: pd.DataFrame, outcomes: list, regressors: np.ndarray) -> FitResult:
    """OLS of each outcome on `regressors` plus rep and period fixed effects,
    with a joint cluster-robust (by rep) covariance across outcomes."""
    reps = pd.Categorical(df.rep_id)
    pers = pd.Categorical(df.period)
    R = np.eye(len(reps.categories))[reps.codes]
    T = np.eye(len(pers.categories))[pers.codes][:, 1:]
    X = np.c_[np.atleast_2d(regressors.T).T, R, T]
    k = regressors.shape[1] if regressors.ndim == 2 else 1
    XtX_inv = np.linalg.pinv(X.T @ X)
    Y = df[outcomes].to_numpy(float)
    B = XtX_inv @ X.T @ Y                                  # (p, n_outcomes)
    E = Y - X @ B
    n, p = X.shape
    G = len(reps.categories)
    # per-cluster scores, stacked across outcomes
    scores = np.zeros((G, p * len(outcomes)))
    for j in range(len(outcomes)):
        s = X * E[:, [j]]
        scores[:, j * p:(j + 1) * p] = pd.DataFrame(s).groupby(reps.codes).sum().to_numpy()
    meat = scores.T @ scores
    bread = np.kron(np.eye(len(outcomes)), XtX_inv)
    corr = G / (G - 1) * (n - 1) / max(n - p, 1)
    V = corr * bread @ meat @ bread
    idx = [j * p + i for j in range(len(outcomes)) for i in range(k)]
    return FitResult(coef=B[:k].T.copy(), cov=V[np.ix_(idx, idx)], n_obs=n, n_clusters=G)


def to_estimate(beta: float, var: float, df_resid: int, alpha: float) -> Estimate:
    se = float(np.sqrt(max(var, 0.0)))
    tcrit = stats.t.ppf(1 - alpha / 2, df_resid)
    p = float(2 * stats.t.sf(abs(beta / se), df_resid)) if se > 0 else 1.0
    return Estimate(beta=float(beta), se=se, ci_low=float(beta - tcrit * se), ci_high=float(beta + tcrit * se),
                    p_value=p, significant=p < alpha)


@dataclass(frozen=True)
class DidResult:
    total: Estimate              # ATT on total bookings, $ per treated rep-month
    target: Estimate             # ATT on target-product bookings
    cannibalisation: Estimate    # 1 - total/target, delta-method CI
    n_obs: int
    n_reps: int
    n_treated: int


def estimate(panel: pd.DataFrame, cfg: MeasureConfig) -> DidResult:
    D = (panel.treated * panel.post).to_numpy()[:, None]
    fit = twfe(panel, ["bookings", "target_bookings"], D)
    dfr = fit.n_clusters - 1
    b_tot, b_tgt = fit.coef[0, 0], fit.coef[1, 0]
    total = to_estimate(b_tot, fit.cov[0, 0], dfr, cfg.alpha)
    target = to_estimate(b_tgt, fit.cov[1, 1], dfr, cfg.alpha)
    if abs(b_tgt) > 0:
        rate = 1 - b_tot / b_tgt
        grad = np.array([-1 / b_tgt, b_tot / b_tgt ** 2])
        cann = to_estimate(rate, float(grad @ fit.cov @ grad), dfr, cfg.alpha)
    else:
        cann = Estimate(np.nan, np.nan, np.nan, np.nan, 1.0, False)
    return DidResult(total=total, target=target, cannibalisation=cann, n_obs=fit.n_obs, n_reps=fit.n_clusters,
                     n_treated=int(panel[panel.treated == 1].rep_id.nunique()))


def naive_comparisons(data: PanelData, treated: tuple, untreated: tuple, start, end, pre_periods: int) -> dict:
    """The two wrong answers the design exists to avoid, on total bookings."""
    b = data.bookings
    win = b[(b.period >= start) & (b.period <= end)]
    pre = b[(b.period >= start - pd.DateOffset(months=pre_periods)) & (b.period < start)]
    t_win = win[win.rep_id.isin(treated)].bookings.mean()
    u_win = win[win.rep_id.isin(untreated)].bookings.mean()
    t_pre = pre[pre.rep_id.isin(treated)].bookings.mean()
    return {"treated_vs_untreated": float(t_win - u_win), "before_after": float(t_win - t_pre)}
