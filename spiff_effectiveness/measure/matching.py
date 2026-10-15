"""Step 3: propensity score matching and the balance table."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import MeasureConfig
from .features import LABELS, feature_columns


@dataclass(frozen=True)
class BalanceRow:
    feature: str
    smd_before: float
    smd_after: float
    passed: bool


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple                 # ((treated_rep, control_rep), ...)
    dropped_treated: tuple       # treated reps with no control inside the caliper
    caliper: float               # in logit units
    method: str
    balance: tuple               # BalanceRow per feature
    balance_passed: bool
    n_treated_pool: int
    n_control_pool: int

    @property
    def matched_treated(self) -> tuple:
        return tuple(t for t, _ in self.pairs)

    @property
    def matched_controls(self) -> tuple:
        return tuple(c for _, c in self.pairs)


def fit_logit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 100) -> np.ndarray:
    """Logistic regression by IRLS with a small ridge penalty (features are
    standardised first), which keeps the fit stable under near-separation."""
    Xc = np.c_[np.ones(len(X)), X]
    beta = np.zeros(Xc.shape[1])
    pen = np.eye(Xc.shape[1]) * ridge
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(Xc @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = Xc.T @ (Xc * W[:, None]) + pen
        step = np.linalg.solve(H, Xc.T @ (y - p) - pen @ beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def smd(x_t: np.ndarray, x_c: np.ndarray, sd_pool: float) -> float:
    """Standardised mean difference. The denominator is the pooled SD of the
    full treated and control pools (before matching), so before and after
    values are on the same scale (Austin 2009)."""
    return float((x_t.mean() - x_c.mean()) / sd_pool) if sd_pool > 0 and len(x_t) and len(x_c) else 0.0


def _refine(pairs: list, Xt: np.ndarray, Xc: np.ndarray, sd_pool: np.ndarray, allowed: np.ndarray,
            max_passes: int = 25) -> list:
    """Swap pass: replace a pair's control with an unused control that is
    also allowed for that treated rep (inside its caliper and segment) when
    the swap lowers the worst |SMD|, then the sum of squared SMDs. Pairs
    are never dropped, and every swap keeps the caliper, so this only ever
    improves on the greedy match."""
    ti = np.array([i for i, _ in pairs])
    cj = np.array([j for _, j in pairs])
    used = np.zeros(len(Xc), dtype=bool)
    used[cj] = True
    n = len(pairs)
    st = Xt[ti].sum(axis=0)
    sc = Xc[cj].sum(axis=0)

    def score(sum_c):
        d = np.abs((st - sum_c) / n / sd_pool)
        return d.max(), (d ** 2).sum()

    best = score(sc)
    for _ in range(max_passes):
        improved = False
        for k in range(n):
            cand = np.flatnonzero(allowed[ti[k]] & ~used)
            if not len(cand):
                continue
            trial = sc[None, :] - Xc[cj[k]] + Xc[cand]
            d = np.abs((st[None, :] - trial) / n / sd_pool)
            keys = list(zip(d.max(axis=1), (d ** 2).sum(axis=1)))
            m = min(range(len(cand)), key=lambda q: (keys[q], cand[q]))
            if keys[m] < best:
                used[cj[k]] = False
                cj[k] = cand[m]
                used[cj[k]] = True
                sc = trial[m]
                best = keys[m]
                improved = True
        if not improved:
            break
    return list(zip(ti.tolist(), cj.tolist()))


def match(features: pd.DataFrame, treated: tuple, controls: tuple, cfg: MeasureConfig) -> MatchResult:
    cols = feature_columns(features.loc[list(treated) + list(controls)])
    X = features.loc[list(treated) + list(controls), cols].to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    y = np.r_[np.ones(len(treated)), np.zeros(len(controls))]
    beta = fit_logit(Z, y)
    logit = np.c_[np.ones(len(Z)), Z] @ beta
    nt = len(treated)
    lt, lc = logit[:nt], logit[nt:]
    caliper = cfg.caliper_sd * float(np.std(logit, ddof=1))
    Xt, Xc = X[:nt], X[nt:]
    sd_pool = np.sqrt((Xt.var(axis=0, ddof=1) + Xc.var(axis=0, ddof=1)) / 2)
    sd_pool[~(sd_pool > 0)] = 1.0

    # Which controls each treated rep may be paired with: inside the caliper on
    # the logit of the propensity score and, for the default method, in the
    # same segment.
    allowed = np.abs(lt[:, None] - lc[None, :]) <= caliper
    seg_cols = [k for k, c in enumerate(cols) if c.startswith("segment=")]
    if cfg.match_method == "mahalanobis" and seg_cols:
        allowed &= (Xt[:, None, seg_cols] == Xc[None, :, seg_cols]).all(axis=2)

    # Greedy 1:1 without replacement, hardest-to-match first (highest
    # propensity), ties broken by rep_id so the result is deterministic.
    # "pscore" takes the nearest logit (the textbook rule); "mahalanobis"
    # takes the nearest control on the features among the allowed ones
    # (Rubin & Thomas 2000), then runs the balance-improving swap pass.
    cov_inv = np.linalg.pinv(np.cov(Z, rowvar=False))
    Zt, Zc = Z[:nt], Z[nt:]
    order = sorted(range(nt), key=lambda i: (-lt[i], treated[i]))
    available = np.ones(len(controls), dtype=bool)
    idx_pairs, dropped = [], []
    for i in order:
        if cfg.match_method == "mahalanobis":
            diff = Zc - Zt[i]
            d = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        else:
            d = np.abs(lc - lt[i])
        d = np.where(allowed[i] & available, d, np.inf)
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            idx_pairs.append((i, j))
            available[j] = False
        else:
            dropped.append(treated[i])
    if cfg.match_method == "mahalanobis" and idx_pairs:
        idx_pairs = _refine(idx_pairs, Xt, Xc, sd_pool, allowed)
    pairs = sorted((treated[i], controls[j]) for i, j in idx_pairs)

    ti = [i for i, _ in idx_pairs]
    cj = [j for _, j in idx_pairs]
    rows = []
    for k, c in enumerate(cols):
        before = smd(Xt[:, k], Xc[:, k], sd_pool[k])
        after = smd(Xt[ti, k], Xc[cj, k], sd_pool[k]) if idx_pairs else float("nan")
        rows.append(BalanceRow(feature=LABELS.get(c, c.replace("=", ": ")), smd_before=round(before, 6),
                               smd_after=round(after, 6), passed=bool(abs(after) < cfg.smd_threshold)))
    return MatchResult(pairs=tuple(pairs), dropped_treated=tuple(sorted(dropped)), caliper=caliper,
                       method=cfg.match_method, balance=tuple(rows), balance_passed=all(r.passed for r in rows),
                       n_treated_pool=nt, n_control_pool=len(controls))
