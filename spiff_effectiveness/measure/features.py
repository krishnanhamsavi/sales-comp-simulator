"""Step 2: pre-period matching features, one row per rep.

Only behaviour a real team can observe before launch. latent_skill is not
available here by construction (the loader never selects it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MeasureConfig
from ..data.loader import PanelData

NUMERIC = ["pre_bookings_mean", "pre_bookings_volatility", "pre_deals_mean", "pre_attainment",
           "pre_focus_mix", "tenure_at_start"]
CATEGORICAL = {"segment": ["Enterprise", "Mid-Market", "SMB"], "region": ["AMER", "EMEA", "APAC"]}
LABELS = {"pre_bookings_mean": "Mean monthly bookings", "pre_bookings_volatility": "Bookings volatility",
          "pre_deals_mean": "Mean deals closed", "pre_attainment": "Quota attainment",
          "pre_focus_mix": "Target-product mix", "tenure_at_start": "Tenure (months)"}


def pre_period_features(data: PanelData, reps: list, start: pd.Timestamp, cfg: MeasureConfig) -> pd.DataFrame:
    b = data.bookings
    pre0 = start - pd.DateOffset(months=cfg.pre_periods)
    pre = b[b.rep_id.isin(reps) & (b.period >= pre0) & (b.period < start)]
    g = pre.groupby("rep_id")
    info = data.reps.set_index("rep_id").loc[list(reps)]
    first = b[b.rep_id.isin(reps)].groupby("rep_id").period.min()
    panel0 = b.period.min()
    months = lambda a, z: (z.year - a.year) * 12 + z.month - a.month  # noqa: E731
    tenure = [info.tenure_months[r] + months(first[r], start) if first[r] > panel0
              else info.tenure_months[r] + months(panel0, start) for r in info.index]
    df = pd.DataFrame({
        "pre_bookings_mean": g.bookings.mean(),
        "pre_bookings_volatility": g.bookings.std(ddof=0) / g.bookings.mean(),
        "pre_deals_mean": g.deals_closed.mean(),
        "pre_attainment": g.bookings.mean() * 12 / info.quota_annual,
        "pre_focus_mix": g.product_mix_pct_focus.mean(),
    }).reindex(info.index)
    df["tenure_at_start"] = np.array(tenure, dtype=float)
    for col, levels in CATEGORICAL.items():
        for lv in levels[1:]:
            df[f"{col}={lv}"] = (info[col] == lv).astype(float)
    return df.sort_index()


def feature_columns(df: pd.DataFrame) -> list:
    """Numeric features plus the categorical dummies that actually vary."""
    dummies = [c for c in df.columns if "=" in c and df[c].std() > 0]
    return NUMERIC + dummies
