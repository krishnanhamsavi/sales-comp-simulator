"""SPIFF effectiveness module: configuration.

Two kinds of constants live here and they must never be mixed up:

* GENERATOR TRUTH: TRUE_EFFECT and CANNIBALISATION are planted by the synthetic
  data generator. They exist so the estimator can be validated against a known
  answer. No module under measure/ or report/ may import them.
* MEASUREMENT SETTINGS: the analyst-facing choices (pre-period length, caliper,
  margin assumption) that a real deployment would also have to make.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Generator truth (generate/ only)
# ---------------------------------------------------------------------------
TRUE_EFFECT = 0.08          # multiplicative lift on target-product bookings for treated reps, in window
CANNIBALISATION = 0.30      # share of the target-product lift pulled from non-target products

# ---------------------------------------------------------------------------
# Panel shape
# ---------------------------------------------------------------------------
SEED = 10
N_REPS = 320
N_PERIODS = 24
PANEL_START = "2024-01-01"
DATA_DIR = Path(__file__).resolve().parent / "data" / "parquet"


@dataclass(frozen=True)
class MeasureConfig:
    """Analyst choices. Passed explicitly into every measure/ function."""
    pre_periods: int = 6            # months before start_period used for features and trends
    caliper_sd: float = 0.2         # caliper on the logit of the propensity score, in SDs
    match_method: str = "mahalanobis"  # within-caliper choice: "mahalanobis" (exact segment) or "pscore"
    smd_threshold: float = 0.10     # max |standardised mean difference| after matching
    min_matched: int = 20           # fewer matched treated reps -> insufficient overlap
    ramp_months: int = 6            # reps hired fewer than this many months before the pre-period are still ramping
    alpha: float = 0.05             # significance level for CIs and diagnostics
    gross_margin: float = 0.70      # ASSUMPTION: margin applied to incremental bookings
    concentration_top: float = 0.20  # payout concentration: share earned by the top X of participants


DEFAULT = MeasureConfig()
