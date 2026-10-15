"""Sales force and simulation assumptions for the forward-looking plan simulator.

All values are synthetic and chosen to be plausible for a B2B hardware and
subscription sales team. They are inputs, not measurements.
"""
from __future__ import annotations

SEED = 11
N_REPS = 320
N_SIMS = 400                     # Monte Carlo draws of a plan year

SEGMENTS = {
    #  segment:      share, OTE,     pay mix (variable share of OTE), annual quota
    "Enterprise":  (0.20, 300_000, 0.50, 1_600_000),
    "Mid-Market":  (0.35, 200_000, 0.45, 900_000),
    "SMB":         (0.45, 140_000, 0.40, 450_000),
}

# Annual quota attainment is lognormal with a median just above 100%, as in
# most healthy plans where roughly 55-65% of reps reach quota.
ATTAINMENT_MEDIAN = {"Enterprise": 1.04, "Mid-Market": 1.05, "SMB": 1.06}
ATTAINMENT_SIGMA = {"Enterprise": 0.34, "Mid-Market": 0.30, "SMB": 0.27}

# Rep-level persistent skill shares a part of attainment across simulated years.
SKILL_SHARE = 0.6

# Attainment shift scenarios shown in the sensitivity table.
SHIFTS = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
