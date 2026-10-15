# Synthetic rep panel: how it is generated

`python -m spiff_effectiveness.generate.programs` writes four Parquet tables to
`spiff_effectiveness/data/parquet/`, plus the planted ground truth in a separate
`_truth/` folder that the analysis code never reads. Every constant below is
named in [`generate/panel.py`](../spiff_effectiveness/generate/panel.py),
[`generate/programs.py`](../spiff_effectiveness/generate/programs.py) or
[`config.py`](../spiff_effectiveness/config.py).

## Tables

| Table | Grain | Columns |
|---|---|---|
| `reps` | rep | rep_id, segment, region, tenure_months, quota_annual, latent_skill (never read by the estimator) |
| `bookings_monthly` | rep × month | rep_id, period, bookings, deals_closed, pipeline_created, product_mix_pct_focus |
| `spiff_programs` | program | program_id, name, start_period, end_period, eligibility_rule, mechanic, rate, budget, target_product |
| `spiff_payouts` | program × rep × month | program_id, rep_id, period, qualifying_units, payout_amount |

There are 320 reps over 24 months (2024-01 to 2025-12), about 7,100 rep-months. The panel is
unbalanced: about 12% of reps join mid-panel (up to month 18) and about 8% leave early.

## The shapes that make the problem hard

| Shape | How | Why it matters |
|---|---|---|
| **Selection** | Participation probability is a logistic function of latent skill (never observed) plus prior-year attainment (SP-01) or recent momentum (SP-02) | The naive participant-vs-non-participant comparison is badly inflated: about $12K vs. a true effect of about $1K |
| **Planted effect** | Treated reps get `TRUE_EFFECT = 0.08` multiplicative lift on target-product bookings, from the month they engage to the window end | Known ground truth for validation |
| **Cannibalisation** | `CANNIBALISATION = 0.30` of the target lift is removed from other products | Total-bookings lift is smaller than target-product lift |
| **Trend and seasonality** | +0.6% per month shared growth; month-in-quarter multipliers 0.88 / 0.97 / 1.30 | A before/after comparison without controls is wrong |
| **Rep dynamics** | Persistent rep-specific drift (SD 0.3% per month) plus a monthly random walk (SD 1.2%) and lognormal noise (SD 8%) | Reps do not move in lockstep; selecting on momentum breaks parallel trends |
| **Churn and ramp** | Joiners ramp from 45% to full productivity over about 6 months; leavers exit mid-panel | Unbalanced panel; ramp growth is a confounder the design has to handle |

## Reps

| Segment | Share | Annual quota | Avg deal | Target-product share of bookings |
|---|---|---|---|---|
| Enterprise | 20% | $1.6M | $60K | 30% |
| Mid-Market | 35% | $0.9M | $25K | 32% |
| SMB | 45% | $0.45M | $9K | 34% |

Regions are AMER 50%, EMEA 30% and APAC 20%. Tenure at panel start follows a gamma
distribution, capped at 1–84 months. Latent skill is N(0, 1) and moves log bookings by
0.18 per SD.

## Programs

| | SP-01 | SP-02 | SP-03 |
|---|---|---|---|
| Window | 2025-10 to 2025-12 | 2024-10 to 2024-12 | 2025-04 to 2025-06 |
| Eligibility | SMB, Mid-Market | Enterprise, Mid-Market | APAC, tenure < 24 months |
| Mechanic | PerUnit, $30 per $1,200 unit | FlatBounty, $275 per attach deal | TieredAccelerator, $35 per unit, 1.5x above 12 units/month |
| Selection on | skill 0.7, prior-year attainment 0.5 | skill 0.3, recent momentum 2.2 | skill 0.8, attainment 0.5 |
| Designed to | be measurable | fail the design checks | lack overlap |

Engaged reps start earning in the first month of the window, or later (more often for reps
with under 12 months' tenure). The planted lift starts from that month.

## Seeds

- The default seed is 10, chosen by the rule described in the README.
- Validation uses seeds 1000–1099.
- The selection, effect and cannibalisation parameters are identical for every seed; only the random draws change.
