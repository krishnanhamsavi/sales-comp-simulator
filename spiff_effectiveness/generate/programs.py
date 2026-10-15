"""SPIFF programs: registry, participation, payouts, and the planted effect.

Three programs are generated, each built to exercise a different path
through the measurement pipeline:

* SP-01 is the clean case: selection on skill and prior attainment, which
  matching and difference-in-differences are designed to correct.
* SP-02 selects reps on recent momentum (growth in the months before launch).
  That breaks parallel trends, and the diagnostics should catch it.
* SP-03 has a tiny eligible pool, so matching cannot find enough controls.

Planted truth: treated reps get a TRUE_EFFECT multiplicative lift on
target-product bookings from the month they engage until the window ends,
and a CANNIBALISATION share of that lift is taken out of non-target products.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C
from . import panel as P

UNIT_PRICE = 1_200           # $ of target-product bookings per qualifying unit
TARGET_PRODUCT = "Access Control"

PROGRAMS = [
    {"program_id": "SP-01", "name": "Q4 Access Control Attach Push", "start": 21, "end": 23,
     "eligibility_rule": "segment in ['SMB', 'Mid-Market']", "mechanic": "PerUnit", "rate": 30.0,
     "budget": 240_000, "selection": {"base": -0.6, "skill": 0.7, "attainment": 0.5, "momentum": 0.0}},
    {"program_id": "SP-02", "name": "Q4 Enterprise Momentum Bounty", "start": 9, "end": 11,
     "eligibility_rule": "segment in ['Enterprise', 'Mid-Market']", "mechanic": "FlatBounty", "rate": 275.0,
     "budget": 150_000, "selection": {"base": -0.6, "skill": 0.3, "attainment": 0.0, "momentum": 2.2}},
    {"program_id": "SP-03", "name": "APAC New-Hire Access Control Sprint", "start": 15, "end": 17,
     "eligibility_rule": "region == 'APAC' and tenure_months < 24", "mechanic": "TieredAccelerator",
     "rate": 35.0, "budget": 40_000, "selection": {"base": 0.0, "skill": 0.8, "attainment": 0.5, "momentum": 0.0}},
]
TIER_UNITS, TIER_MULT = 12, 1.5   # TieredAccelerator: units above 12 in a month pay 1.5x


@dataclass
class Tables:
    reps: pd.DataFrame
    bookings_monthly: pd.DataFrame
    spiff_programs: pd.DataFrame
    spiff_payouts: pd.DataFrame
    truth: pd.DataFrame          # planted lift per treated rep-month; never read by the estimator


def _z(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return np.nan_to_num((x - np.nanmean(x)) / s if s > 0 else x * 0)


def _pre_signals(cf: pd.DataFrame, reps: pd.DataFrame, start: int, n_pre: int = 6) -> pd.DataFrame:
    """What reps (and managers nominating them) can see before launch:
    attainment over the prior fiscal year, and momentum over the last six
    months. Prior-year attainment falls back to the trailing six months when
    the panel does not reach back a full year."""
    cf = cf.assign(total=cf.target_cf + cf.other_cf)
    fy0 = (start // 12 - 1) * 12
    prior = cf[(cf.t >= fy0) & (cf.t < fy0 + 12)] if fy0 >= 0 else cf[(cf.t >= start - n_pre) & (cf.t < start)]
    att = prior.groupby("rep_id").total.mean() * 12 / reps.set_index("rep_id").quota_annual
    pre = cf[(cf.t >= start - n_pre) & (cf.t < start)]
    g = pre.groupby("rep_id")
    mom = g.apply(lambda d: np.polyfit(d.t, np.log(d.total), 1)[0] if len(d) >= 3 else np.nan,
                  include_groups=False)
    return pd.DataFrame({"attainment": att, "momentum": mom})


def build(seed: int = C.SEED, true_effect: float = C.TRUE_EFFECT,
          cannibalisation: float = C.CANNIBALISATION) -> Tables:
    rng = np.random.default_rng(seed)
    reps = P.build_reps(rng)
    cf = P.counterfactual_bookings(reps, rng)
    per = P.periods()
    cf["lift_target"] = 0.0
    cf["program_id"] = None

    progs, pays = [], []
    for spec in PROGRAMS:
        s, e = spec["start"], spec["end"]
        elig = reps.query(spec["eligibility_rule"]).rep_id
        active = cf[(cf.t >= s) & (cf.t <= e)].rep_id.unique()
        pool = reps[reps.rep_id.isin(elig) & reps.rep_id.isin(active)].set_index("rep_id")
        sig = _pre_signals(cf, reps, s).reindex(pool.index)
        w = spec["selection"]
        logit = (w["base"] + w["skill"] * pool.latent_skill.to_numpy()
                 + w["attainment"] * _z(sig.attainment.to_numpy()) + w["momentum"] * _z(sig.momentum.to_numpy()))
        engaged = pool.index[rng.random(len(pool)) < 1 / (1 + np.exp(-logit))]
        # newer reps take longer to engage with a new mechanic
        tenure = pool.loc[engaged, "tenure_months"].to_numpy()
        p_late = np.where(tenure < 12, 0.55, 0.35)
        first = np.where(rng.random(len(engaged)) < p_late, rng.integers(1, e - s + 1, len(engaged)), 0) + s
        first_map = dict(zip(engaged, first))

        win = cf.rep_id.isin(engaged) & (cf.t >= s) & (cf.t <= e)
        on = win & (cf.t >= cf.rep_id.map(first_map))
        cf.loc[on, "lift_target"] += cf.loc[on, "target_cf"] * true_effect
        cf.loc[on, "program_id"] = spec["program_id"]

        rows = cf[on]
        target = rows.target_cf + rows.lift_target
        units = np.floor(target / UNIT_PRICE).astype(int)
        if spec["mechanic"] == "PerUnit":
            amount = units * spec["rate"]
        elif spec["mechanic"] == "FlatBounty":
            units = np.maximum(1, np.round(target / 7_500)).astype(int)    # qualifying attach deals
            amount = units * spec["rate"]
        else:
            amount = (np.minimum(units, TIER_UNITS) + np.maximum(units - TIER_UNITS, 0) * TIER_MULT) * spec["rate"]
        pays.append(pd.DataFrame({"program_id": spec["program_id"], "rep_id": rows.rep_id.to_numpy(),
                                  "period": rows.period.to_numpy(), "qualifying_units": units.to_numpy(),
                                  "payout_amount": np.round(amount.to_numpy(), 2)}))
        progs.append({"program_id": spec["program_id"], "name": spec["name"], "start_period": per[s],
                      "end_period": per[e], "eligibility_rule": spec["eligibility_rule"],
                      "mechanic": spec["mechanic"], "rate": spec["rate"], "budget": float(spec["budget"]),
                      "target_product": TARGET_PRODUCT})

    cf["lift_total"] = cf.lift_target * (1 - cannibalisation)
    target = cf.target_cf + cf.lift_target
    other = cf.other_cf - cannibalisation * cf.lift_target
    total = target + other
    seg = cf.rep_id.map(reps.set_index("rep_id").segment)
    deals = rng.poisson(np.maximum(total / seg.map(P.AVG_DEAL), 0.1))
    bookings = pd.DataFrame({
        "rep_id": cf.rep_id, "period": cf.period, "bookings": total.round(2), "deals_closed": deals,
        "pipeline_created": (total * rng.lognormal(np.log(3.0), 0.25, len(cf))).round(2),
        "product_mix_pct_focus": (target / total).round(4),
    }).sort_values(["rep_id", "period"]).reset_index(drop=True)
    truth = cf.loc[cf.lift_target > 0, ["program_id", "rep_id", "period", "lift_target", "lift_total"]]
    payouts = pd.concat(pays, ignore_index=True).sort_values(["program_id", "rep_id", "period"])
    return Tables(
        reps=reps[["rep_id", "segment", "region", "tenure_months", "quota_annual", "latent_skill"]],
        bookings_monthly=bookings, spiff_programs=pd.DataFrame(progs),
        spiff_payouts=payouts.reset_index(drop=True), truth=truth.reset_index(drop=True))


def write(tables: Tables, out=C.DATA_DIR):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("reps", "bookings_monthly", "spiff_programs", "spiff_payouts"):
        getattr(tables, name).to_parquet(out / f"{name}.parquet", index=False)
    # Ground truth is kept apart from the analysis tables on purpose.
    (out / "_truth").mkdir(exist_ok=True)
    tables.truth.to_parquet(out / "_truth" / "planted_effect.parquet", index=False)


if __name__ == "__main__":
    t = build()
    write(t)
    print(f"reps {len(t.reps)}, rep-months {len(t.bookings_monthly)}, payouts {len(t.spiff_payouts)}")
