"""Compensation plan archetypes.

Every plan maps a rep's annual quota attainment to a payout expressed as a
multiple of their target incentive (TI = OTE x variable pay mix). A payout
multiple of 1.0 at 100% attainment means the plan pays exactly target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARCHETYPES = {
    "Linear commission": "Flat commission rate on every dollar booked, set so 100% attainment pays target. "
                         "Simple and uncapped; pays from the first dollar.",
    "Accelerated commission": "Reduced rate up to a threshold, standard rate to quota, accelerated rate above "
                              "quota, optional cap. The most common enterprise design.",
    "Tiered commission": "Stepped rates by attainment band. Each band pays its own rate on bookings inside "
                         "that band.",
    "Capped target bonus": "Bonus earned in proportion to attainment between a threshold and quota, with a "
                           "fixed kicker above quota and a hard cap.",
}


@dataclass(frozen=True)
class Plan:
    name: str
    archetype: str
    threshold: float = 0.0          # attainment below which nothing is paid
    decelerator: float = 1.0        # rate multiple between threshold and quota
    accelerator: float = 1.0        # rate multiple above quota
    cap: float | None = None        # max payout multiple of TI
    tiers: tuple = field(default_factory=tuple)   # ((upper_attainment, rate_multiple), ...)
    kicker: float = 0.0             # bonus multiple of TI added once quota is reached

    def payout_multiple(self, att: np.ndarray) -> np.ndarray:
        att = np.asarray(att, dtype=float)
        a = self.archetype
        if a == "Linear commission":
            out = att.copy()
        elif a == "Accelerated commission":
            below = np.clip(att, 0, 1.0)
            out = np.where(att < self.threshold, 0.0, below * self.decelerator)
            # Decelerated band is rescaled so 100% attainment pays exactly 1.0
            out = np.where(att >= self.threshold,
                           self.threshold * self.decelerator
                           + (below - self.threshold) * (1 - self.threshold * self.decelerator)
                           / (1 - self.threshold), out)
            out = out + np.clip(att - 1.0, 0, None) * self.accelerator
        elif a == "Tiered commission":
            out = np.zeros_like(att)
            lo = 0.0
            for hi, rate in self.tiers:
                out += np.clip(np.minimum(att, hi) - lo, 0, None) * rate
                lo = hi
        elif a == "Capped target bonus":
            out = np.where(att < self.threshold, 0.0, np.clip(att, 0, 1.0))
            out = out + np.where(att >= 1.0, self.kicker, 0.0)
        else:
            raise ValueError(f"unknown archetype {a}")
        if self.cap is not None:
            out = np.minimum(out, self.cap)
        return out


def presets() -> dict[str, Plan]:
    return {
        "Current plan: linear commission": Plan("Current plan: linear commission", "Linear commission"),
        "Proposed: threshold + 2x accelerator": Plan(
            "Proposed: threshold + 2x accelerator", "Accelerated commission",
            threshold=0.5, decelerator=0.5, accelerator=2.0, cap=3.0),
        "Proposed: tiered 0.8 / 1.0 / 1.6": Plan(
            "Proposed: tiered 0.8 / 1.0 / 1.6", "Tiered commission",
            tiers=((0.6, 0.8), (1.0, 1.3), (1.5, 1.6), (10.0, 1.2))),
        "Proposed: capped bonus": Plan(
            "Proposed: capped bonus", "Capped target bonus", threshold=0.6, kicker=0.25, cap=1.5),
    }
