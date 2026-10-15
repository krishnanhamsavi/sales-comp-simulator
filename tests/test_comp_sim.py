import numpy as np
import pytest

from comp_sim import engine, plans
from comp_sim.plans import Plan

PRESETS = list(plans.presets().values())


@pytest.mark.parametrize("plan", PRESETS, ids=lambda p: p.name)
def test_every_preset_pays_target_at_quota(plan):
    assert plan.payout_multiple(np.array([1.0]))[0] == pytest.approx(1.0 + plan.kicker)


@pytest.mark.parametrize("plan", PRESETS, ids=lambda p: p.name)
def test_payout_is_monotone_and_capped(plan):
    att = np.linspace(0, 3, 301)
    pay = plan.payout_multiple(att)
    assert np.all(np.diff(pay) >= -1e-12)
    if plan.cap is not None:
        assert pay.max() <= plan.cap + 1e-12


def test_threshold_pays_nothing_below():
    p = Plan("t", "Accelerated commission", threshold=0.5, decelerator=0.5, accelerator=2.0)
    assert p.payout_multiple(np.array([0.49]))[0] == 0
    assert p.payout_multiple(np.array([1.5]))[0] == pytest.approx(2.0)


def test_engine_is_deterministic_and_sensible():
    pop = engine.population()
    a = engine.evaluate(PRESETS[0], pop)
    b = engine.evaluate(PRESETS[0], pop)
    assert a.variable_cost == b.variable_cost
    assert 0.45 <= a.pct_at_quota <= 0.75
    assert a.variable_cost_p10 < a.variable_cost < a.variable_cost_p90


def test_higher_attainment_costs_more():
    pop = engine.population()
    s = engine.sensitivity(PRESETS[1], pop)
    assert s["Variable comp"].is_monotonic_increasing
