"""Streamlit tab: Plan Scenarios. If we change the plan, what does it cost?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C
from .. import engine, plans
from ..plans import ARCHETYPES, Plan

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def usd(x: float) -> str:
    x = round(x)
    return f"-${abs(x):,}" if x < 0 else f"${x:,}"


def money(x: float) -> str:
    return f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:.0f}K"


@st.cache_resource
def _pop() -> pd.DataFrame:
    return engine.population()


@st.cache_data(show_spinner="Simulating plan years…")
def _eval(plan: Plan):
    return engine.evaluate(plan, _pop())


@st.cache_data(show_spinner=False)
def _sens(plan: Plan) -> pd.DataFrame:
    return engine.sensitivity(plan, _pop())


def _layout(fig, h=340, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=46 if title else 16, b=10), title=title,
                      title_font=dict(size=14), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def plan_editor(label: str, default: str, key: str) -> Plan:
    presets = plans.presets()
    with st.container(border=True):
        st.markdown(f"**{label}**")
        name = st.selectbox("Start from", list(presets), index=list(presets).index(default), key=f"{key}_preset")
        base = presets[name]
        arch = st.selectbox("Archetype", list(ARCHETYPES), index=list(ARCHETYPES).index(base.archetype),
                            key=f"{key}_arch", help="\n\n".join(f"**{k}**: {v}" for k, v in ARCHETYPES.items()))
        c = st.columns(2)
        kw = dict(name=f"{label}: {arch}", archetype=arch)
        if arch == "Accelerated commission":
            kw["threshold"] = c[0].slider("Threshold (attainment)", 0.0, 0.9, base.threshold or 0.5, 0.05,
                                          key=f"{key}_thr")
            kw["decelerator"] = c[1].slider("Rate below quota (x)", 0.2, 1.0, base.decelerator if base.archetype == arch
                                            else 0.5, 0.05, key=f"{key}_dec")
            kw["accelerator"] = c[0].slider("Rate above quota (x)", 1.0, 3.0, base.accelerator if base.archetype == arch
                                            else 2.0, 0.1, key=f"{key}_acc")
            cap = c[1].slider("Cap (x target incentive, 0 = none)", 0.0, 5.0, base.cap or 3.0, 0.25, key=f"{key}_cap")
            kw["cap"] = cap or None
        elif arch == "Capped target bonus":
            kw["threshold"] = c[0].slider("Threshold (attainment)", 0.0, 0.9, base.threshold or 0.6, 0.05,
                                          key=f"{key}_thr")
            kw["kicker"] = c[1].slider("Kicker at quota (x TI)", 0.0, 0.5, base.kicker or 0.25, 0.05, key=f"{key}_kick")
            kw["cap"] = c[0].slider("Cap (x TI)", 1.0, 2.0, base.cap or 1.5, 0.05, key=f"{key}_cap")
        elif arch == "Tiered commission":
            t = base.tiers if base.archetype == arch else ((0.6, 0.8), (1.0, 1.3), (1.5, 1.6), (10.0, 1.2))
            r1 = c[0].slider("Rate 0–60%", 0.2, 1.5, t[0][1], 0.05, key=f"{key}_t1")
            r2 = c[1].slider("Rate 60–100%", 0.5, 2.0, t[1][1], 0.05, key=f"{key}_t2")
            r3 = c[0].slider("Rate 100–150%", 0.5, 3.0, t[2][1], 0.05, key=f"{key}_t3")
            r4 = c[1].slider("Rate above 150%", 0.0, 3.0, t[3][1], 0.05, key=f"{key}_t4")
            kw["tiers"] = ((0.6, r1), (1.0, r2), (1.5, r3), (10.0, r4))
        else:
            cap = c[0].slider("Cap (x TI, 0 = none)", 0.0, 5.0, 0.0, 0.25, key=f"{key}_cap")
            kw["cap"] = cap or None
        plan = Plan(**kw)
        at_quota = float(plan.payout_multiple(np.array([1.0]))[0])
        st.caption(f"Pays **{at_quota:.2f}x** target incentive at 100% attainment."
                   + ("" if abs(at_quota - 1 - plan.kicker) < 1e-6
                      else " Check the rates: plans usually pay 1.0x at quota."))
    return plan


def render():
    st.markdown("Model what a compensation plan will cost before it launches. Each plan maps quota attainment to a "
                f"payout multiple of target incentive. The engine simulates {C.N_SIMS} plan years for a "
                f"{C.N_REPS}-rep team, with attainment driven by persistent rep skill and year-to-year noise, and "
                "reports expected cost, its range, and who earns it.")
    c1, c2 = st.columns(2)
    with c1:
        base = plan_editor("Current plan", "Current plan: linear commission", "a")
    with c2:
        prop = plan_editor("Proposed plan", "Proposed: threshold + 2x accelerator", "b")
    a, b = _eval(base), _eval(prop)

    k = st.columns(5)
    k[0].metric("Expected variable comp", money(b.variable_cost),
                f"{money(b.variable_cost - a.variable_cost)} vs. current".replace("$", r"\$"),
                delta_color="inverse")
    k[1].metric("Cost of sales", f"{b.cost_of_sales:.1%}", f"{(b.cost_of_sales - a.cost_of_sales) * 100:+.1f} pts",
                delta_color="inverse")
    k[2].metric("Payout vs. target incentive", f"{b.payout_vs_target:.0%}", f"current {a.payout_vs_target:.0%}",
                delta_color="off", delta_arrow="off")
    k[3].metric("Reps at or above quota", f"{b.pct_at_quota:.0%}", "same attainment in both", delta_color="off",
                delta_arrow="off")
    k[4].metric("Pay to top 20% of reps", f"{b.top_quintile_share:.0%}", f"current {a.top_quintile_share:.0%}",
                delta_color="off", delta_arrow="off")

    c1, c2 = st.columns(2)
    att = np.linspace(0, 2.0, 201)
    fig = go.Figure()
    for p, col in ((base, GRAY), (prop, BLUE)):
        fig.add_trace(go.Scatter(x=att, y=p.payout_multiple(att), mode="lines", name=p.name,
                                 line=dict(color=col, width=2.5),
                                 hovertemplate="attainment %{x:.0%}<br>payout %{y:.2f}x TI<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=GRAY, dash="dot", width=1))
    fig.update_xaxes(title="Quota attainment", tickformat=".0%")
    fig.update_yaxes(title="Payout, multiple of target incentive")
    c1.plotly_chart(_layout(fig, 360, "Payout curves"), width="stretch")

    fig = go.Figure()
    for i, (r, col) in enumerate(((a, GRAY), (b, BLUE))):
        fig.add_trace(go.Scatter(x=[r.variable_cost_p10, r.variable_cost_p90], y=[r.plan.name] * 2, mode="lines",
                                 line=dict(color=col, width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.variable_cost], y=[r.plan.name], mode="markers+text", showlegend=False,
                                 marker=dict(size=14, color=col, line=dict(color="white", width=2)),
                                 text=[money(r.variable_cost)], textposition="top center",
                                 hovertemplate="expected %{x:$,.0f}<extra></extra>"))
    fig.update_xaxes(title="Annual variable comp (P10 to P90 across simulated years)", tickprefix="$",
                     tickformat="~s")
    c2.plotly_chart(_layout(fig, 360, "Cost range"), width="stretch")

    st.markdown("##### What if the team over- or under-performs?")
    st.caption("Team-wide attainment shifts. Marginal comp per $1 is the extra variable pay for each extra dollar "
               "of bookings: accelerators make upside expensive, thresholds make downside cheap.")
    sa, sb = _sens(base), _sens(prop)
    tbl = pd.DataFrame({"Attainment shift": sa["Attainment shift"], "Bookings": sa["Bookings"],
                        "Current: variable comp": sa["Variable comp"], "Proposed: variable comp": sb["Variable comp"],
                        "Current: comp per $1": sa["Marginal comp per $1 bookings"],
                        "Proposed: comp per $1": sb["Marginal comp per $1 bookings"]})
    st.dataframe(tbl.style.format({"Attainment shift": "{:+.0%}", "Bookings": "${:,.0f}",
                                   "Current: variable comp": "${:,.0f}", "Proposed: variable comp": "${:,.0f}",
                                   "Current: comp per $1": lambda v: "—" if pd.isna(v) else f"${v:.3f}",
                                   "Proposed: comp per $1": lambda v: "—" if pd.isna(v) else f"${v:.3f}"}, na_rep="—"),
                 hide_index=True, width="stretch")

    st.markdown("##### Payout distribution by segment (multiple of target incentive)")
    seg = b.rep_payout_pcts.rename(columns={"segment": "Segment", "p10": "P10", "p25": "P25", "p50": "Median",
                                            "p75": "P75", "p90": "P90"})
    st.dataframe(seg.style.format({c: "{:.2f}x" for c in ["P10", "P25", "Median", "P75", "P90"]}),
                 hide_index=True, width="stretch")
    with st.expander("Team assumptions"):
        st.dataframe(pd.DataFrame([{"Segment": s, "Share of reps": v[0], "OTE": v[1], "Variable pay mix": v[2],
                                    "Annual quota": v[3], "Median attainment": C.ATTAINMENT_MEDIAN[s],
                                    "Attainment spread (log SD)": C.ATTAINMENT_SIGMA[s]}
                                   for s, v in C.SEGMENTS.items()])
                     .style.format({"Share of reps": "{:.0%}", "OTE": "${:,.0f}", "Variable pay mix": "{:.0%}",
                                    "Annual quota": "${:,.0f}", "Median attainment": "{:.0%}",
                                    "Attainment spread (log SD)": "{:.2f}"}), hide_index=True, width="stretch")
