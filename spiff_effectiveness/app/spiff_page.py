"""Streamlit tab: Program Effectiveness.

Reads a ProgramResult and formats it. No arithmetic happens here: every
number on the page is a field of ProgramResult. Sections run in a fixed
order so the design checks are seen before the headline.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import DEFAULT, MeasureConfig
from ..data import loader
from ..report import assemble, narrate
from ..report.narrate import pct, usd

BLUE, ORANGE, GRAY, GREEN, RED, AMBER = "#2a78d6", "#eb6834", "#8a8984", "#1a7f4b", "#c4382f", "#b86e00"


@st.cache_resource(show_spinner="Loading the rep panel…")
def _data() -> loader.PanelData:
    return loader.load()


@st.cache_resource(show_spinner="Matching cohorts and estimating…")
def _core(program_id: str, cfg: MeasureConfig) -> assemble.Core:
    return assemble.measure(_data(), program_id, cfg)


def _layout(fig, h=320, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=46 if title else 16, b=10), title=title,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12),
                      title_font=dict(size=14),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def _badge(passed: bool | None, label_ok="Pass", label_bad="Fail", na="Not available"):
    if passed is None:
        st.badge(na, color="gray", icon=":material/remove:")
    elif passed:
        st.badge(label_ok, color="green", icon=":material/check:")
    else:
        st.badge(label_bad, color="red", icon=":material/close:")


def _whiskers(fig, xs, lows, highs, color="rgba(60,60,60,0.75)", name="95% CI"):
    """Confidence intervals drawn straight from the stored bounds."""
    x, y = [], []
    for xi, lo, hi in zip(xs, lows, highs):
        x += [xi, xi, None]
        y += [lo, hi, None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2), name=name,
                             hoverinfo="skip", showlegend=False))


def _md(text: str) -> str:
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
def render(margin: float):
    data = _data()
    progs = data.programs
    st.markdown("Did the SPIFF cause incremental revenue, and did that revenue pay for the payouts? Reps who "
                "engage with a SPIFF are usually the reps who were already selling well, so comparing "
                "participants with everyone else mostly measures who chose to take part. This view builds a "
                "matched control group from comparable eligible reps, checks the design, and only then reports "
                "the effect, net of what the program paid out.")
    labels = {r.program_id: f"{r.program_id} · {r['name']}" for _, r in progs.iterrows()}
    pid = st.selectbox("Program", list(labels), format_func=labels.get)
    cfg = assemble.with_margin(DEFAULT, margin)
    core = _core(pid, assemble.with_margin(DEFAULT, DEFAULT.gross_margin))
    r = assemble.build(data, core, cfg)
    s = r.setup

    # 1. setup -----------------------------------------------------------------
    st.markdown("#### 1 · Program setup")
    k = st.columns(6)
    k[0].metric("Window start", s.window_start, f"to {s.window_end} · {s.months} months", delta_color="off",
                delta_arrow="off")
    k[1].metric("Mechanic", s.mechanic, f"{usd(s.rate)} per {'deal' if s.mechanic == 'FlatBounty' else 'unit'}",
                delta_color="off", delta_arrow="off")
    k[2].metric("Budget", usd(s.budget))
    k[3].metric("Eligible reps", s.eligible_reps)
    k[4].metric("Treated reps analysed", s.treated_reps, f"{r.engagement.participants} paid in total",
                delta_color="off", delta_arrow="off")
    k[5].metric("Control pool", s.control_pool)
    st.caption(f"Eligibility rule: `{s.eligibility_rule}` · target product: **{s.target_product}**")
    with st.expander("Who was excluded from the comparison, and why"):
        st.markdown(f"""
| Reason | Reps | Why they are left out |
|---|---|---|
| Fail the eligibility rule | {s.excluded_ineligible} | Structurally different; never used as controls. |
| Paid by another program during the span | {s.excluded_contaminated} | Bookings carry another incentive's effect. |
| New hires still ramping | {s.excluded_ramping} | Ramp-up growth would read as a program effect. |
| Too little pre-period history | {s.excluded_thin_history} | Fewer than half the pre-period months observed. |
""")

    # 2. design quality ----------------------------------------------------------
    st.markdown("#### 2 · Design quality")
    st.caption("Checked before any result is shown. If a check fails, the result section reports the failure "
               "instead of a number.")
    c = st.columns(4)
    with c[0].container(border=True):
        st.markdown("**Overlap**")
        _badge(r.status != assemble.STATUS_NO_OVERLAP)
        st.caption(f"{r.matching.matched_pairs} matched pairs; {r.matching.dropped_treated} treated reps had no "
                   f"control inside the caliper.")
    with c[1].container(border=True):
        st.markdown("**Covariate balance**")
        _badge(r.matching.balance_passed)
        st.caption(f"Every feature within {DEFAULT.smd_threshold:.2f} standardised mean difference after matching.")
    with c[2].container(border=True):
        st.markdown("**Parallel trends**")
        _badge(None if r.parallel_trends is None else r.parallel_trends.passed)
        if r.parallel_trends is not None:
            st.caption(f"Joint test of pre-period event-study coefficients: p = {r.parallel_trends.joint_p:.3f}.")
    with c[3].container(border=True):
        st.markdown("**Placebo window**")
        pl = r.placebo
        _badge(None if pl is None or not pl.available else pl.passed)
        if pl is not None:
            st.caption(f"Fake launch {pl.window_start} → {pl.window_end}, where the true effect is zero."
                       + ("" if pl.available else " Not enough history before launch to run it."))

    t1, t2, t3 = st.tabs(["Covariate balance", "Pre-period trends", "Event study"])
    with t1:
        bal = pd.DataFrame([b.__dict__ for b in r.matching.balance])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bal.smd_before.abs(), y=bal.feature, mode="markers", name="Before matching",
                                 marker=dict(color=GRAY, size=10, symbol="circle-open", line=dict(width=2))))
        fig.add_trace(go.Scatter(x=bal.smd_after.abs(), y=bal.feature, mode="markers", name="After matching",
                                 marker=dict(color=BLUE, size=10)))
        fig.add_vline(x=DEFAULT.smd_threshold, line=dict(color=RED, dash="dash", width=1),
                      annotation_text=f"{DEFAULT.smd_threshold:.2f} threshold", annotation_position="top right")
        fig.update_xaxes(title="|Standardised mean difference|")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(_layout(fig, 380, "Balance on pre-period features"), width="stretch")
        c2.dataframe(bal.rename(columns={"feature": "Feature", "smd_before": "SMD before", "smd_after": "SMD after",
                                         "passed": "Pass"})
                     .style.format({"SMD before": "{:.3f}", "SMD after": "{:.3f}"}),
                     hide_index=True, width="stretch", height=380)
        st.caption(f"Matching: propensity model on pre-period behaviour; 1:1 without replacement inside a caliper of "
                   f"{DEFAULT.caliper_sd} SD of the logit; same segment; nearest by Mahalanobis distance, then a "
                   "balance-improving swap pass. Rep skill is never observed.")
    if r.parallel_trends is None:
        with t2:
            st.info("Not computed: too few matched pairs.")
        with t3:
            st.info("Not computed: too few matched pairs.")
    else:
        tr = pd.DataFrame([p.__dict__ for p in r.parallel_trends.trend])
        with t2:
            fig = go.Figure()
            fig.add_vrect(x0=-0.5, x1=tr.rel_month.max() + 0.5, fillcolor=BLUE, opacity=0.06, line_width=0,
                          annotation_text="Program window", annotation_position="top left")
            fig.add_trace(go.Scatter(x=tr.rel_month, y=tr.treated_mean, name="Treated reps", mode="lines+markers",
                                     line=dict(color=BLUE, width=2), marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=tr.rel_month, y=tr.control_mean, name="Matched controls",
                                     mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8)))
            fig.update_xaxes(title="Months relative to launch", dtick=1)
            fig.update_yaxes(title="Mean monthly bookings", tickprefix="$", tickformat="~s")
            st.plotly_chart(_layout(fig, 360, "Mean bookings, treated vs. matched controls"), width="stretch")
            st.caption("Before launch the two lines should move together. The shared quarter-end spike and upward "
                       "trend are why a before-and-after comparison without controls is wrong.")
        with t3:
            ev = pd.DataFrame([e.__dict__ for e in r.parallel_trends.coefs])
            fig = go.Figure()
            fig.add_hline(y=0, line=dict(color=GRAY, width=1))
            fig.add_vline(x=-0.5, line=dict(color=GRAY, dash="dot", width=1))
            colors = [GRAY if m < 0 else BLUE for m in ev.rel_month]
            _whiskers(fig, ev.rel_month, ev.ci_low, ev.ci_high)
            fig.add_trace(go.Scatter(
                x=ev.rel_month, y=ev.beta, mode="markers", marker=dict(size=10, color=colors),
                hovertemplate="Month %{x}<br>%{y:$,.0f}<extra></extra>", showlegend=False))
            fig.update_xaxes(title="Months relative to launch (month −1 is the reference)", dtick=1)
            fig.update_yaxes(title="Treated − control, $ per rep-month", tickprefix="$", tickformat="~s")
            st.plotly_chart(_layout(fig, 360, "Event study with 95% confidence intervals"), width="stretch")
            st.caption(f"Pre-period coefficients (grey) should sit on zero. Joint test p = "
                       f"{r.parallel_trends.joint_p:.3f}; {r.parallel_trends.n_significant_pre} individually "
                       "significant.")

    # 3. result --------------------------------------------------------------------
    st.markdown("#### 3 · Result")
    if r.status != assemble.STATUS_OK:
        title = ("Not enough comparable reps to estimate an effect" if r.status == assemble.STATUS_NO_OVERLAP
                 else "Headline suppressed: the design failed its checks")
        st.error(_md(f"**{title}.**\n\n" + "\n".join(f"- {x}" for x in r.status_reasons)
                     + "\n\nNo effect size, net return or ROI is reported for this program."),
                 icon=":material/block:")
    else:
        e, ec = r.estimates, r.economics
        k = st.columns(4)
        k[0].metric("Incremental bookings / rep-month", usd(e.att_total.beta),
                    _md(f"95% CI {usd(e.att_total.ci_low)} to {usd(e.att_total.ci_high)}"), delta_color="off",
                    delta_arrow="off")
        k[1].metric("Incremental bookings, total", usd(ec.incremental_bookings.point),
                    _md(f"95% CI {usd(ec.incremental_bookings.low)} to {usd(ec.incremental_bookings.high)}"),
                    delta_color="off", delta_arrow="off")
        k[2].metric("Payout cost", usd(ec.payout_cost), f"{pct(ec.budget_utilisation)} of budget",
                    delta_color="off", delta_arrow="off")
        k[3].metric("Net program return", usd(ec.net_return.point),
                    _md(f"95% CI {usd(ec.net_return.low)} to {usd(ec.net_return.high)}"), delta_color="off",
                    delta_arrow="off")
        k = st.columns(4)
        k[0].metric("Program ROI", pct(ec.roi.point), f"95% CI {pct(ec.roi.low)} to {pct(ec.roi.high)}",
                    delta_color="off", delta_arrow="off")
        k[1].metric("Breakeven effect", usd(ec.breakeven_att), "per treated rep-month", delta_color="off",
                    delta_arrow="off")
        k[2].metric("Breakeven verdict", ec.breakeven_verdict.capitalize(),
                    "CI vs. breakeven", delta_color="off", delta_arrow="off")
        k[3].metric("Gross margin (assumption)", pct(ec.gross_margin, 0), "set in the sidebar", delta_color="off",
                    delta_arrow="off")
        verdict_text = {"cleared": "The whole confidence interval sits above breakeven: the program paid for itself.",
                        "not cleared": "The whole confidence interval sits below breakeven: the program did not "
                                       "pay for itself.",
                        "inconclusive": "Breakeven sits inside the confidence interval: the data cannot say whether "
                                        "the program paid for itself."}[ec.breakeven_verdict]
        st.info(verdict_text, icon=":material/balance:")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for name, est, col in (("Target product", e.att_target, ORANGE), ("Total bookings", e.att_total, BLUE)):
            fig.add_trace(go.Bar(x=[name], y=[est.beta], marker_color=col, showlegend=False, width=0.5,
                                 hovertemplate=f"{name}<br>%{{y:$,.0f}}<extra></extra>"))
            _whiskers(fig, [name], [est.ci_low], [est.ci_high])
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        fig.update_yaxes(title="$ per treated rep-month", tickprefix="$", tickformat="~s")
        c1.plotly_chart(_layout(fig, 340, "Cannibalisation read: target-product vs. total lift"), width="stretch")
        c1.caption(_md(f"Cannibalisation rate {pct(e.cannibalisation_rate.beta)} (95% CI "
                       f"{pct(e.cannibalisation_rate.ci_low)} to {pct(e.cannibalisation_rate.ci_high)}): the share "
                       "of the target-product lift that came out of other products."))
        fig = go.Figure()
        names = ["Participants vs. non-participants", "Participants, before vs. after", "Matched DiD"]
        vals = [r.naive.treated_vs_untreated, r.naive.before_after, e.att_total.beta]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=[GRAY, GRAY, BLUE], showlegend=False,
                             text=[usd(v) for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>"))
        _whiskers(fig, [names[2]], [e.att_total.ci_low], [e.att_total.ci_high])
        fig.update_yaxes(title="$ per rep-month", tickprefix="$", tickformat="~s")
        c2.plotly_chart(_layout(fig, 340, "Why the design matters"), width="stretch")
        c2.caption("The two naive reads (grey) mostly measure who chose to participate and the shared seasonal "
                   "trend. The matched difference-in-differences removes both.")

        ms = pd.DataFrame([{"margin": g, "point": rg.point, "low": rg.low, "high": rg.high}
                           for g, rg in ec.margin_sensitivity])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(ms.margin) + list(ms.margin[::-1]), y=list(ms.high) + list(ms.low[::-1]),
                                 fill="toself", fillcolor="rgba(42,120,214,0.14)", line=dict(width=0),
                                 name="95% CI", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ms.margin, y=ms.point, mode="lines", line=dict(color=BLUE, width=2),
                                 name="Net return", hovertemplate="margin %{x:.0%}<br>%{y:$,.0f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        fig.add_vline(x=ec.gross_margin, line=dict(color=ORANGE, dash="dot", width=2),
                      annotation_text=f"current assumption {pct(ec.gross_margin, 0)}", annotation_position="top")
        fig.update_xaxes(title="Gross margin assumption", tickformat=".0%")
        fig.update_yaxes(title="Net program return", tickprefix="$", tickformat="~s")
        st.plotly_chart(_layout(fig, 320, "The headline depends on an unverified input: net return vs. gross margin"),
                        width="stretch")

    # 4. engagement ------------------------------------------------------------------
    st.markdown("#### 4 · Engagement")
    g = r.engagement
    k = st.columns(5)
    k[0].metric("Participation rate", pct(g.participation_rate), f"{g.participants} of {g.eligible_reps} eligible",
                delta_color="off", delta_arrow="off")
    k[1].metric("Payout concentration", pct(g.payout_concentration),
                f"to the top {pct(g.concentration_top, 0)} of payees", delta_color="off", delta_arrow="off")
    k[2].metric("Median days to first payout", f"{g.median_days_to_first_payout:.0f}", "payouts post monthly",
                delta_color="off", delta_arrow="off")
    k[3].metric("Median payout per participant", usd(g.median_payout_per_participant))
    k[4].metric("Spend vs. budget", usd(g.total_payout), f"{pct(g.budget_utilisation)} of budget",
                delta_color="off", delta_arrow="off")
    c1, c2 = st.columns(2)
    cc = pd.DataFrame(g.concentration_curve, columns=["participants", "payout"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot", width=1),
                             name="Equal payouts", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cc.participants, y=cc.payout, mode="lines", line=dict(color=BLUE, width=2),
                             name="Actual", hovertemplate="top %{x:.0%} of participants<br>%{y:.0%} of payout"
                                                          "<extra></extra>"))
    fig.update_xaxes(title="Share of participants, highest earners first", tickformat=".0%")
    fig.update_yaxes(title="Cumulative share of payout", tickformat=".0%")
    c1.plotly_chart(_layout(fig, 340, "Payout concentration"), width="stretch")
    grp = pd.DataFrame([x.__dict__ for x in g.by_group])
    with c2:
        st.markdown("##### Participation by segment and region")
        st.dataframe(grp.rename(columns={"group": "Dimension", "value": "Group", "eligible": "Eligible",
                                         "participants": "Participants", "participation_rate": "Participation",
                                         "payout": "Payout"})
                     .style.format({"Participation": "{:.1%}", "Payout": "${:,.0f}"}),
                     hide_index=True, width="stretch", height=300)
    st.caption("Engagement describes the program, not individual reps. High concentration means a small group "
               "earned most of the payout, often for work they were already doing.")

    # memo -----------------------------------------------------------------------------
    st.markdown("#### Program post-mortem memo")
    st.caption("Narrative drafted by an AI model from the figures above. Every number in the draft is checked "
               "against the computed result; a draft containing any number that is not in the result is "
               "rejected and regenerated.")
    c = st.columns([1.2, 1, 4])
    use_llm = c[0].toggle("AI-drafted narrative", value=True,
                          help="Without an API key, a structured template memo is produced from the same figures.")
    key = f"memo_{pid}_{margin}"
    if c[1].button("Draft memo", type="primary", icon=":material/description:"):
        with st.spinner("Drafting and checking…"):
            st.session_state[key] = narrate.narrate(r, draft_fn=narrate.draft_llm if use_llm else None)
    memo = st.session_state.get(key)
    if memo:
        if memo.error:
            st.info("AI drafting unavailable; showing the template memo.")
        if memo.rejected:
            st.warning(_md(f"Provenance guard rejected {len(memo.rejected)} draft(s) containing numbers not in the "
                           "result: " + "; ".join(", ".join(x) for x in memo.rejected)))
        st.caption(":material/verified: Every figure in this memo traces to the computed result.")
        with st.container(border=True):
            st.markdown(_md(memo.markdown))
        st.download_button("Download memo (.md)", memo.markdown, f"{pid}_post_mortem.md", "text/markdown",
                           icon=":material/download:")
