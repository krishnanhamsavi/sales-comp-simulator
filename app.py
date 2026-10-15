"""Sales Compensation Plan Simulator.

    streamlit run app.py
"""
import streamlit as st

from comp_sim.app import scenario_page
from spiff_effectiveness.app import spiff_page
from spiff_effectiveness.config import DEFAULT

st.set_page_config(page_title="Sales Comp Plan Simulator", page_icon=":material/payments:", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.55rem;}
  .eyebrow {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #6b7280; margin-bottom: -.4rem;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### :material/payments: Comp Plan Simulator")
    st.caption("Design incentive plans before launch, and measure whether they worked after.")
    st.divider()
    st.markdown("**Program Effectiveness assumption**")
    margin = st.slider("Gross margin on incremental bookings", 0.40, 0.90, DEFAULT.gross_margin, 0.05,
                       help="Not measured anywhere in the data. Net return and ROI move with it, which is the point: "
                            "the headline depends on an input someone has to own.")
    st.caption(":material/warning: An assumption, not a measurement. Watch the net return move as you change it.")
    st.divider()
    st.caption("All data is synthetic, generated from documented distributions with a planted, known effect.")

st.markdown('<p class="eyebrow">Sales compensation · incentive design and measurement</p>', unsafe_allow_html=True)
st.title("Sales Compensation Plan Simulator")
t1, t2 = st.tabs([":material/tune: Plan Scenarios", ":material/insights: Program Effectiveness"])
with t1:
    scenario_page.render()
with t2:
    spiff_page.render(margin)
