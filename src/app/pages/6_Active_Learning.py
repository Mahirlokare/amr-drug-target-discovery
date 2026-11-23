"""
Page 6: Retrospective Active Learning Simulation.
Complies with Project 9 Blueprint Section 12.3 & 15.1 (Active Learning).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Active Learning | AMR Discovery", page_icon="📈", layout="wide")
st.title("📈 Retrospective Active Learning Simulation")

st.markdown("""
Active learning simulates iterative cycles of computational candidate selection followed by wet-lab assay verification.
By balancing **predicted activity**, **model epistemic uncertainty**, and **chemical scaffold diversity**, the platform
prioritizes the most informative molecules to screen next under a fixed experimental budget.
""")

al_file = _ROOT / "artifacts" / "reports" / "active_learning_simulation.json"

if not al_file.exists():
    st.warning("Active learning simulation results not found. Please run `python run_pipeline.py` first.")
    st.stop()

with open(al_file, "r", encoding="utf-8") as f:
    sim_data = json.load(f)

random_res = sim_data.get("random_policy", {})
balanced_res = sim_data.get("balanced_policy", {})

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style="background: rgba(100, 116, 139, 0.1); border-radius: 8px; padding: 12px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Random Selection Hit Rate</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #F1F5F9;">{random_res.get('overall_hit_rate_pct', 0)}%</div>
        <div style="font-size: 0.75rem; color: #64748B;">Uniform random sampling baseline</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px; text-align: center;">
        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Balanced AL Policy Hit Rate</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #34D399;">{balanced_res.get('overall_hit_rate_pct', 0)}%</div>
        <div style="font-size: 0.75rem; color: #64748B;">Activity + Uncertainty + Diversity policy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Cumulative Discovery Curves
st.subheader("Cumulative Actives Discovered vs. Screening Budget")

budget = balanced_res.get("cumulative_budget", [])
balanced_actives = balanced_res.get("cumulative_actives", [])
random_actives = random_res.get("cumulative_actives", [])

# Align lengths
min_len = min(len(budget), len(balanced_actives), len(random_actives))
curve_df = pd.DataFrame({
    "Compounds Screened": budget[:min_len],
    "Balanced Active Learning (Proposed)": balanced_actives[:min_len],
    "Random Sampling (Baseline)": random_actives[:min_len]
}).set_index("Compounds Screened")

st.line_chart(curve_df)
st.caption("Figure: Cumulative confirmed antimicrobial-active molecules discovered over sequential screening rounds on held-out test data.")

st.markdown("---")

# Round-by-Round Breakdown
st.subheader("Iterative Round History (Balanced Policy)")
history = balanced_res.get("round_histories", [])
if history:
    st.table(pd.DataFrame(history))
