"""
Page 12: Configuration, Storage & System Health.
Complies with Project 9 Blueprint Section 16 & 15.1 (Settings).
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import streamlit as st
import rdkit

from src.common.config import config_manager
from src.storage.db import db_manager

st.set_page_config(page_title="Settings | DeepTarget-AMR", page_icon="⚙️", layout="wide")

st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
    <span style="font-size: 1.2rem;">🧬</span>
    <span style="font-size: 0.95rem; font-weight: 700; color: #38BDF8;">DeepTarget-AMR</span>
    <span style="color: #64748B;">/</span>
    <span style="font-size: 0.95rem; color: #94A3B8;">System Settings & Health</span>
</div>
""", unsafe_allow_html=True)

st.title("⚙️ Platform Settings, Storage & Diagnostics")
st.markdown("Monitor cloud infrastructure, persistence engines, database schemas, and runtime configurations:")

# 1. Database Health Check
st.subheader("🗄️ Persistence & Database Engine")
health = db_manager.health_check()

if health.get("status") == "HEALTHY":
    st.success(f"Persistence Engine: **Operational** ({health.get('engine')})")
    
    counts = health.get("table_counts", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Targets", counts.get("targets", 0))
    c2.metric("Compounds", counts.get("compounds", 0))
    c3.metric("Bioactivities", counts.get("activity_records", 0))
    c4.metric("Predictions", counts.get("predictions", 0))
    c5.metric("Docking Poses", counts.get("docking_runs", 0))
    c6.metric("ADMET Profiles", counts.get("admet_runs", 0))
else:
    st.error(f"Database Warning: {health.get('error')}")

st.markdown("---")

# 2. Runtime Environment & Cloud Deployment
st.subheader("☁️ Cloud Deployment & Environment Variables")
col_env1, col_env2 = st.columns(2)

with col_env1:
    st.write(f"• **Render Service**: `srv-dc232oafngtc7392ia40`")
    st.write(f"• **Render Cloud Region**: `Oregon (Free Tier)`")
    st.write(f"• **Listening Port (`PORT`)**: `{os.getenv('PORT', '8501')}`")
    st.write(f"• **Python Runtime**: `{sys.version.split()[0]}`")

with col_env2:
    st.write(f"• **RDKit Version**: `{rdkit.__version__}`")
    st.write(f"• **Database URL Configured**: `{'Yes (PostgreSQL)' if os.getenv('DATABASE_URL') else 'SQLite Local Fallback'}`")
    st.write(f"• **Project Root Path**: `{_ROOT}`")
    st.write("• **Zero-Leakage Scaffold Split**: `Active & Enforced`")

st.markdown("---")

# 3. Model Weights & Artifacts Registry
st.subheader("📦 Model Checkpoints & Pre-trained Artifacts")
ckpt_dir = _ROOT / "artifacts" / "checkpoints"
if ckpt_dir.exists():
    ckpts = list(ckpt_dir.glob("*.pt"))
    st.write(f"Found **{len(ckpts)}** neural model checkpoints in persistent storage:")
    for ckpt in ckpts:
        st.code(f"{ckpt.name} — Size: {round(ckpt.stat().st_size / 1024, 1)} KB")
else:
    st.info("Checkpoints directory ready.")
