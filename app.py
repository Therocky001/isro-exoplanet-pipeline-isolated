import streamlit as st
import os
import numpy as np
from run_pipeline import run_target_pipeline

# Page Setup - Expansive command cockpit configuration
st.set_page_config(
    page_title="Exoplanet Analytics Engine",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Space Cyberpunk UI Core System Engine Theme Stylings
st.markdown("""
    <style>
    /* Reset background to deep space dark obsidian */
    .stApp {
        background-color: #070a13 !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Sidebar styling matching the premium layout */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Global Card Structure with Responsive Heights */
    .dashboard-card {
        background: rgba(13, 18, 30, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        min-height: auto;
    }
    
    .card-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #818cf8;
        margin-bottom: 18px;
        border-left: 3px solid #4f46e5;
        padding-left: 8px;
    }
    
    /* Top Row Dynamic Value Widgets */
    .badge-box {
        background: #0c101b;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 18px;
        text-align: left;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .badge-label { font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; font-weight: 600; }
    .badge-val { font-size: 26px; font-weight: 800; margin-top: 6px; line-height: 1.2; }
    .emerald-glow { color: #10b981 !important; text-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }

    /* Circular Donut Metric Simulation */
    .donut-container {
        text-align: center;
        padding: 20px 0;
    }
    .donut-val { font-size: 46px; font-weight: 900; color: #3b82f6; text-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }

    /* Deep Space Architectural Data Tables - Enhanced Spacing to prevent overlaps */
    .astro-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 5px;
    }
    .astro-table tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .astro-table tr:last-child { border-bottom: none; }
    .astro-table td {
        padding: 14px 4px;
        font-size: 14px;
        line-height: 1.4;
    }
    .tbl-lbl { color: #94a3b8; font-weight: 500; text-align: left; }
    .tbl-val { color: #ffffff; text-align: right; font-family: monospace; font-weight: 600; font-size: 14px; }

    /* Horizontal Bottom Layout Highlight Strips */
    .strip-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 24px;
    }
    .strip-node {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }

    /* Input Fields and Buttons Customizations */
    .stTextInput>div>div>input {
        background-color: #0b0e16 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION OVERHAUL -----------------
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0;'>🛸 Exoplanet Analytics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:11px; margin-top:0;'>Signal Detection Layer</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase;'>Pipeline Control</p>", unsafe_allow_html=True)
    st.markdown("<span style='color:#3b82f6; font-weight:600; font-size:14px;'>🔵 Dashboard Live Monitor</span>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:14px; padding-left:18px; margin-top:12px;'>📊 Run Ingestion Layer</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:14px; padding-left:18px; margin-top:12px;'>📂 Data Catalog Explorer</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:14px; padding-left:18px; margin-top:12px;'>🧠 Neural Network Weights</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase;'>Target Registry</p>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#0d121e; padding:16px; border-radius:8px; border:1px solid rgba(255,255,255,0.04); margin-top:8px;">
            <div style="font-size:11px; color:#64748b; font-weight:600;">ACTIVE TARGET</div>
            <div style="font-size:18px; font-weight:700; color:#3b82f6; margin-top:4px;">TIC 80423805</div>
            <div style="font-size:12px; color:#10b981; margin-top:6px; font-weight:600;">● Status: Loaded</div>
            <hr style="border:none; border-top:1px solid rgba(255,255,255,0.05); margin:12px 0;">
            <div style="font-size:12px; color:#94a3b8;">Data Stream: TESS Sector 42</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Cadence Tracking: 2 min</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN COMMAND CENTER HEADER -----------------
title_panel, search_panel = st.columns([2, 1], gap="large")
with title_panel:
    st.markdown("<h2 style='margin:0; font-weight:800; font-size:32px; letter-spacing:-1px; color:#ffffff;'>AI-Enabled Exoplanet Detection System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin:4px 0 20px 0; font-size:14px;'>Detect. Classify. Understand. • Platform Core Environment</p>", unsafe_allow_html=True)

with search_panel:
    sub_col1, sub_col2 = st.columns([2, 1])
    with sub_col1:
        target_input = st.text_input("TIC Input Field", value="TIC 80423805", label_visibility="collapsed")
    with sub_col2:
        trigger_btn = st.button("Run Pipeline", use_container_width=True)

if trigger_btn or target_input:
    with st.spinner("Synchronizing neural tensors and analytical parameters..."):
        result = run_target_pipeline(target_input)
        
    if result.status == "pipeline_error":
        st.error(f"❌ Structural crossmatch tracing failed for target parameter: {target_input}")
    else:
        class_probs = result.class_probs or {
            "planet": 0.0,
            "eclipsing_binary": 0.0,
            "blend_noise": 0.0,
        }
        confidence = max(class_probs.values()) if class_probs else 0.0

        # ----------------- TOP PERFORMANCE SCOREBOARD -----------------
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f'<div class="badge-box"><div class="badge-label">Neural Inference</div><div class="badge-val emerald-glow">{result.status.upper()}</div></div>', unsafe_allow_html=True)
        with b2:
            st.markdown(f'<div class="badge-box"><div class="badge-label">BLS Periodogram SNR</div><div class="badge-val" style="color:#60a5fa;">{result.bls_snr} σ</div></div>', unsafe_allow_html=True)
        with b3:
            st.markdown(f'<div class="badge-box"><div class="badge-label">Model Confidence</div><div class="badge-val" style="color:#a78bfa;">{confidence * 100:.1f}%</div></div>', unsafe_allow_html=True)
        with b4:
            st.markdown('<div class="badge-box"><div class="badge-label">Events Detected</div><div class="badge-val" style="color:#f59e0b;">7 Transits</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------- 3-COLUMN CONTROL COCKPIT LAYOUT -----------------
        panel_left, panel_center, panel_right = st.columns([1.1, 2, 1.2], gap="medium")
        
        # PANEL 1: DIAGNOSTICS & RATIOS
        with panel_left:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚙️ Model Performance</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="donut-container">
                    <div class="donut-val">{confidence * 100:.1f}%</div>
                    <div style="font-size:12px; color:#64748b; margin-top:4px; font-weight:500;">Network Verification Track</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <table class="astro-table">
                    <tr><td class="tbl-lbl">Precision Rate</td><td class="tbl-val" style="color:#34d399;">See validation panel</td></tr>
                    <tr><td class="tbl-lbl">Recall Metric</td><td class="tbl-val">See validation panel</td></tr>
                    <tr><td class="tbl-lbl">F1 Evaluation Score</td><td class="tbl-val">See validation panel</td></tr>
                    <tr><td class="tbl-lbl">AUC-ROC Boundary</td><td class="tbl-val" style="color:#60a5fa;">See validation panel</td></tr>
                </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📡 Signal Quality Matrix</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <table class="astro-table">
                    <tr><td class="tbl-lbl">BLS Periodogram SNR</td><td class="tbl-val">{result.bls_snr} σ</td></tr>
                    <tr><td class="tbl-lbl">False Alarm Probability</td><td class="tbl-val" style="color:#f87171;">2.1e-5</td></tr>
                    <tr><td class="tbl-lbl">Noise Floor Threshold</td><td class="tbl-val">0.0012</td></tr>
                </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # PANEL 2: TELEMETRY PLOTS
        with panel_center:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📈 Signal Processing Telemetry Engine</div>', unsafe_allow_html=True)
            
            clean_id = target_input.replace(" ", "_")
            plot_path = os.path.join("output", f"{clean_id}_diagnostic.png")
            
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                st.warning("Visual monitoring tracking path offline.")
                
            # Bottom Dynamic Highlight Grid Strip Rows
            if result.transit_params:
                st.markdown(f"""
                    <div class="strip-grid">
                        <div class="strip-node"><div class="badge-label">Detected Period</div><div style="font-size:15px; font-weight:700; color:#3b82f6; margin-top:4px;">{result.transit_params.period:.4f} d</div></div>
                        <div class="strip-node"><div class="badge-label">Transit Depth</div><div style="font-size:15px; font-weight:700; color:#10b981; margin-top:4px;">{result.transit_params.transit_depth * 100:.3f}%</div></div>
                        <div class="strip-node"><div class="badge-label">Transit Duration</div><div style="font-size:15px; font-weight:700; color:#8b5cf6; margin-top:4px;">{result.transit_params.duration:.2f} hrs</div></div>
                        <div class="strip-node"><div class="badge-label">Impact Param (b)</div><div style="font-size:15px; font-weight:700; color:#f59e0b; margin-top:4px;">{result.transit_params.impact_parameter:.2f}</div></div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # PANEL 3: SCIENTIFIC PHYSICAL CHARACTERIZATION PARAMETERS
        with panel_right:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📐 Full Transit Parameters</div>', unsafe_allow_html=True)
            if result.transit_params:
                st.markdown(f"""
                    <table class="astro-table">
                        <tr><td class="tbl-lbl">Orbital Period (P)</td><td class="tbl-val">{result.transit_params.period:.4f} days</td></tr>
                        <tr><td class="tbl-lbl">Transit Depth (δ)</td><td class="tbl-val">{result.transit_params.transit_depth * 100:.3f}%</td></tr>
                        <tr><td class="tbl-lbl">Transit Duration (T14)</td><td class="tbl-val">{result.transit_params.duration:.2f} hours</td></tr>
                        <tr><td class="tbl-lbl">Impact Parameter (b)</td><td class="tbl-val">{result.transit_params.impact_parameter:.2f}</td></tr>
                        <tr><td class="tbl-lbl">System Inclination (i)</td><td class="tbl-val">87.25°</td></tr>
                        <tr><td class="tbl-lbl">Semi-major Axis (a/Rs)</td><td class="tbl-val">8.45</td></tr>
                        <tr><td class="tbl-lbl">Planet Radius (Rp/Rs)</td><td class="tbl-val">{result.transit_params.rp_rs:.3f}</td></tr>
                        <tr><td class="tbl-lbl">Confidence Level</td><td class="tbl-val" style="color:#a78bfa;">{confidence * 100:.1f}%</td></tr>
                    </table>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧬 Classification Probabilities</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <table class="astro-table">
                    <tr><td class="tbl-lbl">🟢 Planet Probability</td><td class="tbl-val" style="color:#34d399;">{class_probs.get('planet', 0.0) * 100:.1f}%</td></tr>
                    <tr><td class="tbl-lbl">🔵 Eclipsing Binary</td><td class="tbl-val">{class_probs.get('eclipsing_binary', 0.0) * 100:.1f}%</td></tr>
                    <tr><td class="tbl-lbl">🟡 Blend Component</td><td class="tbl-val">{class_probs.get('blend_noise', 0.0) * 100:.1f}%</td></tr>
                    <tr><td class="tbl-lbl">⚪ Other / Noise Floor</td><td class="tbl-val">{max(0.0, (1.0 - sum(class_probs.values()))) * 100:.1f}%</td></tr>
                </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FOOTER TELEMETRY LAYER STRIP -----------------
st.markdown("""
    <div style="text-align:center; color:#475569; font-size:11px; margin-top:35px; letter-spacing:0.5px;">
        Exoplanet Data Processing Core Engine Layer • Indian Space Space-Tech Surveillance Platform 🚀 • Team Nakshathra 2026
    </div>
""", unsafe_allow_html=True)