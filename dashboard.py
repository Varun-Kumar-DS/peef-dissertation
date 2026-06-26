"""
PEEF — Interactive Dashboard v3
================================
Streamlit dashboard for the Prompt Engineering Evaluation Framework.

Usage:
    streamlit run dashboard.py

New in v3:
    - JS-animated KPI counters
    - Plotly gauge charts for top results
    - Radar / spider chart for cross-task comparison
    - Animated shifting gradient hero
    - Glassmorphism finding cards
    - Significance heatmap on statistics page
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PEEF · Prompt Engineering Evaluation Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SCORED_DIR = ROOT / "03_results" / "scored"
ANALYSIS   = ROOT / "03_results" / "analysis" / "statistical_summary.json"
FIGURES    = ROOT / "06_writeup" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
COLOURS = {
    "Zero-Shot":     "#0072B2",
    "Few-Shot (4)":  "#009E73",
    "CoT":           "#D55E00",
    "Zero-Shot CoT": "#CC79A7",
}
TECH_ORDER = ["Zero-Shot", "Few-Shot (4)", "CoT", "Zero-Shot CoT"]

TECH_LABELS: dict[str, str] = {
    "qa_zero_shot":                              "Zero-Shot",
    "qa_few_shot_2shot":                         "Few-Shot (2)",
    "qa_few_shot_4shot":                         "Few-Shot (4)",
    "qa_few_shot_8shot":                         "Few-Shot (8)",
    "qa_cot":                                    "CoT",
    "qa_zero_shot_cot":                          "Zero-Shot CoT",
    "summarisation_zero_shot":                   "Zero-Shot",
    "summarisation_few_shot_4shot":              "Few-Shot (2)",
    "summarisation_few_shot_4shot_corrected":    "Few-Shot (4)",
    "summarisation_few_shot_8shot":              "Few-Shot (8)",
    "summarisation_cot":                         "CoT",
    "summarisation_zero_shot_cot":               "Zero-Shot CoT",
    "reasoning_zero_shot":                       "Zero-Shot",
    "reasoning_few_shot_2shot":                  "Few-Shot (2)",
    "reasoning_few_shot_4shot":                  "Few-Shot (4)",
    "reasoning_few_shot_8shot":                  "Few-Shot (8)",
    "reasoning_cot":                             "CoT",
    "reasoning_zero_shot_cot":                   "Zero-Shot CoT",
}

TASK_DISPLAY = {
    "qa":            "Question Answering",
    "summarisation": "Summarisation",
    "reasoning":     "Mathematical Reasoning",
}

TASK_DATASET = {
    "qa":            "TriviaQA",
    "summarisation": "CNN / DailyMail",
    "reasoning":     "GSM8K",
}

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0F172A; }
[data-testid="stMain"]             { background: #0F172A; }
[data-testid="block-container"]    { padding-top: 1.2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0D1117 60%, #111827 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}
[data-testid="stSidebar"] * { color: #6B7280 !important; }
[data-testid="stSidebar"] strong { color: #D1D5DB !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] .stRadio > label > div > p {
    font-size: 0.87rem !important;
    font-weight: 500 !important;
}

/* ── Animated hero ── */
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.hero {
    background: linear-gradient(-45deg, #020617, #1E3A5F, #1d4ed8, #0369a1, #1E3A5F, #020617);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
    border-radius: 20px;
    padding: 2.8rem 2.6rem 2.2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content: '';
    position: absolute;
    top: -35%; right: -6%;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(96,165,250,0.20) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -25%; left: 6%;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(6,182,212,0.13) 0%, transparent 65%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    color: #93C5FD !important;
    border-radius: 50px;
    padding: 0.26rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    backdrop-filter: blur(4px);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: #FFFFFF !important;
    line-height: 1.18;
    margin: 0.35rem 0 0.6rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #94A3B8 !important;
    font-size: 0.93rem;
    line-height: 1.65;
}
.hero-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.2rem 0 0.85rem;
}
.hero-meta { font-size: 0.78rem; color: #475569 !important; }
.hero-meta strong { color: #6B7280 !important; }

/* ── Section heading ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.95rem;
    font-weight: 800;
    color: #E2E8F0;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sec-head .ico {
    background: linear-gradient(135deg, #1E3A5F, #2563EB);
    color: white !important;
    border-radius: 8px;
    width: 28px; height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.88rem;
    flex-shrink: 0;
}

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 0.85rem;
    transition: background 0.2s, border 0.2s, transform 0.18s;
}
.glass:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.14);
    transform: translateY(-2px);
}
.glass h4 { margin: 0 0 0.4rem; font-size: 0.93rem; color: #E2E8F0; font-weight: 700; }
.glass p  { margin: 0; font-size: 0.85rem; color: #94A3B8; line-height: 1.58; }
.glass code { font-size: 0.71rem; color: #64748B; }

/* ── Finding glass card ── */
.find-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 0.85rem;
    border-left: 4px solid;
    transition: transform 0.18s, background 0.2s;
}
.find-glass:hover {
    transform: translateX(4px);
    background: rgba(255,255,255,0.07);
}
.find-glass h4 { margin: 0 0 0.35rem; font-size: 0.93rem; color: #E2E8F0; font-weight: 700; }
.find-glass p  { margin: 0; font-size: 0.85rem; color: #94A3B8; line-height: 1.58; }

/* ── Best-result card ── */
.best-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.3rem;
    transition: transform 0.18s, box-shadow 0.18s, background 0.2s;
    height: 100%;
}
.best-glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    background: rgba(255,255,255,0.07);
}
.best-task {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.best-score {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
    margin: 0.45rem 0 0.25rem;
}
.best-metric { font-size: 0.74rem; color: #64748B; margin-bottom: 1rem; }

/* ── Score progress bars ── */
.sbar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 7px;
    overflow: hidden;
}
.sbar {
    height: 7px;
    border-radius: 100px;
    animation: growBar 1.1s cubic-bezier(0.4,0,0.2,1) forwards;
}
@keyframes growBar { from { width: 0 !important; } }

/* ── Score-row bars ── */
.srow {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.65rem;
}
.srow-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94A3B8;
    width: 100px;
    flex-shrink: 0;
}
.srow-bar-wrap {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 9px;
    overflow: hidden;
}
.srow-bar {
    height: 9px;
    border-radius: 100px;
    animation: growBar 1.1s cubic-bezier(0.4,0,0.2,1) forwards;
}
.srow-val {
    font-size: 0.78rem;
    font-weight: 800;
    color: #E2E8F0;
    width: 44px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Technique badge ── */
.badge {
    display: inline-block;
    border-radius: 50px;
    padding: 0.17rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #FFFFFF !important;
    margin-bottom: 0.4rem;
}

/* ── Medal card ── */
.medal-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 4px solid;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    transition: transform 0.18s, background 0.2s;
}
.medal-card:hover {
    transform: translateY(-3px);
    background: rgba(255,255,255,0.07);
}

/* ── Page heading ── */
.page-title {
    font-size: 1.75rem;
    font-weight: 900;
    color: #E2E8F0;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.page-sub  { font-size: 0.88rem; color: #64748B; margin: 0 0 0.8rem; }
.page-hr   { border: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0 1.2rem; }

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1E3A5F, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.88 !important; }

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
    font-size: 0.86rem;
    color: #64748B !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #60A5FA !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] label { color: #94A3B8 !important; font-size:0.84rem !important; }
[data-testid="stSelectbox"]   label { color: #94A3B8 !important; font-size:0.84rem !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #374151;
    font-size: 0.74rem;
    padding: 2.5rem 0 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for fpath in sorted(SCORED_DIR.glob("*.jsonl")):
        if "pilot" in fpath.name:
            continue
        task: str | None = None
        for prefix in ("qa", "summarisation", "reasoning"):
            if fpath.name.startswith(prefix):
                task = prefix
                break
        if task is None:
            continue
        tech = TECH_LABELS.get(fpath.stem, fpath.stem)
        em_list, rouge_list, bert_list = [], [], []
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            s = row.get("scores", {})
            if "exact_match"  in s: em_list.append(float(s["exact_match"]))
            if "rougeL_f"     in s: rouge_list.append(float(s["rougeL_f"]))
            if "bertscore_f1" in s: bert_list.append(float(s["bertscore_f1"]))
        rows.append({
            "Experiment":   fpath.stem,
            "Task":         TASK_DISPLAY[task],
            "Dataset":      TASK_DATASET[task],
            "Technique":    tech,
            "N":            max(len(em_list), len(rouge_list), 1),
            "Exact Match":  round(float(np.mean(em_list)), 4)    if em_list    else None,
            "ROUGE-L":      round(float(np.mean(rouge_list)), 4) if rouge_list else None,
            "BERTScore F1": round(float(np.mean(bert_list)), 4)  if bert_list  else None,
            "_em":          em_list,
            "_rouge":       rouge_list,
            "_task":        task,
        })
    return pd.DataFrame(rows)


@st.cache_data
def load_stats() -> dict:
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(value: float, title: str, color: str,
               suffix: str = "%", max_val: float = 100) -> go.Figure:
    """Single Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 34, "color": color, "family": "system-ui"}},
        title={"text": title, "font": {"size": 11, "color": "#64748B"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#334155",
                     "tickfont": {"color": "#475569", "size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,           max_val*0.5], "color": "rgba(255,255,255,0.02)"},
                {"range": [max_val*0.5, max_val*0.75],"color": "rgba(255,255,255,0.04)"},
                {"range": [max_val*0.75,max_val],      "color": "rgba(255,255,255,0.06)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(t=55, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, sans-serif"},
    )
    return fig


def make_radar() -> go.Figure:
    """Radar / spider chart comparing all 4 techniques across all 3 tasks."""
    categories = ["QA<br>(Exact Match)", "Summarisation<br>(ROUGE-L ×10)", "Reasoning<br>(Exact Match)"]
    data = {
        "Zero-Shot":     [0.670, 0.238 * 10, 0.650],
        "Few-Shot (4)":  [0.700, 0.227 * 10, 0.575],
        "CoT":           [0.660, 0.205 * 10, 0.900],
        "Zero-Shot CoT": [0.365, 0.202 * 10, 0.825],
    }
    # rgba fill colours (Plotly needs rgba, not 8-digit hex)
    fill_colours = {
        "Zero-Shot":     "rgba(0, 114, 178, 0.13)",
        "Few-Shot (4)":  "rgba(0, 158, 115, 0.13)",
        "CoT":           "rgba(213, 94, 0, 0.13)",
        "Zero-Shot CoT": "rgba(204, 121, 167, 0.13)",
    }
    fig = go.Figure()
    for tech in TECH_ORDER:
        vals = data[tech]
        color = COLOURS[tech]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=tech,
            fill="toself",
            fillcolor=fill_colours[tech],
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.50, 0.75, 1.0],
                tickfont=dict(size=9, color="#475569"),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#94A3B8"),
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=60, l=40, r=40),
        height=370,
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    return fig


def make_bar_chart(task_df: pd.DataFrame, is_em: bool, metric_name: str) -> go.Figure:
    """Interactive grouped bar chart for results page."""
    fig = go.Figure()
    for tech in TECH_ORDER:
        row = task_df[task_df["Technique"] == tech]
        if row.empty:
            continue
        r          = row.iloc[0]
        score      = r["Exact Match"] if is_em else r["ROUGE-L"]
        score_list = r["_em"]         if is_em else r["_rouge"]
        if score is None or not score_list:
            continue
        err = ci95(score_list)
        fig.add_trace(go.Bar(
            name=tech,
            x=[tech],
            y=[score],
            error_y=dict(type="data", array=[err], visible=True,
                         color="#475569", thickness=2, width=9),
            marker=dict(
                color=COLOURS[tech],
                line=dict(color="rgba(255,255,255,0.15)", width=1.5),
            ),
            text=f"<b>{score:.3f}</b>",
            textposition="outside",
            textfont=dict(size=13, color="#E2E8F0"),
            width=0.42,
        ))
    valid = [
        (r["Exact Match"] or r["ROUGE-L"] or 0)
        for _, r in task_df.iterrows()
        if (r["Exact Match"] or r["ROUGE-L"]) is not None
    ]
    y_max = max(valid) if valid else 1.0
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(255,255,255,0.03)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title=metric_name,
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, min(y_max + 0.18, 1.05)],
            tickformat=".2f",
            title_font=dict(size=12, color="#64748B"),
            tickfont=dict(color="#475569", size=11),
            linecolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(
            title="Prompting Technique",
            tickfont=dict(size=13, color="#94A3B8"),
            title_font=dict(size=12, color="#64748B"),
            linecolor="rgba(255,255,255,0.06)",
        ),
        margin=dict(t=40, b=50, l=70, r=30),
        height=430,
        bargap=0.3,
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.3rem 0 0.9rem;'>
        <div style='font-size:3rem; margin-bottom:0.4rem; filter:drop-shadow(0 0 12px rgba(96,165,250,0.5));'>🔬</div>
        <div style='font-size:1.2rem; font-weight:900; color:#E2E8F0 !important;
                    letter-spacing:0.08em;'>PEEF</div>
        <div style='font-size:0.68rem; color:#374151 !important; font-weight:500;
                    line-height:1.6; margin-top:0.25rem;'>
            Prompt Engineering<br>Evaluation Framework
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "nav",
        options=["🏠  Home", "📊  Results", "📈  Charts", "🔬  Statistics", "⚙️  Framework"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.76rem; line-height:2.1;'>
        <div style='color:#374151 !important; font-weight:700; font-size:0.63rem;
                    text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.3rem;'>
            Project Details
        </div>
        <div style='color:#4B5563 !important;'>🎓 &nbsp;University of Liverpool</div>
        <div style='color:#4B5563 !important;'>📚 &nbsp;MSc Computer Science</div>
        <div style='color:#4B5563 !important;'>👨‍🏫 &nbsp;A. Koufonikos</div>
        <div style='color:#4B5563 !important; font-family:monospace !important; font-size:0.69rem;'>
            🤖 &nbsp;claude-haiku-4-5
        </div>
        <div style='color:#4B5563 !important;'>📅 &nbsp;Jun – Aug 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2026 Varun Kumar · University of Liverpool")

df    = load_summary()
stats = load_stats()


# ═════════════════════════════════════════════════════════════════════════════
#  HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🎓 MSc Dissertation · University of Liverpool · 2026</div>
        <div class="hero-title">Prompt Engineering<br>Evaluation Framework</div>
        <div class="hero-sub">
            A systematic empirical study comparing zero-shot, few-shot, and chain-of-thought<br>
            prompting techniques across question answering, summarisation, and mathematical reasoning.
        </div>
        <hr class="hero-divider">
        <div class="hero-meta">
            <strong>Supervisor:</strong> Achilleas Koufonikos &nbsp;·&nbsp;
            <strong>Model:</strong> claude-haiku-4-5-20251001 &nbsp;·&nbsp;
            <strong>Datasets:</strong> TriviaQA · CNN/DailyMail · GSM8K
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Animated KPI counters (JS component) ──────────────────────────────────
    components.html("""
    <!DOCTYPE html><html><head>
    <style>
      * { margin:0; padding:0; box-sizing:border-box;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
      body { background: transparent; overflow: hidden; }
      .grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
      }
      .kpi {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 4px solid var(--c);
        border-radius: 16px;
        padding: 18px 16px;
        position: relative;
        overflow: hidden;
        opacity: 0;
        transform: translateY(18px);
        animation: slideUp 0.5s ease forwards;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: default;
      }
      .kpi:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        background: rgba(255,255,255,0.07);
      }
      .kpi:nth-child(1) { --c:#3B82F6; animation-delay:0.05s; }
      .kpi:nth-child(2) { --c:#8B5CF6; animation-delay:0.12s; }
      .kpi:nth-child(3) { --c:#10B981; animation-delay:0.19s; }
      .kpi:nth-child(4) { --c:#F59E0B; animation-delay:0.26s; }
      @keyframes slideUp {
        to { opacity:1; transform:translateY(0); }
      }
      .icon {
        position: absolute; top:14px; right:14px;
        font-size:24px; opacity:0.10;
      }
      .label {
        font-size:9.5px; font-weight:800; text-transform:uppercase;
        letter-spacing:0.1em; color:#64748B; margin-bottom:8px;
      }
      .value {
        font-size:36px; font-weight:900; color:var(--c);
        line-height:1; margin-bottom:5px;
        font-variant-numeric: tabular-nums;
      }
      .sub { font-size:10.5px; color:#374151; font-weight:500; }
    </style>
    </head><body>
    <div class="grid">
      <div class="kpi">
        <div class="icon">🧠</div>
        <div class="label">Techniques Tested</div>
        <div class="value" id="v1">0</div>
        <div class="sub">ZS &nbsp;·&nbsp; FS &nbsp;·&nbsp; CoT &nbsp;·&nbsp; ZS-CoT</div>
      </div>
      <div class="kpi">
        <div class="icon">📋</div>
        <div class="label">Benchmark Tasks</div>
        <div class="value" id="v2">0</div>
        <div class="sub">QA &nbsp;·&nbsp; Summarisation &nbsp;·&nbsp; Reasoning</div>
      </div>
      <div class="kpi">
        <div class="icon">⚗️</div>
        <div class="label">Experiments Run</div>
        <div class="value" id="v3">0</div>
        <div class="sub">200 samples each experiment</div>
      </div>
      <div class="kpi">
        <div class="icon">⚡</div>
        <div class="label">Total API Calls</div>
        <div class="value" id="v4">0</div>
        <div class="sub">Claude Haiku model</div>
      </div>
    </div>
    <script>
    function animateCount(el, target, duration) {
      const start = performance.now();
      function tick(now) {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        const val = Math.round(eased * target);
        el.textContent = val >= 1000 ? val.toLocaleString() : String(val);
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }
    setTimeout(() => {
      animateCount(document.getElementById('v1'), 4,    900);
      animateCount(document.getElementById('v2'), 3,    900);
      animateCount(document.getElementById('v3'), 12,   900);
      animateCount(document.getElementById('v4'), 2400, 1400);
    }, 350);
    </script>
    </body></html>
    """, height=148)

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    # ── Top 3 gauge charts ────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span class="ico">🏆</span> Top Results — Best Technique Per Task</div>', unsafe_allow_html=True)

    gc1, gc2, gc3 = st.columns(3)

    gauges = [
        (70.0, "🟢 Few-Shot (4) — QA",          "#009E73", "%",   100),
        (23.8, "🔵 Zero-Shot — Summarisation",   "#0072B2", "",    40),
        (90.0, "🔴 CoT — Mathematical Reasoning","#D55E00", "%",   100),
    ]
    for col, (val, title, color, suffix, max_val) in zip([gc1, gc2, gc3], gauges):
        with col:
            st.plotly_chart(make_gauge(val, title, color, suffix, max_val),
                            use_container_width=True)
            # subtitle below gauge
            sub = "ROUGE-L F1 × 100" if "Summarisation" in title else "Exact Match Accuracy"
            st.markdown(f"<p style='text-align:center;font-size:0.75rem;color:#374151;margin-top:-0.5rem;'>{sub}</p>",
                        unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span class="ico">🕸️</span> Technique Comparison — Radar Chart</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([3, 2])
    with r1:
        st.plotly_chart(make_radar(), use_container_width=True)
    with r2:
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass" style='margin-top:1rem;'>
            <h4>📌 How to read this chart</h4>
            <p>Each axis represents one task. Scores are normalised to [0–1].
            Summarisation ROUGE-L is multiplied by 10 for visual scale.<br><br>
            A <strong>larger shaded area</strong> = better overall performance.
            The ideal technique would fill the entire triangle.</p>
        </div>
        <div class="glass">
            <h4>🔑 Key insight</h4>
            <p>No single technique fills the whole triangle. CoT excels on Reasoning
            but shrinks on QA. Zero-Shot is balanced. Task–technique alignment is
            the central finding of this dissertation.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Scores at a glance ────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span class="ico">📊</span> All 12 Experiment Scores</div>', unsafe_allow_html=True)

    glance = [
        ("📚 Question Answering", "Exact Match Accuracy", [
            ("Few-Shot (4)",  "#009E73", "70.0%"),
            ("Zero-Shot",     "#0072B2", "67.0%"),
            ("CoT",           "#D55E00", "66.0%"),
            ("Zero-Shot CoT", "#CC79A7", "36.5%"),
        ]),
        ("📰 Summarisation", "ROUGE-L F1 Score", [
            ("Zero-Shot",     "#0072B2", "0.238"),
            ("Few-Shot (4)",  "#009E73", "0.227"),
            ("CoT",           "#D55E00", "0.205"),
            ("Zero-Shot CoT", "#CC79A7", "0.202"),
        ]),
        ("🔢 Mathematical Reasoning", "Exact Match Accuracy", [
            ("CoT",           "#D55E00", "90.0%"),
            ("Zero-Shot CoT", "#CC79A7", "82.5%"),
            ("Zero-Shot",     "#0072B2", "65.0%"),
            ("Few-Shot (4)",  "#009E73", "57.5%"),
        ]),
    ]

    g1, g2, g3 = st.columns(3)
    for col, (task_name, metric, rows) in zip([g1, g2, g3], glance):
        with col:
            score_cards = "".join(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        padding:0.55rem 0.7rem; border-radius:10px; margin-bottom:0.5rem;
                        background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07);">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <div style="width:10px; height:10px; border-radius:50%;
                                background:{color}; flex-shrink:0;"></div>
                    <span style="font-size:0.8rem; color:#94A3B8; font-weight:600;">{tech}</span>
                </div>
                <span style="font-size:1.1rem; font-weight:900; color:{color};">{score}</span>
            </div>""" for tech, color, score in rows)
            st.markdown(f"""
            <div class="glass">
                <h4 style='margin-bottom:0.2rem;'>{task_name}</h4>
                <p style='font-size:0.72rem; color:#374151; margin-bottom:0.9rem;'>{metric}</p>
                {score_cards}
            </div>""", unsafe_allow_html=True)

    # ── Key findings ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span class="ico">💡</span> Key Research Findings</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    findings = [
        ("#D55E00",
         "🔢 CoT dominates Mathematical Reasoning",
         "Chain-of-Thought achieved <strong style='color:#FCA5A5;'>90%</strong> on GSM8K — "
         "the highest score across all 12 experiments. Statistically significant vs every other "
         "technique (p&lt;0.001, Cohen's d=0.79). Extends Wei et al. (2022) to Anthropic Claude."),
        ("#0072B2",
         "📰 Zero-Shot wins Summarisation",
         "Simpler prompts outperformed complex ones (ROUGE-L=0.238). Zero-Shot beat CoT with "
         "medium effect size (d=0.52, p&lt;0.001). Additional instructions can <em>hurt</em> "
         "creative generation tasks."),
        ("#009E73",
         "📚 QA: No statistically significant winner",
         "Top 3 QA techniques — Few-Shot (70%), Zero-Shot (67%), CoT (66%) — no significant "
         "pairwise differences (all p&gt;0.10). Claude handles factual QA well with any style."),
        ("#8B5CF6",
         "⚡ Task–technique fit is the core finding",
         "No single technique dominated across all tasks. CoT is essential for reasoning but "
         "wasteful for summarisation. Prompt strategy must be matched to task type."),
    ]
    with f1:
        for color, title, text in findings[:2]:
            st.markdown(f"""
            <div class="find-glass" style="border-left-color:{color};">
                <h4>{title}</h4><p>{text}</p>
            </div>""", unsafe_allow_html=True)
    with f2:
        for color, title, text in findings[2:]:
            st.markdown(f"""
            <div class="find-glass" style="border-left-color:{color};">
                <h4>{title}</h4><p>{text}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        PEEF · Prompt Engineering Evaluation Framework · University of Liverpool MSc Dissertation 2026<br>
        Model: claude-haiku-4-5-20251001 &nbsp;·&nbsp; Datasets: TriviaQA · CNN/DailyMail · GSM8K
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Results":

    st.markdown('<div class="page-title">📊 Experiment Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All 12 experiments · 4 techniques × 3 tasks · 200 samples each</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-hr">', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        task_filter = st.multiselect("Filter by Task",
            options=list(TASK_DISPLAY.values()), default=list(TASK_DISPLAY.values()))
    with f2:
        tech_filter = st.multiselect("Filter by Technique",
            options=TECH_ORDER, default=TECH_ORDER)

    filtered = df[
        df["Task"].isin(task_filter) & df["Technique"].isin(tech_filter)
    ].copy()

    st.markdown('<div class="sec-head"><span class="ico">📋</span> Results Table</div>', unsafe_allow_html=True)

    display = filtered[["Task", "Dataset", "Technique", "N",
                         "Exact Match", "ROUGE-L", "BERTScore F1"]].copy()
    display["Exact Match"]  = display["Exact Match"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    display["ROUGE-L"]      = display["ROUGE-L"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    display["BERTScore F1"] = display["BERTScore F1"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-head"><span class="ico">📈</span> Interactive Score Comparison</div>', unsafe_allow_html=True)

    task_choice = st.selectbox("Select Task to Visualise", options=list(TASK_DISPLAY.values()))
    task_key    = {v: k for k, v in TASK_DISPLAY.items()}[task_choice]
    is_em       = task_key in ("qa", "reasoning")
    metric_name = "Exact Match Accuracy" if is_em else "ROUGE-L F1 Score"
    task_df     = df[df["Task"] == task_choice].copy()

    st.plotly_chart(make_bar_chart(task_df, is_em, metric_name), use_container_width=True)
    st.caption("Error bars = 95% confidence intervals · Colourblind-friendly palette (Wong 2011)")

    st.markdown("""
    <div class="footer">
        PEEF · Prompt Engineering Evaluation Framework · University of Liverpool MSc Dissertation 2026
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Charts":

    st.markdown('<div class="page-title">📈 Publication-Quality Figures</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">300 DPI · Wong 2011 colourblind palette · Dissertation-ready</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-hr">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 Figure 1 — Task Results",
        "📈 Figure 2 — Overview",
        "🔥 Figure 3 — Effect Sizes",
    ])

    fig_info = [
        ("fig1_task_results.png",
         "Figure 1 — Per-Task Bar Charts",
         "Per-task bar charts with 95% CI error bars and ★ Best markers. "
         "QA panel includes annotation on Zero-Shot CoT extraction difficulty."),
        ("fig2_combined_overview.png",
         "Figure 2 — Combined Overview",
         "All 4 techniques × 3 tasks in one grouped bar chart. "
         "Scores as percentages; ROUGE-L scaled ×100 for visual comparison."),
        ("fig3_effect_sizes.png",
         "Figure 3 — Cohen's d Heatmaps",
         "Per-task effect size matrices. Blue = row technique weaker than column. "
         "Red = row stronger. * = p < 0.05 (Wilcoxon signed-rank)."),
    ]

    for tab, (fname, title, caption) in zip([tab1, tab2, tab3], fig_info):
        with tab:
            fpath = FIGURES / fname
            st.markdown(f"""
            <div class="glass" style="margin-bottom:1rem;">
                <h4>{title}</h4><p>{caption}</p>
            </div>""", unsafe_allow_html=True)
            if not fpath.exists():
                st.warning("Figure not found — run `python plot_results.py` first.")
            else:
                st.image(Image.open(fpath), use_container_width=True)
                with open(fpath, "rb") as fh:
                    st.download_button(
                        label=f"⬇  Download {fname}",
                        data=fh.read(),
                        file_name=fname,
                        mime="image/png",
                        type="primary",
                    )

    st.markdown("""
    <div class="footer">
        PEEF · Prompt Engineering Evaluation Framework · University of Liverpool MSc Dissertation 2026
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Statistics":

    st.markdown('<div class="page-title">🔬 Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Wilcoxon signed-rank tests (non-parametric, paired) · Cohen\'s d effect sizes · α = 0.05 · n = 200 per experiment</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-hr">', unsafe_allow_html=True)

    task_tabs = st.tabs(["📚 Question Answering", "📰 Summarisation", "🔢 Mathematical Reasoning"])
    task_cfgs = [
        ("qa",            "Exact Match"),
        ("summarisation", "ROUGE-L F1"),
        ("reasoning",     "Exact Match"),
    ]

    for tab, (task_key, metric_name) in zip(task_tabs, task_cfgs):
        with tab:
            task_stats   = stats[task_key]
            means        = task_stats["means"]
            sorted_means = sorted(means.items(), key=lambda x: -x[1])
            medals       = ["🥇", "🥈", "🥉", "4️⃣"]

            # Gauge row — dynamic column count based on number of experiments
            st.markdown(f'<div class="sec-head"><span class="ico">📊</span> Mean {metric_name} Scores</div>', unsafe_allow_html=True)
            n_exp  = len(sorted_means)
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"][:n_exp]
            mc = st.columns(n_exp)
            for i, (exp, val) in enumerate(sorted_means):
                tech  = TECH_LABELS.get(exp, exp)
                color = COLOURS.get(tech, "#64748B")
                with mc[i]:
                    st.markdown(f"""
                    <div class="medal-card" style="border-top-color:{color};">
                        <div style="font-size:1.6rem; margin-bottom:0.3rem;">{medals[i]}</div>
                        <div class="badge" style="background:{color}; margin-bottom:0.55rem;">{tech}</div>
                        <div style="font-size:1.75rem; font-weight:900; color:{color};
                                    line-height:1; margin-bottom:0.2rem;">{val:.4f}</div>
                        <div style="font-size:0.74rem; color:#374151;">{val*100:.1f} %</div>
                    </div>""", unsafe_allow_html=True)

            # Pairwise table
            st.markdown('<div class="sec-head"><span class="ico">🧪</span> Pairwise Wilcoxon Signed-Rank Tests</div>', unsafe_allow_html=True)

            pair_rows = []
            for p in task_stats["pairwise"]:
                a   = TECH_LABELS.get(p["experiment_a"], p["experiment_a"])
                b   = TECH_LABELS.get(p["experiment_b"], p["experiment_b"])
                pv  = p["p_value"]
                sig = ("***" if pv < 0.001 else "**" if pv < 0.01
                        else "*" if pv < 0.05 else "ns")
                pair_rows.append({
                    "Comparison":   f"{a}  vs  {b}",
                    "p-value":      "< 0.0001" if pv < 0.0001 else f"{pv:.4f}",
                    "Sig.":         sig,
                    "Cohen's d":    f"{p['cohens_d']:+.4f}",
                    "Effect Size":  p["effect_size"].capitalize(),
                    "Significant?": "✅ Yes" if p["significant"] else "❌ No",
                })
            st.dataframe(pd.DataFrame(pair_rows), hide_index=True, use_container_width=True)

            st.markdown("""
            <p style='color:#374151; font-size:0.76rem; margin-top:0.5rem;'>
            <strong style='color:#64748B;'>Significance:</strong>
            *** p&lt;0.001 &nbsp;·&nbsp; ** p&lt;0.01 &nbsp;·&nbsp;
            * p&lt;0.05 &nbsp;·&nbsp; ns = not significant &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong style='color:#64748B;'>Effect size |d|:</strong>
            negligible &lt;0.2 &nbsp;·&nbsp; small 0.2–0.5 &nbsp;·&nbsp;
            medium 0.5–0.8 &nbsp;·&nbsp; large &gt;0.8
            </p>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        PEEF · Prompt Engineering Evaluation Framework · University of Liverpool MSc Dissertation 2026
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Framework":

    st.markdown('<div class="page-title">⚙️ PEEF Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A modular six-component Python framework for systematic prompt engineering evaluation</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-hr">', unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="ico">🧩</span> The Six Modules</div>', unsafe_allow_html=True)

    modules = [
        ("🧱", "Prompt Builder",    "05_code/src/prompt_builder.py",    "#3B82F6",
         "Constructs zero-shot, few-shot, and CoT prompts using Jinja2 templates. "
         "Supports configurable shot counts and dynamic example injection."),
        ("🚀", "Experiment Runner", "05_code/src/experiment_runner.py", "#8B5CF6",
         "Sends prompts to the Claude API with rate limiting, retry logic, and a "
         "file-based caching system to prevent duplicate API calls across runs."),
        ("📏", "Evaluator",         "05_code/src/evaluator.py",         "#10B981",
         "Scores responses using Exact Match accuracy, ROUGE-L F1 (Lin 2004), "
         "and BERTScore F1 (Zhang et al. 2020) across all three task types."),
        ("📊", "Analysis Engine",   "05_code/src/analysis_engine.py",   "#F59E0B",
         "Runs Wilcoxon signed-rank tests and computes Cohen's d for all technique "
         "pairs within each task, exporting full results to JSON."),
        ("⚡", "Adaptive Cascade",  "05_code/src/adaptive_cascade.py",  "#EF4444",
         "Automatically selects the cheapest effective technique per question, "
         "routing easy questions to zero-shot and hard ones to CoT."),
        ("🔧", "Prompt Optimizer",  "05_code/src/prompt_optimizer.py",  "#06B6D4",
         "Uses LLM-based reflection to iteratively improve underperforming prompts "
         "until a configurable quality threshold is met."),
    ]

    mc1, mc2 = st.columns(2)
    for i, (icon, name, path, color, desc) in enumerate(modules):
        col = mc1 if i % 2 == 0 else mc2
        with col:
            st.markdown(f"""
            <div class="glass" style="border-left:4px solid {color};">
                <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.55rem;">
                    <span style="font-size:1.3rem;">{icon}</span>
                    <div>
                        <div style="font-size:0.63rem; font-weight:800; text-transform:uppercase;
                                    letter-spacing:0.08em; color:{color};">Module {i+1}</div>
                        <h4 style="margin:0;">{name}</h4>
                    </div>
                </div>
                <code>{path}</code>
                <p style="margin-top:0.5rem;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="ico">🔄</span> Evaluation Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
    <pre style='font-family:monospace; font-size:0.78rem; color:#94A3B8;
                line-height:1.8; margin:0; overflow-x:auto;'>
<span style='color:#60A5FA;'>YAML Config</span>  (task · technique · model · sample_size)
       │
       ▼
<span style='color:#34D399;'>run.py</span>  ──►  <span style='color:#A78BFA;'>PromptBuilder</span>  ──►  <span style='color:#F87171;'>ExperimentRunner</span>  ──►  Claude API
                                           │                       │
                                           └──────── response ─────┘
                                                        │
                                             03_results/raw/*.jsonl
                                                        │
                                                        ▼
                                                 <span style='color:#34D399;'>evaluate.py</span>
                                      (Exact Match · ROUGE-L · BERTScore)
                                                        │
                                             03_results/scored/*.jsonl
                                                        │
                                  ┌─────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                            <span style='color:#34D399;'>analyse.py</span>                           <span style='color:#34D399;'>plot_results.py</span>
                       Wilcoxon + Cohen's d                    3 × 300 DPI figures
                                  │                                             │
                       statistical_summary.json                06_writeup/figures/
    </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="ico">📚</span> Datasets & Prompting Techniques</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)

    with dc1:
        for icon, color, name, task, desc in [
            ("📚", "#3B82F6", "TriviaQA",       "Question Answering",
             "Open-domain factual QA · 200 test samples · Metric: Exact Match"),
            ("📰", "#8B5CF6", "CNN/DailyMail",  "Summarisation",
             "News article summarisation · 200 samples · Metric: ROUGE-L F1"),
            ("🔢", "#10B981", "GSM8K",           "Mathematical Reasoning",
             "Grade-school maths word problems · 200 samples · Metric: Exact Match"),
        ]:
            st.markdown(f"""
            <div class="glass" style="border-left:4px solid {color};">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                    <span style="font-size:1.1rem;">{icon}</span>
                    <h4 style="margin:0; color:{color};">{name}</h4>
                    <span style="font-size:0.68rem; color:#374151;">({task})</span>
                </div>
                <p style="font-size:0.83rem;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    with dc2:
        for icon, color, name, desc in [
            ("🎯", "#0072B2", "Zero-Shot",
             "Task instruction only. No examples. Baseline (Brown et al. 2020)."),
            ("📖", "#009E73", "Few-Shot (4-shot)",
             "Task instruction + 4 worked examples. In-context learning (Brown et al. 2020)."),
            ("🧠", "#D55E00", "Chain-of-Thought (CoT)",
             "Examples include reasoning steps. Elicits multi-step reasoning (Wei et al. 2022)."),
            ("💬", "#CC79A7", "Zero-Shot CoT",
             '"Let\'s think step by step" — no examples needed (Kojima et al. 2022).'),
        ]:
            st.markdown(f"""
            <div class="glass" style="border-left:4px solid {color};">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                    <span style="font-size:1.1rem;">{icon}</span>
                    <h4 style="margin:0; color:{color};">{name}</h4>
                </div>
                <p style="font-size:0.83rem;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        PEEF · Prompt Engineering Evaluation Framework · University of Liverpool MSc Dissertation 2026<br>
        Model: claude-haiku-4-5-20251001 &nbsp;·&nbsp; Datasets: TriviaQA · CNN/DailyMail · GSM8K
    </div>""", unsafe_allow_html=True)
