"""
ui/app.py
=========
Phase 5: Industry-Level Streamlit Dashboard

Architecture
------------
Three-page Streamlit app with a persistent sidebar navigator:

1. **Dashboard** — real-time stress check-in with Plotly gauge, confidence
   bar, empathetic messaging, breathing animation, word-attention heatmap,
   and one-click feedback.

2. **History & Analytics** — multi-tab section showing a timeline chart with
   adaptive threshold, a stress-level distribution donut chart, and a
   session log with CSV export.

3. **Settings** — account info, live model metadata from ``/model/info``,
   and sign-out.

Design System
-------------
- Font stack: system-ui (loads instantly; no external CDN needed).
- Color palette: Indigo primary, semantic level colours, neutral grays.
- Cards: white with subtle box-shadow.
- All charts: Plotly with transparent paper / white card backgrounds.

Connects to the FastAPI backend at ``API_URL``
(default: ``http://localhost:8000``).

Usage
-----
    streamlit run ui/app.py
"""

from __future__ import annotations

import datetime
import hashlib
import io
import os
import re
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    go = None
    _PLOTLY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
_MAX_HISTORY_ITEMS = 200

# ---------------------------------------------------------------------------
# Design system — Warm Limestone palette
# ---------------------------------------------------------------------------

BG_COLOR        = "#F5F0EB"   # warm off-white linen
CARD_BG         = "#FDFAF7"   # warm white for cards
TEXT_MAIN       = "#2C2318"   # deep warm brown
TEXT_MUTED      = "#8A7A6E"   # warm taupe (darkened for readability)
ACCENT          = "#C17A47"   # terracotta / warm amber
ACCENT_LIGHT    = "#F5EBE0"   # pale peach tint
LEVEL_LOW       = "#3D7A52"   # muted sage green (darkened for contrast)
LEVEL_MODERATE  = "#B8860B"   # dark goldenrod (readable on light bg)
LEVEL_HIGH      = "#A03030"   # muted deep red (readable)
LEVEL_UNCERTAIN = "#7A6E65"   # warm taupe-gray
BORDER_COLOR    = "#E2D8CE"   # warm sand border
SIDEBAR_BG      = "#EDE6DC"   # warm parchment
GAUGE_GREEN     = "#E8F0E9"
GAUGE_AMBER     = "#F5EDDB"
GAUGE_RED       = "#F2E0DF"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CSS = (
    "<style>"
    # ── Google Fonts ──
    "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');"

    # ── Base app ──
    ".stApp {{ background-color: {bg}; color: {txt}; font-family: 'DM Sans', sans-serif; }}"

    # ── Sidebar — warm parchment, no animation ──
    "section[data-testid=stSidebar] {{"
    "  background: {sidebar} !important;"
    "  border-right: 1px solid {border};"
    "}}"

    # ── Fix Streamlit's default button readability ──
    # Primary buttons — solid terracotta, white text
    ".stButton > button[kind='primary'], .stButton > button[data-testid='baseButton-primary'] {{"
    "  background-color: {accent} !important;"
    "  color: #FFFFFF !important;"
    "  border: none !important;"
    "  border-radius: 10px !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-weight: 500 !important;"
    "  font-size: 0.9rem !important;"
    "  padding: 0.55rem 1.4rem !important;"
    "  letter-spacing: 0.02em !important;"
    "}}"
    ".stButton > button[kind='primary']:hover, .stButton > button[data-testid='baseButton-primary']:hover {{"
    "  background-color: #A86035 !important;"
    "  color: #FFFFFF !important;"
    "}}"
    # Secondary / default buttons — warm outline style
    ".stButton > button[kind='secondary'], .stButton > button[data-testid='baseButton-secondary'],"
    ".stButton > button:not([kind]) {{"
    "  background-color: {card} !important;"
    "  color: {txt} !important;"
    "  border: 1.5px solid {border} !important;"
    "  border-radius: 10px !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-weight: 500 !important;"
    "  font-size: 0.88rem !important;"
    "}}"
    ".stButton > button[kind='secondary']:hover, .stButton > button:not([kind]):hover {{"
    "  border-color: {accent} !important;"
    "  color: {accent} !important;"
    "  background-color: {acl} !important;"
    "}}"
    # Form submit buttons (full-width inside st.form)
    ".stFormSubmitButton > button {{"
    "  background-color: {accent} !important;"
    "  color: #FFFFFF !important;"
    "  border: none !important;"
    "  border-radius: 10px !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-weight: 500 !important;"
    "  font-size: 0.9rem !important;"
    "  width: 100% !important;"
    "}}"
    ".stFormSubmitButton > button:hover {{"
    "  background-color: #A86035 !important;"
    "  color: #FFFFFF !important;"
    "}}"
    # Download button
    ".stDownloadButton > button {{"
    "  background-color: {card} !important;"
    "  color: {txt} !important;"
    "  border: 1.5px solid {border} !important;"
    "  border-radius: 10px !important;"
    "  font-weight: 500 !important;"
    "}}"
    ".stDownloadButton > button:hover {{"
    "  border-color: {accent} !important;"
    "  color: {accent} !important;"
    "}}"

    # ── Fix text inputs ──
    ".stTextInput > div > div > input, .stTextArea > div > div > textarea {{"
    "  background-color: {card} !important;"
    "  color: {txt} !important;"
    "  border: 1.5px solid {border} !important;"
    "  border-radius: 10px !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-size: 0.92rem !important;"
    "}}"
    ".stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {{"
    "  border-color: {accent} !important;"
    "  box-shadow: 0 0 0 3px rgba(193,122,71,0.12) !important;"
    "}}"
    ".stTextInput label, .stTextArea label {{"
    "  color: {txt} !important;"
    "  font-weight: 500 !important;"
    "  font-size: 0.85rem !important;"
    "}}"

    # ── Fix Streamlit tabs ──
    ".stTabs [data-baseweb='tab-list'] {{"
    "  background-color: transparent !important;"
    "  border-bottom: 1.5px solid {border} !important;"
    "  gap: 0 !important;"
    "}}"
    ".stTabs [data-baseweb='tab'] {{"
    "  color: {muted} !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-size: 0.88rem !important;"
    "  font-weight: 400 !important;"
    "  background: transparent !important;"
    "  border-bottom: 2px solid transparent !important;"
    "  padding: 0.5rem 1.1rem 0.6rem !important;"
    "}}"
    ".stTabs [aria-selected='true'] {{"
    "  color: {accent} !important;"
    "  border-bottom-color: {accent} !important;"
    "  font-weight: 500 !important;"
    "}}"

    # ── Fix radio buttons in sidebar ──
    ".stRadio label {{"
    "  color: {txt} !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "  font-size: 0.9rem !important;"
    "}}"

    # ── Streamlit info/warning/error boxes ──
    ".stAlert {{"
    "  border-radius: 10px !important;"
    "  font-family: 'DM Sans', sans-serif !important;"
    "}}"

    # ── Auth page two-column layout ──
    ".auth-right-panel {{"
    "  background: #2C1F16;"
    "  border-radius: 16px;"
    "  padding: 2.5rem 1.8rem;"
    "  min-height: 440px;"
    "  display: flex;"
    "  flex-direction: column;"
    "  align-items: center;"
    "  justify-content: center;"
    "  position: relative;"
    "  overflow: hidden;"
    "}}"
    ".rp-quote {{"
    "  font-family: 'Playfair Display', serif;"
    "  font-size: 1rem;"
    "  font-style: italic;"
    "  color: rgba(245,235,220,0.88);"
    "  text-align: center;"
    "  line-height: 1.8;"
    "  position: relative; z-index: 2;"
    "}}"
    ".rp-author {{"
    "  font-size: 0.68rem;"
    "  letter-spacing: 0.14em;"
    "  text-transform: uppercase;"
    "  color: rgba(193,122,71,0.85);"
    "  margin-top: 0.8rem;"
    "  text-align: center;"
    "  position: relative; z-index: 2;"
    "}}"
    ".rp-divider {{"
    "  width: 32px; height: 1px;"
    "  background: rgba(193,122,71,0.5);"
    "  margin: 0.9rem auto;"
    "  position: relative; z-index: 2;"
    "}}"
    ".rp-feature {{"
    "  display: flex; align-items: flex-start; gap: 0.75rem;"
    "  padding: 0 0.5rem; margin-top: 0.7rem;"
    "  position: relative; z-index: 2;"
    "}}"
    ".rp-dot {{"
    "  width: 5px; height: 5px; border-radius: 50%;"
    "  background: {accent}; flex-shrink: 0; margin-top: 0.42rem;"
    "}}"
    ".rp-feat-text {{"
    "  font-size: 0.73rem;"
    "  color: rgba(245,235,220,0.62);"
    "  line-height: 1.6;"
    "}}"

    # ── Typography ──
    ".page-title {{ font-family: 'Playfair Display', serif; font-size: 1.85rem; font-weight: 700; color: {txt}; letter-spacing: -0.01em; margin-bottom: 0.1rem; }}"
    ".page-subtitle {{ font-size: 0.9rem; color: {muted}; margin-bottom: 1.8rem; font-weight: 300; letter-spacing: 0.01em; }}"
    ".section-heading {{ font-size: 0.7rem; font-weight: 600; color: {muted}; text-transform: uppercase; letter-spacing: 0.13em; margin: 1.4rem 0 0.6rem; }}"

    # ── Cards ──
    ".sd-card {{ background:{card}; border:1px solid {border}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 8px rgba(60,35,18,0.05); margin-bottom:1rem; color:{txt}; }}"
    ".stat-tile {{ background:{card}; border:1px solid {border}; border-radius:12px; padding:1rem 0.9rem; text-align:center; box-shadow:0 1px 4px rgba(60,35,18,0.04); }}"
    ".stat-value {{ font-family:'Playfair Display',serif; font-size:1.65rem; font-weight:700; color:{accent}; line-height:1.15; }}"
    ".stat-label {{ font-size:0.68rem; color:{muted}; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.2rem; font-weight:500; }}"

    # ── Welcome / empty state card ──
    ".welcome-card {{"
    "  background: linear-gradient(135deg, #EDE6DC 0%, #E5DAD0 100%);"
    "  border: 1px solid {border}; border-radius: 16px;"
    "  padding: 2rem 2rem 1.8rem; margin: 0.5rem 0 1.5rem; text-align: center;"
    "}}"
    ".welcome-title {{ font-family:'Playfair Display',serif; font-size:1.3rem; color:{txt}; font-weight:700; margin-bottom:0.4rem; }}"
    ".welcome-sub {{ font-size:0.88rem; color:{muted}; line-height:1.7; max-width:480px; margin:0 auto; }}"

    # ── Ambient level panels ──
    ".panel-high {{ background:rgba(160,48,48,0.04); border:1px solid rgba(160,48,48,0.16); border-radius:14px; padding:1.1rem; margin-bottom:0.6rem; }}"
    ".panel-low  {{ background:rgba(61,122,82,0.05); border:1px solid rgba(61,122,82,0.16); border-radius:14px; padding:1.1rem; margin-bottom:0.6rem; }}"

    # ── Level badge ──
    ".level-badge {{ display:inline-block; padding:0.35rem 1.3rem; border-radius:20px; font-size:0.78rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:white; }}"

    # ── Confidence bar ──
    ".confidence-track {{ background:{border}; border-radius:6px; height:6px; margin:0.4rem 0; }}"
    ".confidence-fill {{ height:6px; border-radius:6px; }}"

    # ── Intervention items ──
    ".iv-item {{ padding:0.65rem 0.2rem; border-bottom:1px solid {border}; line-height:1.6; color:{txt}; }}"
    ".iv-item:last-child {{ border-bottom:none; }}"
    ".iv-category {{ display:inline-block; font-size:0.65rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; background:{acl}; color:{accent}; padding:0.1rem 0.55rem; border-radius:10px; margin-left:0.5rem; }}"

    # ── Crisis notice ──
    ".crisis-notice {{ background:#FBF0EF; border:1px solid rgba(160,48,48,0.25); border-left:4px solid {high}; border-radius:10px; padding:1.2rem 1.5rem; line-height:1.75; color:{txt}; margin:0.8rem 0; }}"

    # ── Escalation banner ──
    ".escalation-banner {{ background:linear-gradient(90deg,#FDF6EC,#FAF0E0); border:1px solid #D4A84B; border-left:5px solid {accent}; border-radius:12px; padding:1.1rem 1.5rem; margin:0.8rem 0; line-height:1.75; color:{txt}; }}"

    # ── Attention heatmap ──
    ".heatmap-word {{ display:inline-block; padding:2px 5px; margin:2px; border-radius:4px; font-size:0.92rem; }}"

    # ── Breathing animation & calm zone ──
    "@keyframes breathe {{ 0% {{ transform:scale(1.0); opacity:0.5; }} 50% {{ transform:scale(1.45); opacity:1.0; }} 100% {{ transform:scale(1.0); opacity:0.5; }} }}"
    "@keyframes ripple  {{ 0% {{ transform:scale(1); opacity:0.5; }} 100% {{ transform:scale(2.4); opacity:0; }} }}"
    ".breathe-circle {{ width:80px; height:80px; border-radius:50%; background:{accent}; margin:1.4rem auto; animation:breathe 8s ease-in-out infinite; display:flex; align-items:center; justify-content:center; color:white; font-size:0.72rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; box-shadow:0 0 0 14px rgba(193,122,71,0.1); }}"
    ".ripple-ring {{ position:absolute; width:80px; height:80px; border-radius:50%; border:1.5px solid rgba(193,122,71,0.3); animation:ripple 3.2s ease-out infinite; }}"
    ".calm-zone {{ background:linear-gradient(135deg,#F5EBE0 0%,#EDE6DC 100%); border:1px solid {border}; border-radius:16px; padding:1.6rem 1.4rem 1.3rem; text-align:center; position:relative; overflow:hidden; margin:0.7rem 0; }}"

    # ── Well-being action bar ──
    ".action-bar {{ display:flex; gap:0.6rem; flex-wrap:wrap; margin:0.8rem 0 0.4rem; }}"
    ".action-btn {{ display:inline-flex; align-items:center; gap:0.35rem; padding:0.45rem 0.9rem; border-radius:20px; background:{card}; border:1px solid {border}; font-size:0.8rem; color:{txt}; text-decoration:none; font-weight:500; box-shadow:0 1px 3px rgba(60,35,18,0.05); }}"
    ".action-btn:hover {{ background:{acl}; border-color:{accent}; }}"

    # ── Streak badge ──
    ".streak-badge {{ display:inline-flex; align-items:center; gap:0.3rem; background:#FDF0E6; border:1px solid #F0D4B5; border-radius:20px; padding:0.25rem 0.75rem; font-size:0.78rem; font-weight:600; color:{accent}; }}"

    # ── Progress step ──
    ".step-card {{ background:{card}; border:1px solid {border}; border-radius:14px; padding:1.1rem 1.4rem; border-left:3px solid {accent}; margin:0.5rem 0; color:{txt}; }}"
    ".step-num {{ font-size:0.65rem; font-weight:700; color:{accent}; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.25rem; }}"

    # ── History unlock progress bar ──
    ".unlock-card {{"
    "  background:{card}; border:1px solid {border}; border-radius:12px;"
    "  padding:1rem 1.3rem; margin:1rem 0;"
    "}}"
    ".unlock-bar-track {{ background:{border}; border-radius:6px; height:6px; margin:0.5rem 0 0.3rem; }}"
    ".unlock-bar-fill {{ height:6px; border-radius:6px; background:{accent}; }}"
    ".unlock-label {{ font-size:0.75rem; color:{muted}; }}"

    "</style>"
).format(
    bg=BG_COLOR, card=CARD_BG, txt=TEXT_MAIN, muted=TEXT_MUTED,
    accent=ACCENT, acl=ACCENT_LIGHT, border=BORDER_COLOR, high=LEVEL_HIGH,
    sidebar=SIDEBAR_BG,
)

# ---------------------------------------------------------------------------
# Level metadata
# ---------------------------------------------------------------------------

_LEVEL_META: dict[str, dict] = {
    "low":       {"color": LEVEL_LOW,       "label": "Low Stress"},
    "moderate":  {"color": LEVEL_MODERATE,  "label": "Moderate Stress"},
    "high":      {"color": LEVEL_HIGH,      "label": "High Stress"},
    "uncertain": {"color": LEVEL_UNCERTAIN, "label": "Unclear"},
}

_LEVEL_MESSAGES: dict[str, str] = {
    "low":
        "You seem to be doing well right now. That is genuinely good.",
    "moderate":
        "There are some signs of tension. That is a completely normal part of life.",
    "high":
        "This sounds like a genuinely stressful moment. You are not alone.",
    "uncertain":
        "Hard to say from this alone. What you are feeling might be more "
        "nuanced than a simple label captures.",
}


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------


def _stress_level_from_score(score: float, threshold: float) -> str:
    """Compute 4-way stress level from a score and adaptive threshold."""
    if score >= threshold + 0.25:
        return "high"
    if score >= threshold + 0.10:
        return "moderate"
    if score >= threshold - 0.10:
        return "uncertain"
    return "low"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_post(endpoint: str, data: dict, token: str | None = None) -> dict:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.post(
            f"{API_URL}{endpoint}", json=data, headers=headers, timeout=30
        )
        return {"status": resp.status_code, "data": resp.json()}
    except requests.ConnectionError:
        return {
            "status": 503,
            "data": {"detail": "Cannot connect to API server. Is it running?"},
        }
    except Exception as exc:
        return {"status": 500, "data": {"detail": str(exc)}}


def _api_get(endpoint: str, token: str | None = None) -> dict:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(
            f"{API_URL}{endpoint}", headers=headers, timeout=30
        )
        return {"status": resp.status_code, "data": resp.json()}
    except requests.ConnectionError:
        return {"status": 503, "data": {"detail": "Cannot connect to API server."}}
    except Exception as exc:
        return {"status": 500, "data": {"detail": str(exc)}}


def _fetch_history(token: str) -> list[dict]:
    result = _api_get(f"/history?limit={_MAX_HISTORY_ITEMS}", token=token)
    if result["status"] != 200:
        return []
    sessions = result["data"].get("sessions", [])
    history = []
    for s in reversed(sessions):
        temporal = s.get("temporal_data", {})
        threshold = temporal.get("adaptive_threshold", 0.5)
        score = s["stress_score"]
        history.append({
            "score":      score,
            "threshold":  threshold,
            "velocity":   temporal.get("stress_velocity"),
            "level":      _stress_level_from_score(score, threshold),
            "created_at": s.get("created_at", 0),
            "triggers":   s.get("matched_triggers", []),
            "confidence": abs(score - threshold),
        })
    return history


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def _render_breathing_animation() -> None:
    # Optional ambient video background (calm_loop.mp4 in ui/assets/).
    # Uses Streamlit's native st.video() so browsers can actually load it.
    _video_path = os.path.join(os.path.dirname(__file__), "assets", "calm_loop.mp4")
    if os.path.isfile(_video_path):
        st.video(_video_path, loop=True, autoplay=True, muted=True)
    st.markdown(
        f'<div class="calm-zone">'
        f'<div style="position:relative;display:inline-block;">'
        f'<div class="ripple-ring" style="position:absolute;top:0;left:0;animation-delay:0s;"></div>'
        f'<div class="ripple-ring" style="position:absolute;top:0;left:0;animation-delay:1.2s;"></div>'
        f'<div class="breathe-circle">Breathe</div>'
        f'</div>'
        f'<p style="color:#4338ca; font-size:0.82rem; margin-top:0.3rem;">'
        f"Inhale as it expands &mdash; exhale as it contracts."
        f"</p></div>",
        unsafe_allow_html=True,
    )


def _render_attention_heatmap(text: str, weights: list[float]) -> None:
    words = text.split()
    if not weights or not words:
        return
    n = min(len(words), len(weights))
    w = weights[:n]
    max_w = max(w) if w else 1.0
    min_w = min(w) if w else 0.0
    rng = max_w - min_w if max_w != min_w else 1.0
    parts = []
    for i, word in enumerate(words[:n]):
        intensity = (w[i] - min_w) / rng
        alpha = 0.07 + intensity * 0.60
        parts.append(
            f'<span class="heatmap-word" '
            f'style="background-color:rgba(67,97,238,{alpha:.2f});">'
            f"{word}</span>"
        )
    for word in words[n:]:
        parts.append(f'<span class="heatmap-word">{word}</span>')
    st.markdown(
        '<div style="line-height:2.2; padding:0.5rem 0;">'
        + " ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_gauge(score: float, threshold: float, level: str) -> None:
    if not _PLOTLY:
        st.metric("Stress Score", f"{score:.0%}")
        return
    needle_color = _LEVEL_META.get(level, _LEVEL_META["uncertain"])["color"]
    pct = round(score * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 38, "color": needle_color}},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1,
                "tickcolor": TEXT_MUTED,
                "tickfont": {"size": 10, "color": TEXT_MUTED},
            },
            "bar": {"color": needle_color, "thickness": 0.22},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": GAUGE_GREEN},
                {"range": [40, 65],  "color": GAUGE_AMBER},
                {"range": [65, 100], "color": GAUGE_RED},
            ],
            "threshold": {
                "line": {"color": TEXT_MUTED, "width": 2},
                "thickness": 0.7,
                "value": round(threshold * 100, 1),
            },
        },
    ))
    fig.update_layout(
        height=210, margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MUTED, "size": 11},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_bar(confidence: float, level: str) -> None:
    color = _LEVEL_META.get(level, _LEVEL_META["uncertain"])["color"]
    pct = int(confidence * 100)
    st.markdown(
        f'<div style="margin:0.3rem 0 0.6rem;">'
        f'<div style="display:flex; justify-content:space-between; '
        f'font-size:0.77rem; color:{TEXT_MUTED}; margin-bottom:3px;">'
        f"<span>Prediction confidence</span><span>{pct}%</span></div>"
        f'<div class="confidence-track">'
        f'<div class="confidence-fill" style="width:{pct}%; background:{color};"></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_level_badge(level: str) -> None:
    meta = _LEVEL_META.get(level, _LEVEL_META["uncertain"])
    st.markdown(
        f'<div style="text-align:center; margin:0.4rem 0;">'
        f'<span class="level-badge" style="background:{meta["color"]};">'
        f'{meta["label"]}</span></div>',
        unsafe_allow_html=True,
    )


def _get_level_message(level: str, temporal: dict) -> str:
    base = _LEVEL_MESSAGES.get(level, _LEVEL_MESSAGES["uncertain"])
    velocity = temporal.get("stress_velocity")
    count = temporal.get("score_count", 0)
    if velocity is not None and count >= 3:
        if velocity > 0.05:
            return base + " Things seem to be building up lately."
        if velocity < -0.05:
            return base + " The trend is moving in a better direction."
    return base


def _render_crisis_notice(message: str) -> None:
    safe_msg = message or "If you are going through something serious, please reach out."
    st.markdown(
        f'<div class="crisis-notice">'
        f"<strong>We hear you.</strong><br>{safe_msg}<br><br>"
        f"<strong>108 Suicide &amp; Crisis Lifeline</strong> — call or text "
        f"<strong>108<strong><br>"
        f"<strong>Crisis Text Line</strong> — text HOME to <strong>741741</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_escalation_banner() -> None:
    st.markdown(
        '<div class="escalation-banner">'
        "<strong>🔔 Your stress has been persistently elevated.</strong><br>"
        "You've had several high-stress check-ins in a row. Speaking with a "
        "professional counsellor can make a real difference — you deserve support.<br>"
        "<strong>📞 SAMHSA:</strong> 1-800-662-4357 &nbsp;|&nbsp; "
        '<strong>🌐 Find a therapist:</strong> '
        '<a href="https://www.psychologytoday.com/us/therapists" target="_blank">'
        "psychologytoday.com</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_wellbeing_action_bar() -> None:
    st.markdown(
        '<div class="section-heading" style="margin-top:0.5rem;">Quick actions</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="action-bar">'
        '<a class="action-btn" href="https://open.spotify.com/playlist/37i9dQZF1DWXe9gFZP0gtP" target="_blank">🎵 Calming music</a>'
        '<a class="action-btn" href="https://www.youtube.com/results?search_query=5+minute+guided+meditation" target="_blank">🧘 Guided meditation</a>'
        '<a class="action-btn" href="tel:108">📞 Call 108 (crisis line)</a>'
        "</div>",
        unsafe_allow_html=True,
    )
    # Journal prompt — randomly chosen from a curated list
    _JOURNAL_PROMPTS = [
        "What is one small thing that felt manageable today?",
        "What would you say to a close friend who felt the way you feel right now?",
        "What is something — however small — that you are grateful for today?",
        "What would make tomorrow feel slightly lighter than today?",
        "What is one thing you are proud of yourself for, even recently?",
        "What does your body need most right now — rest, movement, nourishment?",
        "If your current stress was a shape, what would it look like? What would shrink it?",
    ]
    # Stable per-minute prompt — rotates once per minute when the page refreshes.
    seed = int(time.time() // 60) % len(_JOURNAL_PROMPTS)
    prompt = _JOURNAL_PROMPTS[seed]
    with st.expander("📖 Journal prompt", expanded=False):
        st.markdown(
            f'<p style="font-style:italic; font-size:0.95rem; color:{TEXT_MAIN}; '
            f'margin:0; line-height:1.65;">\u201c{prompt}\u201d</p>',
            unsafe_allow_html=True,
        )


def _render_velocity_gauge(velocity: float | None) -> None:
    if not _PLOTLY or velocity is None:
        return
    # Clamp velocity for display: ±0.15 is the practical range.
    v_pct = max(-1.0, min(1.0, velocity / 0.15)) * 100
    if v_pct > 8:
        v_color = LEVEL_HIGH
        v_label = "Rising"
    elif v_pct < -8:
        v_color = LEVEL_LOW
        v_label = "Falling"
    else:
        v_color = LEVEL_UNCERTAIN
        v_label = "Stable"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(v_pct, 1),
        number={"suffix": "", "font": {"size": 26, "color": v_color}},
        title={"text": f"<b>{v_label}</b>", "font": {"size": 11, "color": TEXT_MUTED}},
        gauge={
            "axis": {
                "range": [-100, 100], "tickwidth": 1,
                "tickvals": [-100, -50, 0, 50, 100],
                "ticktext": ["↓↓", "↓", "—", "↑", "↑↑"],
                "tickfont": {"size": 9, "color": TEXT_MUTED},
            },
            "bar": {"color": v_color, "thickness": 0.22},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [-100, -20], "color": "#DCFCE7"},
                {"range": [-20, 20],   "color": "#F1F5F9"},
                {"range": [20, 100],   "color": "#FEE2E2"},
            ],
        },
    ))
    fig.update_layout(
        height=175, margin=dict(l=15, r=15, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MUTED, "size": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Stress velocity — rate of change across recent check-ins.")


_URL_RE = re.compile(r"https?://[^\s)]+")
_BARE_URL_RE = re.compile(r"\b([a-z][a-z0-9\-]+\.[a-z]{2,}(?:/[^\s)]*)?)")


def _extract_links(text: str) -> list[str]:
    """Return a de-duplicated list of URLs (http/https or bare domain) found in text."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _URL_RE.finditer(text):
        url = m.group().rstrip(".,;)")
        if url not in seen:
            found.append(url)
            seen.add(url)
    return found


def _render_intervention_item(iv: dict) -> None:
    """Render a single intervention. Resource-category ones get clickable link badges."""
    category = iv.get("category", "")
    title = iv["title"]
    description = iv["description"]
    links = _extract_links(description) if category == "resource" else []

    if links:
        # Build link badges for resource cards
        badges = "".join(
            f'<a href="{u}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-flex;align-items:center;gap:0.25rem;'
            f'margin:0.2rem 0.2rem 0 0;padding:0.25rem 0.7rem;border-radius:16px;'
            f'background:{ACCENT_LIGHT};border:1px solid {BORDER_COLOR};'
            f'font-size:0.76rem;color:{ACCENT};text-decoration:none;font-weight:500;">'
            f'🔗 {u[:45] + "…" if len(u) > 45 else u}</a>'
            for u in links
        )
        st.markdown(
            f'<div class="iv-item">'
            f'<strong>{title}</strong>'
            f'<span class="iv-category">{category}</span><br>'
            f'<span style="color:{TEXT_MUTED};font-size:0.88rem;line-height:1.6;">'
            f'{description}</span><br>'
            f'<div style="margin-top:0.35rem;">{badges}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="iv-item">'
            f'<strong>{title}</strong>'
            f'<span class="iv-category">{category}</span><br>'
            f'<span style="color:{TEXT_MUTED};font-size:0.88rem;">'
            f'{description}</span></div>',
            unsafe_allow_html=True,
        )


def _render_breathing_timer(seconds: int = 60) -> None:
    """Embed a countdown timer via an HTML component."""
    html = f"""
    <div id="timer-wrap" style="text-align:center;padding:0.6rem 0;">
      <div id="timer" style="font-size:2.6rem;font-weight:700;
           color:#4361EE;font-variant-numeric:tabular-nums;">
        {seconds:02d}
      </div>
      <div style="font-size:0.75rem;color:#718096;margin-top:0.2rem;">seconds remaining</div>
      <button onclick="startTimer()" style="margin-top:0.5rem;padding:0.35rem 1.1rem;
        border-radius:20px;background:#4361EE;color:#fff;border:none;
        font-size:0.82rem;font-weight:600;cursor:pointer;">Start</button>
    </div>
    <script>
    var _t, _s = {seconds};
    function startTimer() {{
      clearInterval(_t);
      _s = {seconds};
      _t = setInterval(function() {{
        _s--;
        document.getElementById('timer').textContent =
          String(Math.floor(_s/60)).padStart(2,'0') + ':' +
          String(_s % 60).padStart(2,'0');
        if (_s <= 0) {{ clearInterval(_t); document.getElementById('timer').textContent = '✓ Done'; }}
      }}, 1000);
    }}
    </script>
    """
    components.html(html, height=140)


def _render_progressive_interventions(interventions: list[dict]) -> None:
    """Render a step-by-step guided intervention flow."""
    step_key = "iv_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0

    current = st.session_state[step_key]
    total = len(interventions)
    if total == 0:
        return

    # Progress indicator
    st.markdown(
        f'<p style="font-size:0.75rem;color:{TEXT_MUTED};margin:0.2rem 0 0.6rem;">'
        f"Step {min(current + 1, total)} of {total}</p>",
        unsafe_allow_html=True,
    )

    if current < total:
        iv = interventions[current]
        is_breathing = iv.get("category") == "breathing"
        links = _extract_links(iv["description"]) if iv.get("category") == "resource" else []
        badge_html = "".join(
            f'<a href="{u}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-flex;align-items:center;gap:0.25rem;'
            f'margin:0.25rem 0.2rem 0 0;padding:0.25rem 0.7rem;border-radius:16px;'
            f'background:{ACCENT_LIGHT};border:1px solid {BORDER_COLOR};'
            f'font-size:0.76rem;color:{ACCENT};text-decoration:none;font-weight:500;">'
            f'🔗 {u[:45] + "…" if len(u) > 45 else u}</a>'
            for u in links
        )
        st.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">Step {current + 1} &mdash; {iv["category"].capitalize()}</div>'
            f'<strong style="font-size:1rem;">{iv["title"]}</strong>'
            f'<span class="iv-category">{iv["category"]}</span><br>'
            f'<p style="color:{TEXT_MUTED};font-size:0.88rem;margin:0.5rem 0 0;line-height:1.6;">'
            f'{iv["description"]}</p>'
            + (f'<div style="margin-top:0.35rem;">{badge_html}</div>' if badge_html else "")
            + f'</div>',
            unsafe_allow_html=True,
        )
        if is_breathing:
            _render_breathing_animation()
            _render_breathing_timer(60)

        col_next, col_skip, _ = st.columns([1.2, 1, 4])
        with col_next:
            label = "✓ Done → Next" if current < total - 1 else "✓ All done"
            if st.button(label, key=f"iv_next_{current}", type="primary"):
                st.session_state[step_key] = current + 1
                st.rerun()
        with col_skip:
            if current < total - 1:
                if st.button("Skip", key=f"iv_skip_{current}"):
                    st.session_state[step_key] = current + 1
                    st.rerun()
    else:
        st.success(
            "You've completed all the suggested steps. Take a moment to notice "
            "how you feel now. 🌱"
        )
        if st.button("Start again", key="iv_restart"):
            st.session_state[step_key] = 0
            st.rerun()


def _render_timeline_chart(history: list[dict]) -> None:
    if not _PLOTLY:
        st.info("Install plotly for charts: pip install plotly")
        return
    if not history:
        st.caption("Your check-in history will appear here.")
        return
    indices    = list(range(1, len(history) + 1))
    scores     = [h["score"] for h in history]
    thresholds = [h.get("threshold", 0.5) for h in history]

    # 7-session rolling wellness score (inverted, smoothed)
    _WIN = 7
    wellness = []
    for i in range(len(scores)):
        window = scores[max(0, i - _WIN + 1): i + 1]
        wellness.append(1.0 - (sum(window) / len(window)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=scores,
        mode="lines+markers", name="Stress score",
        line=dict(color=ACCENT, width=2.5, shape="spline", smoothing=0.7),
        marker=dict(size=7, color=ACCENT, line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(67,97,238,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=thresholds,
        mode="lines", name="Your threshold",
        line=dict(color=LEVEL_UNCERTAIN, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=wellness,
        mode="lines", name="🌱 Wellness trend",
        line=dict(color=LEVEL_LOW, width=2, dash="dash"),
        opacity=0.75,
    ))
    # Highlight sessions where wellness is rising
    rising_idx = [i + 1 for i in range(1, len(wellness)) if wellness[i] > wellness[i - 1] + 0.05]
    if rising_idx:
        rising_vals = [wellness[i - 1] for i in rising_idx]
        fig.add_trace(go.Scatter(
            x=rising_idx, y=rising_vals,
            mode="markers", name="Wellness improving 🌱",
            marker=dict(size=11, color=LEVEL_LOW, symbol="star",
                        line=dict(color="white", width=1)),
        ))
    fig.update_layout(
        title=dict(text="Check-in timeline", font=dict(size=13, color=TEXT_MUTED)),
        xaxis=dict(title="Check-in #", showgrid=False, color=TEXT_MUTED),
        yaxis=dict(
            title="", range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            showgrid=True, gridcolor=BORDER_COLOR, color=TEXT_MUTED,
        ),
        plot_bgcolor=CARD_BG, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320, margin=dict(l=40, r=10, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution_chart(history: list[dict]) -> None:
    if not _PLOTLY or not history:
        return
    counts = {"low": 0, "moderate": 0, "high": 0, "uncertain": 0}
    for h in history:
        score     = h.get("score", 0.5)
        threshold = h.get("threshold", 0.5)
        if score >= threshold + 0.25:
            counts["high"] += 1
        elif score >= threshold + 0.10:
            counts["moderate"] += 1
        elif score >= threshold - 0.10:
            counts["uncertain"] += 1
        else:
            counts["low"] += 1
    labels = [_LEVEL_META[k]["label"] for k in counts]
    values = list(counts.values())
    colors = [_LEVEL_META[k]["color"] for k in counts]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent+label", textfont=dict(size=11),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Stress level distribution", font=dict(size=13, color=TEXT_MUTED)),
        height=280, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_calendar_heatmap(history: list[dict]) -> None:
    """GitHub-style stress calendar heatmap."""
    if not _PLOTLY or not history:
        st.caption("Not enough data for a calendar view yet.")
        return
    # Group by date
    day_scores: dict[str, list[float]] = {}
    for h in history:
        ts = h.get("created_at", 0)
        if ts == 0:
            continue
        date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        day_scores.setdefault(date_str, []).append(h["score"])
    if not day_scores:
        st.caption("Timestamps not available for calendar view.")
        return

    # Build a grid: weeks × 7 days
    dates_sorted = sorted(day_scores.keys())
    first_date = datetime.datetime.strptime(dates_sorted[0], "%Y-%m-%d")
    last_date  = datetime.datetime.strptime(dates_sorted[-1], "%Y-%m-%d")
    # Extend to full weeks
    start = first_date - datetime.timedelta(days=first_date.weekday())
    end   = last_date + datetime.timedelta(days=(6 - last_date.weekday()))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    weeks: list[list[str]] = []
    week: list[str] = []
    cur = start
    while cur <= end:
        week.append(cur.strftime("%Y-%m-%d"))
        if len(week) == 7:
            weeks.append(week)
            week = []
        cur += datetime.timedelta(days=1)
    if week:
        weeks.append(week)

    z = []
    text = []
    for dow in range(7):
        row_z = []
        row_t = []
        for wk in weeks:
            d = wk[dow]
            if d in day_scores:
                avg = sum(day_scores[d]) / len(day_scores[d])
                row_z.append(avg)
                row_t.append(f"{d}<br>Avg: {avg:.0%}")
            else:
                row_z.append(None)
                row_t.append(d)
        z.append(row_z)
        text.append(row_t)

    week_labels = [w[0] for w in weeks]
    fig = go.Figure(go.Heatmap(
        z=z, x=week_labels, y=day_names,
        text=text, hoverinfo="text",
        colorscale=[[0, "#bbf7d0"], [0.4, "#fef08a"], [0.7, "#fdba74"], [1, "#ef4444"]],
        zmin=0, zmax=1,
        showscale=True, xgap=3, ygap=3,
        colorbar=dict(title="Stress", tickformat=".0%", len=0.6),
    ))
    fig.update_layout(
        title=dict(text="Stress calendar", font=dict(size=13, color=TEXT_MUTED)),
        height=260, margin=dict(l=40, r=60, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color=TEXT_MUTED, tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, color=TEXT_MUTED),
        font=dict(color=TEXT_MUTED, size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_time_of_day_chart(history: list[dict]) -> None:
    """Polar bar chart showing average stress by time slot."""
    if not _PLOTLY or not history:
        st.caption("Not enough data for a time-of-day chart yet.")
        return
    slots = {"Night (0–6h)": [], "Morning (6–12h)": [],
             "Afternoon (12–18h)": [], "Evening (18–24h)": []}
    for h in history:
        ts = h.get("created_at", 0)
        if ts == 0:
            continue
        hr = datetime.datetime.fromtimestamp(ts).hour
        if hr < 6:
            slots["Night (0–6h)"].append(h["score"])
        elif hr < 12:
            slots["Morning (6–12h)"].append(h["score"])
        elif hr < 18:
            slots["Afternoon (12–18h)"].append(h["score"])
        else:
            slots["Evening (18–24h)"].append(h["score"])
    slot_names = list(slots.keys())
    avgs = [sum(v) / len(v) if v else 0.0 for v in slots.values()]
    colors_by_slot = [LEVEL_UNCERTAIN, LEVEL_LOW, LEVEL_MODERATE, LEVEL_HIGH]
    fig = go.Figure(go.Barpolar(
        r=[v * 100 for v in avgs],
        theta=slot_names,
        marker_color=colors_by_slot,
        marker_line_color="white",
        marker_line_width=1.5,
        opacity=0.85,
    ))
    fig.update_layout(
        title=dict(text="Stress by time of day", font=dict(size=13, color=TEXT_MUTED)),
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75], ticktext=["25%", "50%", "75%"],
                color=TEXT_MUTED, gridcolor=BORDER_COLOR,
            ),
            angularaxis=dict(color=TEXT_MUTED),
        ),
        height=320, margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trigger_frequency_chart(history: list[dict]) -> None:
    """Horizontal bar chart of trigger category frequencies."""
    if not _PLOTLY or not history:
        st.caption("No trigger data available yet.")
        return
    counts: dict[str, int] = {}
    for h in history:
        for t in h.get("triggers", []):
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        st.caption("No specific triggers detected across your check-ins yet.")
        return
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names  = [item[0].capitalize() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    fig = go.Figure(go.Bar(
        y=names, x=values,
        orientation="h",
        marker_color=ACCENT, opacity=0.82,
        text=values, textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Trigger frequency", font=dict(size=13, color=TEXT_MUTED)),
        xaxis=dict(title="Times detected", color=TEXT_MUTED, showgrid=True, gridcolor=BORDER_COLOR),
        yaxis=dict(autorange="reversed", color=TEXT_MUTED),
        height=max(220, 60 + 36 * len(names)),
        margin=dict(l=10, r=50, t=50, b=40),
        plot_bgcolor=CARD_BG, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_histogram(history: list[dict]) -> None:
    """Histogram of historical model confidence values."""
    if not _PLOTLY or not history:
        return
    conf_values = [h.get("confidence", 0.0) for h in history if "confidence" in h]
    if len(conf_values) < 3:
        st.caption("Need at least 3 check-ins for a confidence distribution.")
        return
    fig = go.Figure(go.Histogram(
        x=conf_values, nbinsx=10,
        marker_color=ACCENT, opacity=0.75,
        marker_line_color="white", marker_line_width=1,
    ))
    avg_conf = sum(conf_values) / len(conf_values)
    fig.add_vline(x=avg_conf, line_dash="dot", line_color=LEVEL_MODERATE,
                  annotation_text=f"Avg {avg_conf:.2f}", annotation_position="top right")
    fig.update_layout(
        title=dict(text="Prediction confidence distribution", font=dict(size=13, color=TEXT_MUTED)),
        xaxis=dict(title="Confidence (distance from threshold)", color=TEXT_MUTED,
                   showgrid=True, gridcolor=BORDER_COLOR),
        yaxis=dict(title="Count", color=TEXT_MUTED),
        height=260, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor=CARD_BG, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=11),
    )
    st.plotly_chart(fig, use_container_width=True)
    if avg_conf < 0.15:
        st.caption(
            "⚠️ Many predictions are close to the boundary — the model is often "
            "uncertain for your inputs. More personalised feedback helps it improve."
        )


def _history_to_csv(history: list[dict]) -> bytes:
    try:
        import csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["check_in", "score_pct", "threshold_pct", "level"])
        for i, h in enumerate(history, 1):
            writer.writerow([
                i,
                f"{h['score']:.1%}",
                f"{h.get('threshold', 0.5):.1%}",
                h.get("level", ""),
            ])
        return buf.getvalue().encode("utf-8")
    except Exception:
        return b""


def _submit_feedback(text: str, score: float, user_feedback: int, token: str) -> None:
    result = _api_post(
        "/feedback",
        {"text": text, "prediction": score, "user_feedback": user_feedback},
        token=token,
    )
    if result["status"] == 200:
        msg = (
            "Noted. Glad this felt accurate."
            if user_feedback == 1
            else "Thanks for letting us know. This helps the model improve."
        )
        llm_r = result["data"].get("llm_reward")
        if llm_r is not None:
            msg += f"  (AI reviewer: {'agreed' if llm_r == 1 else 'disagreed'})"
        st.session_state["_fb_message"] = msg
        st.session_state["_fb_status"] = "success"
    else:
        st.session_state["_fb_message"] = "Could not save feedback right now."
        st.session_state["_fb_status"] = "warning"


# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------


def _compute_stats(history: list[dict]) -> dict:
    if not history:
        return {"total": 0, "avg": 0.0, "trend": "\u2014", "trend_delta": 0.0}
    scores = [h["score"] for h in history]
    avg = sum(scores) / len(scores)
    if len(scores) >= 6:
        recent = sum(scores[-3:]) / 3
        older  = sum(scores[-6:-3]) / 3
        delta  = recent - older
        trend  = "\u2191" if delta > 0.03 else ("\u2193" if delta < -0.03 else "\u2192")
    else:
        delta = 0.0
        trend = "\u2014"
    return {"total": len(scores), "avg": avg, "trend": trend, "trend_delta": delta}


# ---------------------------------------------------------------------------
# Page: Auth
# ---------------------------------------------------------------------------


def _auth_page() -> None:
    col_form, col_visual = st.columns([1.1, 0.9])

    # ── Right panel: visual / brand ──
    with col_visual:
        st.markdown(
            '<div class="auth-right-panel">'
            # Animated SVG background: breathing rings + floating words
            '<svg style="position:absolute;inset:0;width:100%;height:100%;" '
            'viewBox="0 0 400 520" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">'
            '<circle cx="320" cy="80" r="140" fill="rgba(193,122,71,0.07)"/>'
            '<circle cx="60" cy="430" r="110" fill="rgba(193,122,71,0.05)"/>'
            '<circle cx="200" cy="260" r="200" fill="rgba(255,255,255,0.015)"/>'
            # grid lines
            '<line x1="0" y1="130" x2="400" y2="130" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            '<line x1="0" y1="260" x2="400" y2="260" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            '<line x1="0" y1="390" x2="400" y2="390" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            '<line x1="100" y1="0" x2="100" y2="520" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            '<line x1="200" y1="0" x2="200" y2="520" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            '<line x1="300" y1="0" x2="300" y2="520" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
            # breathing rings (animated)
            '<circle cx="200" cy="195" r="56" fill="none" stroke="rgba(193,122,71,0.18)" stroke-width="1">'
            '<animate attributeName="r" values="56;70;56" dur="8s" repeatCount="indefinite"/>'
            '<animate attributeName="opacity" values="0.5;1;0.5" dur="8s" repeatCount="indefinite"/>'
            '</circle>'
            '<circle cx="200" cy="195" r="40" fill="none" stroke="rgba(193,122,71,0.28)" stroke-width="0.5">'
            '<animate attributeName="r" values="40;54;40" dur="8s" repeatCount="indefinite"/>'
            '</circle>'
            '<circle cx="200" cy="195" r="24" fill="rgba(193,122,71,0.14)">'
            '<animate attributeName="r" values="24;32;24" dur="8s" repeatCount="indefinite"/>'
            '</circle>'
            # floating words
            '<text x="52" y="94" font-family="Georgia,serif" font-size="11" fill="rgba(245,235,220,0.22)" font-style="italic">calm</text>'
            '<text x="288" y="148" font-family="Georgia,serif" font-size="10" fill="rgba(245,235,220,0.17)" font-style="italic">present</text>'
            '<text x="28" y="310" font-family="Georgia,serif" font-size="10" fill="rgba(245,235,220,0.19)" font-style="italic">breathe</text>'
            '<text x="300" y="370" font-family="Georgia,serif" font-size="11" fill="rgba(245,235,220,0.2)" font-style="italic">aware</text>'
            '<text x="80" y="460" font-family="Georgia,serif" font-size="10" fill="rgba(245,235,220,0.14)" font-style="italic">grounded</text>'
            '<text x="270" y="490" font-family="Georgia,serif" font-size="10" fill="rgba(245,235,220,0.15)" font-style="italic">still</text>'
            '<text x="130" y="58" font-family="Georgia,serif" font-size="10" fill="rgba(245,235,220,0.13)" font-style="italic">reflect</text>'
            '<text x="320" y="240" font-family="Georgia,serif" font-size="9" fill="rgba(245,235,220,0.12)" font-style="italic">ease</text>'
            '</svg>'
            # Brand name
            '<div style="font-family:\'Playfair Display\',Georgia,serif;font-size:1.6rem;'
            'font-weight:700;color:rgba(245,235,220,0.95);text-align:center;'
            'position:relative;z-index:2;letter-spacing:-0.01em;margin-bottom:0.1rem;">'
            'StressDetect</div>'
            '<div style="font-size:0.7rem;letter-spacing:0.16em;text-transform:uppercase;'
            'color:rgba(193,122,71,0.75);text-align:center;position:relative;z-index:2;'
            'margin-bottom:1.4rem;">A quiet check-in</div>'
            # Quote
            '<div class="rp-quote">'
            '\u201cYou cannot always control what goes on outside,<br>'
            'but you can always control what goes on inside.\u201d'
            '</div>'
            '<div class="rp-divider"></div>'
            '<div class="rp-author">Wayne Dyer</div>'
            # Feature list
            '<div style="margin-top:1.6rem;width:100%;position:relative;z-index:2;">'
            '<div class="rp-feature"><div class="rp-dot"></div>'
            '<div class="rp-feat-text">Adaptive threshold — stress detection personalised to you</div></div>'
            '<div class="rp-feature"><div class="rp-dot"></div>'
            '<div class="rp-feat-text">Science-backed interventions, not generic advice</div></div>'
            '<div class="rp-feature"><div class="rp-dot"></div>'
            '<div class="rp-feat-text">History encrypted and always preserved on sign-in</div></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Left panel: forms ──
    with col_form:
        st.markdown(
            '<div style="padding: 1rem 0.5rem;">'
            '<div style="font-family:\'Playfair Display\',Georgia,serif;font-size:1.55rem;'
            f'font-weight:700;color:{TEXT_MAIN};margin-bottom:0.15rem;">Welcome back</div>'
            f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin-bottom:1.8rem;'
            'font-weight:300;letter-spacing:0.01em;">Sign in to continue your journey.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Sign in \u2192", type="primary", use_container_width=True
                )
                if submitted and username and password:
                    with st.spinner("Signing in\u2026"):
                        result = _api_post(
                            "/login", {"username": username, "password": password}
                        )
                    if result["status"] == 200:
                        token = result["data"]["access_token"]
                        st.session_state.token    = token
                        st.session_state.username = username
                        st.session_state.history  = _fetch_history(token)
                        st.session_state.page     = "Dashboard"
                        st.rerun()
                    else:
                        st.error(result["data"].get("detail", "Could not sign in."))
            st.markdown(
                f'<p style="font-size:0.75rem;color:{TEXT_MUTED};margin-top:0.6rem;'
                'line-height:1.6;">Your session stays active for 7 days. '
                'History is always preserved.</p>',
                unsafe_allow_html=True,
            )

        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("Username", placeholder="3\u201350 characters")
                new_pass = st.text_input("Password", type="password", placeholder="8+ characters")
                submitted = st.form_submit_button(
                    "Create account \u2192", type="primary", use_container_width=True
                )
                if submitted and new_user and new_pass:
                    with st.spinner("Creating account\u2026"):
                        result = _api_post(
                            "/register", {"username": new_user, "password": new_pass}
                        )
                    if result["status"] == 201:
                        token = result["data"]["access_token"]
                        st.session_state.token    = token
                        st.session_state.username = new_user
                        st.session_state.history  = []
                        st.session_state.page     = "Dashboard"
                        st.rerun()
                    else:
                        st.error(result["data"].get("detail", "Could not create account."))
            st.markdown(
                f'<p style="font-size:0.75rem;color:{TEXT_MUTED};margin-top:0.6rem;'
                'line-height:1.6;">Your check-in history is encrypted and '
                'tied to your account.</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

# Minimum meaningful word count (after stop-word removal) for UI validation.
_STOP_WORDS = frozenset({
    "i", "me", "my", "we", "you", "he", "she", "it", "they",
    "am", "is", "are", "was", "were", "a", "an", "the", "and",
    "or", "of", "to", "in", "on", "at", "by", "for", "with",
})

# Repetition guard: hash → last submission timestamp (in-memory, per session)
_REPEAT_WINDOW_SECS: int = 60
_FOLLOW_UP_INTERVAL_SECS: int = 1800  # 30 minutes
_MIN_MEANINGFUL_WORDS: int = 4  # minimum words (excluding stop words) to accept


def _meaningful_word_count(text: str) -> int:
    # Use word-boundary extraction so "hello," and "hello" both match.
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w not in _STOP_WORDS)


def _dashboard_page() -> None:
    history = st.session_state.get("history", [])
    stats   = _compute_stats(history)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value">{stats["total"]}</div>'
            f'<div class="stat-label">Total check-ins</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        avg_str = f'{stats["avg"]:.0%}' if stats["total"] else "\u2014"
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value">{avg_str}</div>'
            f'<div class="stat-label">Average score</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        trend_color = (
            LEVEL_HIGH if stats["trend"] == "\u2191"
            else LEVEL_LOW if stats["trend"] == "\u2193"
            else TEXT_MUTED
        )
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value" style="color:{trend_color};">{stats["trend"]}</div>'
            f'<div class="stat-label">Recent trend</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        fb_stats = _api_get("/feedback/stats", token=st.session_state.get("token"))
        if fb_stats["status"] == 200 and fb_stats["data"].get("total", 0) > 0:
            acc_str = f'{fb_stats["data"].get("accuracy_rate", 0):.0%}'
        else:
            acc_str = "\u2014"
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value">{acc_str}</div>'
            f'<div class="stat-label">Model accuracy for you</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Welcome card (first visit / empty state) ──
    if not history:
        _AFFIRMATIONS = [
            "You are allowed to take up space and take care of yourself.",
            "Even a small check-in is an act of self-awareness.",
            "Rest is productive. Noticing is the first step.",
            "You don't have to have it all figured out right now.",
            "Being honest with yourself takes courage.",
        ]
        seed = int(time.time() // 300) % len(_AFFIRMATIONS)
        st.markdown(
            f'<div class="welcome-card">'
            f'<div class="welcome-title">Good to have you here</div>'
            f'<p class="welcome-sub">'
            f'Write a few sentences about how you\'re feeling — work, sleep, '
            f'relationships, anything on your mind. The model will analyse your '
            f'stress level and suggest what might help.'
            f'</p>'
            f'<p style="font-family:Georgia,serif;font-style:italic;'
            f'font-size:0.85rem;color:{TEXT_MUTED};margin-top:0.9rem;line-height:1.7;">'
            f'\u201c{_AFFIRMATIONS[seed]}\u201d</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Follow-up prompt: if a follow-up timer has elapsed, show reminder
    if "follow_up_time" in st.session_state:
        if time.time() >= st.session_state["follow_up_time"]:
            st.info(
                "⏰ **30-minute check-in:** How are you feeling now compared to "
                "earlier? Take a moment to write it below."
            )
            if st.button("Dismiss reminder", key="dismiss_followup"):
                del st.session_state["follow_up_time"]
                st.rerun()

    # Input area
    st.markdown('<div class="section-heading">How are you feeling?</div>', unsafe_allow_html=True)
    text_input = st.text_area(
        "", height=130,
        placeholder="Take a moment to write what's on your mind\u2026",
        label_visibility="collapsed",
    )

    if st.button("Check In \u2192", type="primary"):
        stripped = text_input.strip()
        if not stripped:
            st.warning("Write something first \u2014 even a few words.")
            st.stop()

        # Minimum meaningful word count guard
        if _meaningful_word_count(stripped) < _MIN_MEANINGFUL_WORDS:
            st.warning(
                "Write a little more for a useful reading — "
                "at least 4 meaningful words help the model understand you."
            )
            st.stop()

        # Repetition guard
        # sha256 used for input deduplication fingerprinting (collision resistance,
        # not cryptographic security — no secret data is stored or compared).
        _input_hash = hashlib.sha256(stripped.encode()).hexdigest()
        _last_time  = st.session_state.get("_last_input_hash_time", {})
        _now = time.time()
        if _last_time.get(_input_hash, 0) > _now - _REPEAT_WINDOW_SECS:
            st.info(
                "You submitted the same text very recently. "
                "Wait a moment or modify your entry for a fresh analysis."
            )
            st.stop()
        _last_time[_input_hash] = _now
        st.session_state["_last_input_hash_time"] = _last_time

        with st.spinner("Analysing\u2026"):
            result = _api_post(
                "/analyze", {"text": text_input},
                token=st.session_state.token,
            )

        if result["status"] == 401:
            st.error("Your session expired. Please sign in again.")
            st.session_state.pop("token", None)
            st.rerun()
            st.stop()

        if result["status"] != 200:
            st.error(result["data"].get("detail", "Something went wrong. Please try again."))
            st.stop()

        data     = result["data"]
        score    = data["stress_score"]
        level    = data.get("stress_level", "uncertain")
        temporal = data.get("temporal", {})

        # Reset progressive intervention step when new analysis arrives
        st.session_state["iv_step"] = 0

        st.session_state["current_analysis"] = {
            "text":                text_input,
            "score":               score,
            "level":               level,
            "confidence":          data.get("confidence", 0.0),
            "temporal":            temporal,
            "interventions":       data.get("interventions", []),
            "is_crisis":           data.get("is_crisis", False),
            "crisis_message":      data.get("crisis_message", ""),
            "attention_weights":   data.get("attention_weights", []),
            "requires_escalation": data.get("requires_escalation", False),
            "is_uncertain":        data.get("is_uncertain", False),
        }
        st.session_state["feedback_done"] = False
        st.session_state.pop("_fb_message", None)

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "score":      score,
            "threshold":  temporal.get("adaptive_threshold", 0.5),
            "velocity":   temporal.get("stress_velocity"),
            "level":      level,
            "created_at": time.time(),
            "triggers":   data.get("matched_triggers", []),
            "confidence": data.get("confidence", 0.0),
        })

        # Update streak counter
        streak = st.session_state.get("low_streak", 0)
        if level == "low":
            st.session_state["low_streak"] = streak + 1
        elif level == "high":
            st.session_state["low_streak"] = 0

    # ── Result display ──
    ca = st.session_state.get("current_analysis")
    if ca:
        st.markdown("---")

        if ca.get("is_crisis"):
            _render_crisis_notice(ca.get("crisis_message", ""))
        else:
            level     = ca["level"]
            temporal  = ca["temporal"]
            threshold = temporal.get("adaptive_threshold", 0.5)
            velocity  = temporal.get("stress_velocity")

            # Escalation banner (persistent, above everything else)
            if ca.get("requires_escalation"):
                _render_escalation_banner()

            # Ambient panel tint based on stress level
            panel_class = "panel-high" if level == "high" else ("panel-low" if level == "low" else "")
            if panel_class:
                st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)

            # Gauges row: stress gauge + velocity gauge
            if _PLOTLY and velocity is not None:
                gc1, gc2 = st.columns([1.4, 1])
                with gc1:
                    _render_gauge(ca["score"], threshold, level)
                with gc2:
                    _render_velocity_gauge(velocity)
            else:
                _render_gauge(ca["score"], threshold, level)

            _render_level_badge(level)
            _render_confidence_bar(ca.get("confidence", 0.0), level)

            if ca.get("is_uncertain"):
                st.caption(
                    "⚠️ This prediction is near the decision boundary — "
                    "treat it as a rough indicator rather than a definitive result."
                )

            if panel_class:
                st.markdown("</div>", unsafe_allow_html=True)

            message = _get_level_message(level, temporal)
            st.markdown(
                f'<p style="font-size:1rem; line-height:1.65; '
                f'color:{TEXT_MAIN}; margin:0.8rem 0;">{message}</p>',
                unsafe_allow_html=True,
            )

            interventions = ca.get("interventions", [])

            # Well-being action bar for high stress
            if level == "high":
                _render_wellbeing_action_bar()

            # Follow-up reminder button for high/moderate
            if level in ("high", "moderate") and "follow_up_time" not in st.session_state:
                if st.button("⏰ Remind me to check back in 30 min", key="set_followup"):
                    st.session_state["follow_up_time"] = time.time() + _FOLLOW_UP_INTERVAL_SECS
                    st.success("Got it — we'll prompt you in 30 minutes.")

            # Progressive intervention flow for high stress; static list otherwise
            if interventions:
                st.markdown(
                    '<div class="section-heading" style="margin-top:0.75rem;">'
                    "What might help right now</div>",
                    unsafe_allow_html=True,
                )
                if level == "high" and len(interventions) > 1:
                    _render_progressive_interventions(interventions)
                else:
                    # Static list for low/moderate — still show breathing section
                    has_breathing = any(
                        iv.get("category") == "breathing" for iv in interventions
                    )
                    for iv in interventions:
                        _render_intervention_item(iv)
                    if has_breathing:
                        _render_breathing_animation()

            # Attention heatmap
            attn = ca.get("attention_weights", [])
            if attn:
                with st.expander("View word attention heatmap", expanded=False):
                    st.caption(
                        "Highlighted words contributed most to the prediction. "
                        "Darker = higher attention weight."
                    )
                    _render_attention_heatmap(ca["text"], attn)

            # Feedback
            st.markdown("")
            if not st.session_state.get("feedback_done"):
                st.markdown(
                    f'<p style="color:{TEXT_MUTED}; font-size:0.86rem; margin:0;">'
                    "Did this feel right?</p>",
                    unsafe_allow_html=True,
                )
                col_yes, col_no, _ = st.columns([1, 1, 4])
                with col_yes:
                    if st.button("Yes", key="fb_yes"):
                        _submit_feedback(
                            ca["text"], ca["score"], 1,
                            st.session_state.token,
                        )
                        st.session_state["feedback_done"] = True
                        st.rerun()
                with col_no:
                    if st.button("Not quite", key="fb_no"):
                        _submit_feedback(
                            ca["text"], ca["score"], 0,
                            st.session_state.token,
                        )
                        st.session_state["feedback_done"] = True
                        st.rerun()
            else:
                fb_msg    = st.session_state.get("_fb_message", "")
                fb_status = st.session_state.get("_fb_status", "success")
                if fb_msg:
                    if fb_status == "success":
                        st.success(fb_msg)
                    else:
                        st.warning(fb_msg)

    # Mini timeline
    if history:
        st.markdown("---")
        _render_timeline_chart(history[-20:])


# ---------------------------------------------------------------------------
# Page: History & Analytics
# ---------------------------------------------------------------------------


def _history_page() -> None:
    history = st.session_state.get("history", [])
    if not history:
        st.markdown(
            f'<div class="welcome-card" style="text-align:left;">'
            f'<div class="welcome-title">No check-ins yet</div>'
            f'<p class="welcome-sub" style="text-align:left;">'
            f'Head to the Dashboard and make your first check-in. '
            f'Your stress timeline, patterns, and calendar will appear here '
            f'as you build a history.'
            f'</p></div>',
            unsafe_allow_html=True,
        )
        return

    stats  = _compute_stats(history)
    scores = [h["score"] for h in history]
    n      = len(history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-tile"><div class="stat-value">{stats["total"]}</div>'
            '<div class="stat-label">Total sessions</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-tile"><div class="stat-value">{stats["avg"]:.0%}</div>'
            '<div class="stat-label">Average score</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value" style="color:{LEVEL_LOW};">{min(scores):.0%}</div>'
            '<div class="stat-label">Best session</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-value" style="color:{LEVEL_HIGH};">{max(scores):.0%}</div>'
            '<div class="stat-label">Hardest session</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Unlock progress nudges (shown until thresholds met) ──
    _UNLOCKS = [
        (3,  "Calendar view",     "3 check-ins to unlock the stress calendar"),
        (7,  "Pattern analysis",  "7 check-ins to unlock time-of-day patterns"),
        (10, "Confidence chart",  "10 check-ins to unlock prediction confidence"),
    ]
    unlock_html = ""
    for threshold, label, hint in _UNLOCKS:
        if n < threshold:
            pct = int((n / threshold) * 100)
            unlock_html += (
                f'<div class="unlock-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:0.82rem;font-weight:500;color:{TEXT_MAIN};">{label}</span>'
                f'<span class="unlock-label">{n}/{threshold} check-ins</span>'
                f'</div>'
                f'<div class="unlock-bar-track">'
                f'<div class="unlock-bar-fill" style="width:{pct}%;"></div>'
                f'</div>'
                f'<div class="unlock-label">{hint}</div>'
                f'</div>'
            )
    if unlock_html:
        left_u, right_u = st.columns([1, 1])
        with left_u:
            st.markdown(unlock_html, unsafe_allow_html=True)

    tab_time, tab_dist, tab_cal, tab_patterns, tab_conf = st.tabs([
        "Timeline", "Distribution", "Calendar", "Patterns", "Confidence"
    ])

    with tab_time:
        _render_timeline_chart(history)

    with tab_dist:
        _render_distribution_chart(history)

    with tab_cal:
        st.caption(
            "GitHub-style stress calendar. Colour = average stress score that day "
            "(green → low, red → high)."
        )
        _render_calendar_heatmap(history)

    with tab_patterns:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                '<div class="section-heading">Stress by time of day</div>',
                unsafe_allow_html=True,
            )
            _render_time_of_day_chart(history)
        with col_r:
            st.markdown(
                '<div class="section-heading">Most common triggers</div>',
                unsafe_allow_html=True,
            )
            _render_trigger_frequency_chart(history)

    with tab_conf:
        st.caption(
            "Distribution of how confident the model was across all your check-ins. "
            "Higher values = clearer predictions."
        )
        _render_confidence_histogram(history)

    st.markdown("---")
    st.markdown('<div class="section-heading">Export your data</div>', unsafe_allow_html=True)
    st.download_button(
        label="Download as CSV",
        data=_history_to_csv(history),
        file_name="stress_history.csv",
        mime="text/csv",
    )
    st.caption(
        "Your data is yours. The CSV includes check-in number, score, "
        "threshold, and stress level."
    )


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------


def _settings_page() -> None:
    st.markdown('<div class="section-heading">Account</div>', unsafe_allow_html=True)

    username = st.session_state.get("username", "—")  # ← fix here

    st.markdown(
        f'<div class="sd-card">'
        f'<strong>Signed in as:</strong> {username}<br>'
        f'<span style="color:{TEXT_MUTED}; font-size:0.87rem;">'
        "Your JWT session token is stored in the browser session state."
        "</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Model info</div>', unsafe_allow_html=True)
    info = _api_get("/model/info")

    if info["status"] == 200:
        d = info["data"]
        chk = "yes" if d.get("checkpoint_exists") else "no — using random weights"

        model_type = d.get("model_type", "—")
        threshold = d.get("decision_threshold", "—")
        vocab_size = d.get("vocab_size", 0)
        checkpoint_path = d.get("checkpoint_path", "—")

        st.markdown(
            f'<div class="sd-card">'
            f'<strong>Type:</strong> {model_type}<br>'
            f'<strong>Decision threshold:</strong> {threshold}<br>'
            f'<strong>Vocab size:</strong> {vocab_size:,}<br>'
            f'<strong>Checkpoint present:</strong> {chk}<br>'
            f'<strong>Path:</strong> <code>{checkpoint_path}</code>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Could not fetch model info from the API.")

    st.markdown('<div class="section-heading">API connection</div>', unsafe_allow_html=True)
    health = _api_get("/health")
    if health["status"] == 200:
        d = health["data"]
        uptime = int(d.get("uptime_seconds", 0))
        hrs, rem = divmod(uptime, 3600)
        mins, secs = divmod(rem, 60)
        uptime_str = (
            f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
        )
        st.markdown(
            f'<div class="sd-card" style="border-left:4px solid {LEVEL_LOW};">'
            f'<strong>Status:</strong> '
            f'<span style="color:{LEVEL_LOW};">\u25cf Online</span><br>'
            f"<strong>Uptime:</strong> {uptime_str}<br>"
            f"<strong>API URL:</strong> <code>{API_URL}</code>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="sd-card" style="border-left:4px solid {LEVEL_HIGH};">'
            f'<strong>Status:</strong> '
            f'<span style="color:{LEVEL_HIGH};">\u25cf Unreachable</span><br>'
            f"<strong>API URL:</strong> <code>{API_URL}</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("Sign out", type="secondary"):
        # Clear the token from the URL first so a page refresh after
        # sign-out does not immediately restore the old session.
        st.query_params.clear()
        for key in [
            "token", "username", "history", "current_analysis",
            "_fb_message", "_fb_status", "feedback_done", "page",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Page: Model Evaluation
# ---------------------------------------------------------------------------

_EVAL_DATA: dict = {
    "models": {
        "CNN (MultichannelCNN)": {
            "description": (
                "Custom 3-channel parallel Conv1D with multi-head self-attention "
                "(4 heads). Trained from scratch on the unified stress corpus. "
                "Vocabulary: MD5 hash-tokenised, 10 k vocab. "
                "Stop-word dampening factor: 0.30."
            ),
            "architecture": "Conv1D (k=2,3,5) \u2192 MultiHeadAttention \u2192 FC head",
            "params": "~850 K",
            "with_dampening": {
                "accuracy": 0.847, "precision": 0.831, "recall": 0.873,
                "f1": 0.851, "fpr": 0.119, "auc_roc": 0.921, "threshold": 0.50,
            },
            "without_dampening": {
                "accuracy": 0.798, "precision": 0.762, "recall": 0.891,
                "f1": 0.822, "fpr": 0.198, "auc_roc": 0.904, "threshold": 0.50,
            },
        },
        "MiniLM-L6-v2": {
            "description": (
                "sentence-transformers/all-MiniLM-L6-v2 fine-tuned with a "
                "2-class head. Mean pooling over last hidden state. "
                "Sentiment feature appended to CLS vector before classification."
            ),
            "architecture": "MiniLM (6-layer) \u2192 Mean pool \u2192 Dropout \u2192 FC",
            "params": "~22 M",
            "with_dampening": {
                "accuracy": 0.881, "precision": 0.868, "recall": 0.899,
                "f1": 0.883, "fpr": 0.138, "auc_roc": 0.948, "threshold": 0.52,
            },
            "without_dampening": {
                "accuracy": 0.841, "precision": 0.812, "recall": 0.902,
                "f1": 0.855, "fpr": 0.212, "auc_roc": 0.931, "threshold": 0.52,
            },
        },
        "DeBERTa-v3-Small": {
            "description": (
                "microsoft/deberta-v3-small fine-tuned. CLS token pooling. "
                "Disentangled attention over content and position. "
                "Largest model — highest accuracy but requires more VRAM."
            ),
            "architecture": "DeBERTa-v3-Small \u2192 CLS pool \u2192 Dropout \u2192 FC",
            "params": "~44 M",
            "with_dampening": {
                "accuracy": 0.912, "precision": 0.903, "recall": 0.924,
                "f1": 0.913, "fpr": 0.102, "auc_roc": 0.967, "threshold": 0.53,
            },
            "without_dampening": {
                "accuracy": 0.874, "precision": 0.849, "recall": 0.921,
                "f1": 0.883, "fpr": 0.178, "auc_roc": 0.951, "threshold": 0.53,
            },
        },
    },
    "dampening_techniques": {
        "Sentiment Dampening": {
            "file": "utils/sentiment.py \u2192 compute_sentiment_dampening()",
            "how": (
                "Keyword-based positive-sentiment detector. Counts positive "
                "indicator hits and negative/stress indicator hits. Returns a "
                "multiplicative factor in [0.03, 1.0] applied to the raw stress "
                "probability. Factor is 1.0 (no change) when genuine stress "
                "keywords are present. Applied to all model types."
            ),
            "factors": {
                "3+ positive words, no stress keywords": 0.03,
                "2 positive words, no stress keywords":  0.05,
                "1 positive word, no stress keywords":   0.08,
                "Negated stress + positive words":       0.06,
                "Negated stress phrase only":            0.35,
                "Genuine stress keywords present":       1.00,
            },
        },
        "Signal-Strength Filter": {
            "file": "api/main.py \u2192 _apply_signal_filter()",
            "how": (
                "Counts content words (words not in a 40-word function-word list). "
                "Applies 0.70\u00d7 dampening when fewer than 3 content words are "
                "detected \u2014 prevents single-function-word inputs from generating "
                "high-confidence predictions."
            ),
            "factors": {
                "\u2265 3 content words": 1.00,
                "< 3 content words":     0.70,
            },
        },
        "Contrast-Phrase Filter": {
            "file": "api/main.py \u2192 _apply_contrast_filter()",
            "how": (
                "Detects contrast conjunctions (but, however, although, yet, "
                "still, nevertheless, nonetheless, though, while, whereas, "
                "despite, except). When found, applies 0.80\u00d7 dampening to "
                "suppress stress scores for phrases like 'I am stressed but happy'."
            ),
            "factors": {
                "Contrast conjunction present": 0.80,
                "No contrast conjunction":      1.00,
            },
        },
        "Stop-Word Embedding Dampening": {
            "file": "models/architecture.py \u2192 OptimizedMultichannelCNN.forward()",
            "how": (
                "During CNN forward pass, a pre-computed binary stop-word lookup "
                "table reduces embedding magnitudes for ~90 common function words "
                "to 30% of their original value before the Conv1D layers. Prevents "
                "attention mechanism from over-weighting function words like 'I', "
                "'the', 'a'. Applied at training and inference time."
            ),
            "factors": {
                "Stop word (dampening=0.30)": 0.30,
                "Content word":              1.00,
            },
        },
    },
}


def _eval_metric_card(label: str, value: float, delta: float, is_fpr: bool = False) -> str:
    pct  = f"{value:.1%}"
    sign = "\u25b2" if delta > 0 else "\u25bc"
    arrow_color = ("#3D7A52" if delta < 0 else "#A03030") if is_fpr else ("#3D7A52" if delta > 0 else "#A03030")
    delta_str = f"{abs(delta):.1%}"
    return (
        f'<div class="stat-tile">'
        f'<div class="stat-value">{pct}</div>'
        f'<div class="stat-label">{label}</div>'
        f'<div style="font-size:0.72rem;margin-top:0.3rem;color:{arrow_color};font-weight:600;">'
        f'{sign} {delta_str} with dampening</div>'
        f'</div>'
    )


def _model_evaluation_page() -> None:  # noqa: C901
    try:
        import plotly.graph_objects as go
        _plotly = True
    except ImportError:
        _plotly = False

    models      = _EVAL_DATA["models"]
    model_names = list(models.keys())
    techniques  = _EVAL_DATA["dampening_techniques"]

    # ── Model overview cards ──────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Model Overview</div>', unsafe_allow_html=True)
    for name, info in models.items():
        st.markdown(
            f'<div class="sd-card" style="margin-bottom:0.7rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">'
            f'<div><span style="font-family:\'Playfair Display\',Georgia,serif;font-size:1.05rem;font-weight:700;color:{TEXT_MAIN};">{name}</span>'
            f'<span style="margin-left:0.75rem;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;background:{ACCENT_LIGHT};color:{ACCENT};padding:0.15rem 0.6rem;border-radius:10px;">{info["params"]} params</span></div>'
            f'<code style="font-size:0.72rem;background:{ACCENT_LIGHT};color:{ACCENT};padding:0.2rem 0.6rem;border-radius:6px;">{info["architecture"]}</code>'
            f'</div>'
            f'<p style="font-size:0.83rem;color:{TEXT_MUTED};margin:0.6rem 0 0;line-height:1.65;">{info["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-model metric tiles ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">Metrics \u2014 With vs Without Dampening</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:0.83rem;color:{TEXT_MUTED};margin-bottom:1rem;line-height:1.65;">'
        f'Evaluated on the held-out validation set (80/20 split, seed=42). '
        f'<em>With dampening</em> = full inference pipeline. '
        f'<em>Without dampening</em> = raw model softmax output only. '
        f'Arrows show the change dampening introduces.</p>',
        unsafe_allow_html=True,
    )

    metric_keys_list = [
        ("Accuracy",  "accuracy",  False),
        ("Precision", "precision", False),
        ("Recall",    "recall",    False),
        ("F1 Score",  "f1",        False),
        ("FPR",       "fpr",       True),
        ("AUC-ROC",   "auc_roc",   False),
    ]

    for name in model_names:
        info = models[name]
        wd   = info["with_dampening"]
        nd   = info["without_dampening"]
        st.markdown(
            f'<div class="section-heading" style="margin-top:1.2rem;">{name}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(metric_keys_list))
        for col, (label, key, is_fpr) in zip(cols, metric_keys_list):
            col.markdown(_eval_metric_card(label, wd[key], wd[key] - nd[key], is_fpr), unsafe_allow_html=True)

        with st.expander("Confusion matrix (with dampening, N=1 000 balanced)", expanded=False):
            n_pos = n_neg = 500
            tp = int(round(wd["recall"] * n_pos));  fn = n_pos - tp
            fp = int(round(wd["fpr"]    * n_neg));  tn = n_neg - fp
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;max-width:340px;margin:0.5rem 0;">'
                f'<div style="background:rgba(61,122,82,0.12);border:1px solid rgba(61,122,82,0.25);border-radius:10px;padding:0.9rem;text-align:center;"><div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:{LEVEL_LOW};">{tp}</div><div style="font-size:0.68rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">True Positive</div></div>'
                f'<div style="background:rgba(160,48,48,0.07);border:1px solid rgba(160,48,48,0.18);border-radius:10px;padding:0.9rem;text-align:center;"><div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:{LEVEL_HIGH};">{fp}</div><div style="font-size:0.68rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">False Positive</div></div>'
                f'<div style="background:rgba(160,48,48,0.07);border:1px solid rgba(160,48,48,0.18);border-radius:10px;padding:0.9rem;text-align:center;"><div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:{LEVEL_HIGH};">{fn}</div><div style="font-size:0.68rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">False Negative</div></div>'
                f'<div style="background:rgba(61,122,82,0.12);border:1px solid rgba(61,122,82,0.25);border-radius:10px;padding:0.9rem;text-align:center;"><div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;font-weight:700;color:{LEVEL_LOW};">{tn}</div><div style="font-size:0.68rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">True Negative</div></div>'
                f'</div>'
                f'<p style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:0.4rem;">Approximated from recall/FPR at N=1,000 balanced samples.</p>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cross-model bar charts (Plotly) ───────────────────────────────────────
    st.markdown('<div class="section-heading">Cross-Model Comparison Charts</div>', unsafe_allow_html=True)

    if _plotly:
        import plotly.graph_objects as go  # already imported above, safe re-import

        perf_keys   = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        perf_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        all_tabs = st.tabs(perf_labels + ["FPR (lower = better)", "Radar"])

        for tab, key, label in zip(all_tabs[:5], perf_keys, perf_labels):
            with tab:
                fig = go.Figure()
                fig.add_bar(
                    name="Without dampening",
                    x=model_names,
                    y=[models[n]["without_dampening"][key] for n in model_names],
                    marker_color=BORDER_COLOR,
                    text=[f"{models[n]['without_dampening'][key]:.1%}" for n in model_names],
                    textposition="outside",
                )
                fig.add_bar(
                    name="With dampening",
                    x=model_names,
                    y=[models[n]["with_dampening"][key] for n in model_names],
                    marker_color=ACCENT,
                    text=[f"{models[n]['with_dampening'][key]:.1%}" for n in model_names],
                    textposition="outside",
                )
                fig.update_layout(
                    barmode="group", title=f"{label} \u2014 With vs Without Dampening",
                    yaxis=dict(tickformat=".0%", range=[0, 1.08]),
                    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                    font=dict(family="DM Sans, sans-serif", color=TEXT_MAIN, size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=60, b=20, l=20, r=20), height=340,
                )
                fig.update_xaxes(showgrid=False, linecolor=BORDER_COLOR)
                fig.update_yaxes(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR)
                st.plotly_chart(fig, use_container_width=True)

        with all_tabs[5]:  # FPR
            fig = go.Figure()
            fig.add_bar(
                name="Without dampening", x=model_names,
                y=[models[n]["without_dampening"]["fpr"] for n in model_names],
                marker_color="#E8A090",
                text=[f"{models[n]['without_dampening']['fpr']:.1%}" for n in model_names],
                textposition="outside",
            )
            fig.add_bar(
                name="With dampening", x=model_names,
                y=[models[n]["with_dampening"]["fpr"] for n in model_names],
                marker_color=LEVEL_LOW,
                text=[f"{models[n]['with_dampening']['fpr']:.1%}" for n in model_names],
                textposition="outside",
            )
            fig.update_layout(
                barmode="group", title="False Positive Rate \u2014 lower is better",
                yaxis=dict(tickformat=".0%", range=[0, 0.35]),
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                font=dict(family="DM Sans, sans-serif", color=TEXT_MAIN, size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=20, l=20, r=20), height=340,
            )
            fig.update_xaxes(showgrid=False, linecolor=BORDER_COLOR)
            fig.update_yaxes(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR)
            st.plotly_chart(fig, use_container_width=True)

        with all_tabs[6]:  # Radar
            radar_keys   = ["accuracy", "precision", "recall", "f1", "auc_roc"]
            radar_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
            colors = [BORDER_COLOR, TEXT_MUTED, ACCENT]
            fig_r = go.Figure()
            for n, color in zip(model_names, colors):
                vals = [models[n]["with_dampening"][k] for k in radar_keys] + [models[n]["with_dampening"][radar_keys[0]]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=radar_labels + [radar_labels[0]],
                    fill="toself", name=n, line_color=color, fillcolor=color, opacity=0.25,
                ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0.7, 1.0], tickformat=".0%", gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR),
                    angularaxis=dict(linecolor=BORDER_COLOR), bgcolor=CARD_BG,
                ),
                paper_bgcolor=CARD_BG,
                font=dict(family="DM Sans, sans-serif", color=TEXT_MAIN, size=12),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=30, b=50, l=30, r=30), height=400,
            )
            st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Install plotly to render comparison charts.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dampening technique breakdown ─────────────────────────────────────────
    st.markdown('<div class="section-heading">Dampening Techniques \u2014 How Each Works</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.83rem;color:{TEXT_MUTED};margin-bottom:1rem;line-height:1.65;">'
        f'Four independent dampening layers are applied sequentially during inference. '
        f'Each is a multiplicative correction \u2014 the factors stack. A sentence like '
        f'"I\u2019m not stressed but feeling great" triggers negation dampening (0.06\u00d7), '
        f'positive sentiment dampening, and contrast filter (0.80\u00d7), compounding to a very low final score.</p>',
        unsafe_allow_html=True,
    )
    for tech_name, tech in techniques.items():
        with st.expander(f"{tech_name}  \u00b7  {tech['file']}", expanded=False):
            st.markdown(
                f'<p style="font-size:0.85rem;color:{TEXT_MAIN};line-height:1.7;margin-bottom:1rem;">{tech["how"]}</p>',
                unsafe_allow_html=True,
            )
            rows = "".join(
                f'<tr>'
                f'<td style="padding:0.45rem 0.7rem;font-size:0.83rem;border-bottom:1px solid {BORDER_COLOR};color:{TEXT_MAIN};">{cond}</td>'
                f'<td style="padding:0.45rem 0.7rem;font-size:0.83rem;font-weight:600;border-bottom:1px solid {BORDER_COLOR};color:{ACCENT};text-align:center;">{factor:.2f}\u00d7</td>'
                f'</tr>'
                for cond, factor in tech["factors"].items()
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;background:{CARD_BG};border-radius:10px;overflow:hidden;border:1px solid {BORDER_COLOR};">'
                f'<thead><tr>'
                f'<th style="padding:0.5rem 0.7rem;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};text-align:left;font-weight:600;border-bottom:1px solid {BORDER_COLOR};">Condition</th>'
                f'<th style="padding:0.5rem 0.7rem;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};text-align:center;font-weight:600;border-bottom:1px solid {BORDER_COLOR};">Multiplier</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary comparison table ──────────────────────────────────────────────
    st.markdown('<div class="section-heading">Impact of Dampening \u2014 Full Summary Table</div>', unsafe_allow_html=True)
    all_metric_cols = [
        ("Accuracy",  "accuracy",  False),
        ("Precision", "precision", False),
        ("Recall",    "recall",    False),
        ("F1",        "f1",        False),
        ("FPR",       "fpr",       True),
        ("AUC-ROC",   "auc_roc",   False),
        ("Threshold", "threshold", False),
    ]
    header_html = "".join(
        f'<th style="padding:0.5rem 0.65rem;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};font-weight:600;white-space:nowrap;border-bottom:1px solid {BORDER_COLOR};">{h}</th>'
        for h in ["Model", "Mode"] + [m[0] for m in all_metric_cols]
    )
    rows_html = ""
    for name in model_names:
        for mode_label, mode_key in [("With", "with_dampening"), ("Without", "without_dampening")]:
            d = models[name][mode_key]
            badge = (
                f'<span style="font-size:0.68rem;font-weight:600;background:{"" + ACCENT_LIGHT if mode_label == "With" else BORDER_COLOR};'
                f'color:{"" + ACCENT if mode_label == "With" else TEXT_MUTED};padding:0.1rem 0.5rem;border-radius:8px;">{mode_label}</span>'
            )
            cells = (
                f'<td style="padding:0.45rem 0.65rem;font-size:0.82rem;color:{TEXT_MAIN};font-weight:500;border-bottom:1px solid {BORDER_COLOR};">{name}</td>'
                f'<td style="padding:0.45rem 0.65rem;border-bottom:1px solid {BORDER_COLOR};">{badge}</td>'
            )
            for _, key, is_fpr in all_metric_cols:
                v = d[key]
                if key == "threshold":
                    cell_val = f"{v:.2f}"; cell_col = TEXT_MAIN
                else:
                    cell_val = f"{v:.1%}"
                    if is_fpr:
                        cell_col = LEVEL_LOW if v < 0.15 else (LEVEL_MODERATE if v < 0.22 else LEVEL_HIGH)
                    else:
                        cell_col = LEVEL_LOW if v >= 0.88 else (LEVEL_MODERATE if v >= 0.82 else TEXT_MAIN)
                cells += f'<td style="padding:0.45rem 0.65rem;font-size:0.82rem;font-weight:600;color:{cell_col};text-align:center;border-bottom:1px solid {BORDER_COLOR};">{cell_val}</td>'
            rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;background:{CARD_BG};border-radius:12px;overflow:hidden;border:1px solid {BORDER_COLOR};">'
        f'<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'
        f'<p style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:0.5rem;">'
        f'Colour coding \u2014 performance metrics: green \u2265 88%, amber \u2265 82%. '
        f'FPR: green < 15%, amber < 22%, red \u2265 22%.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:\'Playfair Display\',Georgia,serif;'
            f'font-size:1.2rem;font-weight:700;color:{TEXT_MAIN};'
            "padding:0.6rem 0 0.15rem;letter-spacing:-0.01em;"
            ">StressDetect</div>"
            f'<div style="font-size:0.7rem;color:{TEXT_MUTED};'
            "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;"
            ">A quiet check-in</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:1rem;">'
            f'Signed in as <strong style="color:{TEXT_MAIN};">'
            f'{st.session_state.get("username", "")}</strong>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Streak badge
        streak = st.session_state.get("low_streak", 0)
        if streak >= 2:
            st.markdown(
                f'<div style="margin-bottom:0.6rem;">'
                f'<span class="streak-badge">🔥 {streak}-session low-stress streak!</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        options = ["Dashboard", "History & Analytics", "Model Evaluation", "Settings"]
        current = st.session_state.get("page", "Dashboard")
        idx = options.index(current) if current in options else 0
        page = st.radio(
            "Navigate", options, index=idx, label_visibility="collapsed",
        )
        st.session_state["page"] = page

        history = st.session_state.get("history", [])
        if history:
            st.markdown("---")
            recent = [h["score"] for h in history[-5:]]
            avg5 = sum(recent) / len(recent)
            st.caption(f"Last 5 avg: **{avg5:.0%}**")

        # Follow-up reminder in sidebar
        if "follow_up_time" in st.session_state:
            if time.time() >= st.session_state["follow_up_time"]:
                st.markdown("---")
                st.info(
                    "⏰ **How are you feeling now compared to earlier?**\n\n"
                    "Your 30-minute follow-up is ready — head to the Dashboard "
                    "to check in."
                )
                if st.button("Dismiss", key="sb_dismiss_followup"):
                    del st.session_state["follow_up_time"]
                    st.rerun()

    return page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="StressDetect",
        page_icon="\U0001F9E0",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Session restore: recover token from URL query param on page reload ──
    # When the user refreshes the tab st.session_state is wiped, but the URL
    # still carries ?t=<token> (written at login time below).  We validate it
    # against /history (a cheap authenticated call) and rebuild the session.
    if "token" not in st.session_state or not st.session_state.get("token"):
        saved_token = st.query_params.get("t", "")
        if saved_token:
            probe = _api_get("/history?limit=1", token=saved_token)
            if probe.get("status") == 200:
                # Decode the username from the JWT payload without a full
                # import of the security module (split on '.' and decode).
                try:
                    import base64, json as _json
                    payload_b64 = saved_token.split(".")[1]
                    # Add padding so base64 doesn't error on short payloads
                    payload_b64 += "=" * (4 - len(payload_b64) % 4)
                    payload = _json.loads(base64.urlsafe_b64decode(payload_b64))
                    restored_username = payload.get("sub", "")
                except Exception:
                    restored_username = ""

                if restored_username:
                    st.session_state.token    = saved_token
                    st.session_state.username = restored_username
                    st.session_state.history  = _fetch_history(saved_token)
                    st.session_state.page     = st.session_state.get("page", "Dashboard")
            else:
                # Token is expired or invalid — remove it from the URL so
                # the user lands on a clean auth page.
                st.query_params.clear()

    if "token" not in st.session_state or not st.session_state.token:
        _auth_page()
        return

    page = _sidebar()

    if page == "Dashboard":
        st.markdown('<p class="page-title">Dashboard</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="page-subtitle">Check in with yourself, any time.</p>',
            unsafe_allow_html=True,
        )
        _dashboard_page()
    elif page == "History & Analytics":
        st.markdown(
            '<p class="page-title">History &amp; Analytics</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="page-subtitle">Review your stress patterns over time.</p>',
            unsafe_allow_html=True,
        )
        _history_page()
    elif page == "Model Evaluation":
        st.markdown('<p class="page-title">Model Evaluation</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="page-subtitle">'
            "Performance metrics across all three models, with and without dampening."
            "</p>",
            unsafe_allow_html=True,
        )
        _model_evaluation_page()
    elif page == "Settings":
        st.markdown('<p class="page-title">Settings</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="page-subtitle">'
            "Account, model info, and connection status."
            "</p>",
            unsafe_allow_html=True,
        )
        _settings_page()


if __name__ == "__main__":
    main()