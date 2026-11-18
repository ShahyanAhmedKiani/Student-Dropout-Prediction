"""Student Dropout Prediction - Streamlit app.

Loads the joblib bundle in trainmodel.pkl (LogisticRegression + StandardScaler +
feature metadata) and predicts the probability that a student drops out.
"""

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = Path(__file__).parent / "trainmodel.pkl"

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
MODEL = bundle["model"]
SCALER = bundle["scaler"]
FEATURES = bundle["feature_columns"]
DEFAULTS = bundle["defaults"]
MINS = bundle["mins"]
MAXS = bundle["maxs"]


def predict_dropout(age, gender, sem1, sem2, scholarship, tuition):
    """Return dropout probability (0-1). Unexposed features use stored medians."""
    row = dict(DEFAULTS)
    row.update(
        {
            "Age at enrollment": age,
            "Gender": 1 if gender == "Male" else 0,  # UCI coding: 1 = male, 0 = female
            "Curricular units 1st sem (approved)": sem1,
            "Curricular units 2nd sem (approved)": sem2,
            "Scholarship holder": 1 if scholarship == "Yes" else 0,
            "Tuition fees up to date": 1 if tuition == "Yes" else 0,
        }
    )
    X = pd.DataFrame([row], columns=FEATURES)
    X_scaled = SCALER.transform(X)
    return float(MODEL.predict_proba(X_scaled)[0, 1])


def risk_level(p):
    if p < 0.30:
        return "Low Dropout Risk", "low"
    if p < 0.60:
        return "Medium Dropout Risk", "medium"
    return "High Dropout Risk", "high"


# --------------------------------------------------------------------------- #
# Icons (inline SVG so nothing external is required)
# --------------------------------------------------------------------------- #
ICON = {
    "cap": '<svg viewBox="0 0 640 512" fill="currentColor"><path d="M320 32c-8 0-16 1.5-23.5 4.3L15.8 140.4C6.2 144 0 153.1 0 163.3s6.2 19.3 15.8 22.9l57.9 21.4C57.3 229.3 48 256.4 48 285.5V312c0 0 0 0 0 0v1.5c-12.5 8.5-21.3 22-23.5 37.5L8 456c-1 8 5.3 15.1 13.3 15.1h53.3c8 0 14.3-7.1 13.3-15.1L71.5 351c-2.2-15.5-11-29-23.5-37.5V285.5c0-24.1 8.4-46.6 22.9-64.4L296.4 307.7c7.5 2.8 15.5 4.3 23.6 4.3s16-1.5 23.5-4.3L624.2 203.6c9.6-3.6 15.8-12.7 15.8-22.9s-6.2-19.3-15.8-22.9L343.5 36.3C336 33.5 328 32 320 32zM128 408c0 35.3 86 72 192 72s192-36.7 192-72L496.7 262.6 354.5 315.1c-11.1 4.1-22.8 6.2-34.5 6.2s-23.5-2.1-34.5-6.2L143.3 262.6 128 408z"/></svg>',
    "user": '<svg viewBox="0 0 448 512" fill="currentColor"><path d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3 0 498.7 13.3 512 29.7 512h388.6c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/></svg>',
    "chart": '<svg viewBox="0 0 448 512" fill="currentColor"><rect x="0" y="288" width="112" height="192" rx="24"/><rect x="168" y="176" width="112" height="304" rx="24"/><rect x="336" y="48" width="112" height="432" rx="24"/></svg>',
    "book": '<svg viewBox="0 0 576 512" fill="currentColor"><path d="M249.6 471.5c10.8 3.8 22.4-4.1 22.4-15.5V78.6c0-4.2-1.6-8.4-5-11C247.4 52 202.4 32 144 32 93.5 32 46.3 45.3 18.1 56.1 6.8 60.5 0 71.7 0 83.8v370.3c0 11.9 12.8 20.2 24.1 16.5C55.6 460.1 105.5 448 144 448c33.9 0 79 14 105.6 23.5zm76.8 0C353 462 398.1 448 432 448c38.5 0 88.4 12.1 119.9 22.6 11.3 3.8 24.1-4.6 24.1-16.5V83.8c0-12.1-6.8-23.3-18.1-27.6C529.7 45.3 482.5 32 432 32c-58.4 0-103.4 20-123 35.6-3.3 2.6-5 6.8-5 11V456c0 11.4 11.7 19.3 22.4 15.5z"/></svg>',
    "money": '<svg viewBox="0 0 512 512" fill="currentColor"><path d="M320 96H192L144.6 24.9C137.5 14.2 145.1 0 157.9 0h196.2c12.8 0 20.4 14.2 13.3 24.9L320 96zM192 128h128c3.8 2.5 8.1 5.3 13 8.4C389.7 172.7 512 250.9 512 416c0 53-43 96-96 96H96c-53 0-96-43-96-96 0-165.1 122.3-243.3 179-279.6 4.8-3.1 9.2-5.9 13-8.4zm84 88c0-11-9-20-20-20s-20 9-20 20v14c-7.6 1.7-15.2 4.4-22.2 8.5-13.9 8.3-25.9 22.8-25.8 43.9.1 20.3 12 33.1 24.7 40.7 11 6.6 24.7 10.8 35.6 14l1.7.5c12.6 3.8 21.8 6.8 28 10.7 5.1 3.2 5.8 5.4 5.9 8.2.1 5-1.8 8-5.9 10.5-5 3.1-12.9 5-21.4 4.7-11.1-.4-21.5-3.9-35.1-8.5-2.3-.8-4.7-1.6-7.2-2.4-10.5-3.5-21.8 2.2-25.3 12.6s2.2 21.8 12.6 25.3c1.9.6 4 1.3 6.1 2.1 8.3 2.9 17.9 6.2 28.2 8.4V424c0 11 9 20 20 20s20-9 20-20v-13.8c8-1.7 16-4.5 23.2-9 14.3-8.9 25.1-24.1 24.8-45-.3-20.3-11.7-33.4-24.6-41.6-11.5-7.2-25.9-11.6-37.1-15l-.7-.2c-12.8-3.9-21.9-6.7-28.3-10.5-5.2-3.1-5.3-4.9-5.3-6.7 0-3.7 1.4-6.5 6.2-9.3 5.4-3.2 13.6-5.1 21.5-5 9.6.1 20.2 2.2 31.2 5.2 10.7 2.8 21.6-3.5 24.5-14.2s-3.5-21.6-14.2-24.5c-6.5-1.7-13.7-3.4-21.1-4.7V216z"/></svg>',
    "check": '<svg viewBox="0 0 512 512" fill="currentColor"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zm113-303L241 337c-9.4 9.4-24.6 9.4-33.9 0l-64-64c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l47 47L335 175c9.4-9.4 24.6-9.4 33.9 0s9.4 24.6 0 33.9z"/></svg>',
    "warn": '<svg viewBox="0 0 512 512" fill="currentColor"><path d="M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3 27.7.2 40.1S486.3 480 472 480H40c-14.3 0-27.6-7.7-34.7-20.1s-7-27.8.2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128c-13.3 0-24 10.7-24 24v112c0 13.3 10.7 24 24 24s24-10.7 24-24V184c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0-64 0 32 32 0 1 0 64 0z"/></svg>',
    "info": '<svg viewBox="0 0 512 512" fill="currentColor"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336h24v-64h-24c-13.3 0-24-10.7-24-24s10.7-24 24-24h48c13.3 0 24 10.7 24 24v88h8c13.3 0 24 10.7 24 24s-10.7 24-24 24h-80c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"/></svg>',
    "heart": '<svg viewBox="0 0 512 512" fill="#e53935"><path d="M47.6 300.4 228.3 469.1c7.5 7 17.4 10.9 27.7 10.9s20.2-3.9 27.7-10.9l180.7-168.7c30.4-28.3 47.6-68 47.6-109.5v-5.8c0-69.9-50.5-129.5-119.4-141C347 36.5 300.6 51.4 268 84L256 96 244 84c-32.6-32.6-79-47.5-124.6-39.9C50.5 55.6 0 115.2 0 185.1v5.8c0 41.5 17.2 81.2 47.6 109.5z"/></svg>',
}

# Decorative illustration for the banner (books + graduation cap)
BANNER_ART = """
<svg class="banner-art" viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="130" cy="140" rx="120" ry="12" fill="#1f7a5a" opacity=".55"/>
  <g>
    <rect x="40" y="104" width="180" height="30" rx="4" fill="#1d4f91"/>
    <rect x="46" y="108" width="170" height="22" rx="2" fill="#f4f6fb"/>
    <rect x="46" y="112" width="170" height="2" fill="#d5dcea"/>
    <rect x="46" y="118" width="170" height="2" fill="#d5dcea"/>
    <rect x="46" y="124" width="170" height="2" fill="#d5dcea"/>
    <rect x="58" y="76" width="160" height="28" rx="4" fill="#163d73"/>
    <rect x="62" y="80" width="150" height="20" rx="2" fill="#eef2f9"/>
    <rect x="62" y="85" width="150" height="2" fill="#d5dcea"/>
    <rect x="62" y="92" width="150" height="2" fill="#d5dcea"/>
    <rect x="48" y="50" width="150" height="26" rx="4" fill="#2563a8"/>
    <rect x="52" y="54" width="140" height="18" rx="2" fill="#f4f6fb"/>
    <rect x="52" y="59" width="140" height="2" fill="#d5dcea"/>
    <rect x="52" y="65" width="140" height="2" fill="#d5dcea"/>
  </g>
  <g>
    <path d="M100 40 L160 14 L230 38 L170 64 Z" fill="#0d1f3c"/>
    <path d="M122 46 L122 70 Q166 88 206 70 L206 46 L170 62 Z" fill="#132a4f"/>
    <circle cx="165" cy="38" r="4" fill="#0a1830"/>
    <path d="M165 38 Q205 46 214 60 L214 96" stroke="#f2b632" stroke-width="3" fill="none"/>
    <path d="M208 96 L220 96 L224 122 L204 122 Z" fill="#f2b632"/>
  </g>
</svg>
"""


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
st.markdown(
    f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def card_header(icon: str, title: str):
    html(f'<div class="card-header">{ICON[icon]}<span>{title}</span></div>')


def section(icon: str, title: str):
    html(f'<div class="section">{ICON[icon]}<span>{title}</span></div>')


def gauge_svg(p: float, level: str) -> str:
    color = {"low": "#1aab3c", "medium": "#f2a51a", "high": "#e03131"}[level]
    pct = round(p * 100)
    # Semicircle arc: centre (200,200), radius 160, length = pi * r
    arc = "M 40 200 A 160 160 0 0 1 360 200"
    length = 3.14159 * 160
    filled = max(0.0, min(1.0, p)) * length
    return f"""
<svg viewBox="0 0 400 225" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Dropout probability {pct} percent">
  <path d="{arc}" fill="none" stroke="#e1e4ea" stroke-width="44"/>
  <path d="{arc}" fill="none" stroke="{color}" stroke-width="44"
        stroke-dasharray="{filled:.2f} {length:.2f}"/>
  <text x="200" y="192" text-anchor="middle" font-family="Inter, sans-serif"
        font-size="92" font-weight="800" fill="#13213d">{pct}%</text>
</svg>"""


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
html(
    f"""
<div class="banner">
  <div class="banner-icon">{ICON['cap']}</div>
  <div class="banner-divider"></div>
  <div class="banner-text">
    <h1>Student Dropout Prediction</h1>
    <p>AI-powered Student Risk Assessment</p>
  </div>
  {BANNER_ART}
</div>
"""
)

left, right = st.columns([1.1, 1], gap="medium")

with left:
    with st.container(key="input_card"):
        card_header("user", "Student Information")

        section("user", "Personal Details")
        c1, c2 = st.columns(2)
        age = c1.number_input(
            "Age at Enrollment",
            min_value=int(MINS["Age at enrollment"]),
            max_value=int(MAXS["Age at enrollment"]),
            value=int(DEFAULTS["Age at enrollment"]),
            step=1,
        )
        gender = c2.selectbox("Gender", ["Female", "Male"], index=0)

        section("book", "Academic Information")
        c3, c4 = st.columns(2)
        sem1 = c3.number_input(
            "First Semester Approved Units",
            min_value=0,
            max_value=int(MAXS["Curricular units 1st sem (approved)"]),
            value=5,
            step=1,
        )
        sem2 = c4.number_input(
            "Second Semester Approved Units",
            min_value=0,
            max_value=int(MAXS["Curricular units 2nd sem (approved)"]),
            value=6,
            step=1,
        )

        section("money", "Financial Information")
        c5, c6 = st.columns(2)
        scholarship = c5.selectbox("Scholarship Holder", ["Yes", "No"], index=0)
        tuition = c6.selectbox("Tuition Fees Up to Date", ["Yes", "No"], index=0)

        clicked = st.button(
            "Predict Dropout Risk", icon=":material/bar_chart:", width="stretch"
        )

inputs = (age, gender, sem1, sem2, scholarship, tuition)
if clicked or "prob" not in st.session_state:
    st.session_state.prob = predict_dropout(*inputs)
    st.session_state.inputs = inputs

prob = st.session_state.prob
label, level = risk_level(prob)
stale = st.session_state.inputs != inputs

with right:
    with st.container(key="result_card"):
        card_header("chart", "Prediction Result")
        html(f'<div class="gauge-wrap">{gauge_svg(prob, level)}</div>')

        icon = "check" if level == "low" else "warn"
        html(f'<div class="risk {level}">{ICON[icon]}<span>{label}</span></div>')

        def cell(key, name, rng, color):
            active = " active" if key == level else ""
            return (
                f'<div class="guide-cell{active}"><div class="name">'
                f'<span class="dot" style="background:{color}"></span>{name}</div>'
                f'<div class="range">{rng}</div></div>'
            )

        html(
            f"""
<div class="guide">
  <div class="guide-head">{ICON['info']}<span>Risk Guide</span></div>
  <div class="guide-grid">
    {cell("low", "Low Risk", "&lt; 30%", "#1aab3c")}
    {cell("medium", "Medium Risk", "30% &ndash; 59%", "#f2a51a")}
    {cell("high", "High Risk", "&ge; 60%", "#e03131")}
  </div>
</div>
"""
        )

        note = (
            "Inputs changed &mdash; click <b>Predict Dropout Risk</b> to update."
            if stale
            else "Predictions are decision-support estimates, not guarantees."
        )
        html(f'<div class="note">{ICON["info"]}<span>{note}</span></div>')

html(
    f"""
<div class="app-footer">
  <div class="left">{ICON['cap']}
    <span>Student Dropout Prediction<span class="sep">|</span>Machine Learning Project</span>
  </div>
  <div class="right">Built with {ICON['heart']} using Streamlit</div>
</div>
"""
)
