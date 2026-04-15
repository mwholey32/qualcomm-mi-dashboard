"""
app.py — Streamlit dashboard: Qualcomm market-intelligence framework.

Single-page layout with global sidebar filters across four stages:

    Stage 2  Supply / publishing  leading  3–12 mo
    Stage 3  Developer intent     leading  1–6 mo
    Stage 4  Productionization    concurrent
    Stage 5  End-user runtime     lagging

Run:
    streamlit run app.py

Data comes from the parquet snapshots in ./data. Run the collectors first:
    python collect.py            # HF vendor + framework + spaces
    python collect_cards.py      # Qualcomm model card READMEs
    python collect_pypi.py       # PyPI downloads (developer intent)
    python collect_playstore.py  # Google Play Store install ranges
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import analyze
import products as products_mod

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
CURATED_DIR = Path(__file__).parent / "curated"

st.set_page_config(
    page_title="Qualcomm MI Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — match policy-tracker design language
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg:           #f4f4f4;
    --surface:      #ffffff;
    --border:       #e0e0e0;
    --border-faint: #ebebeb;
    --navy:         #2c2c2c;
    --text:         #1a1a1a;
    --text-2:       #555555;
    --text-3:       #999999;
  }

  /* Global font */
  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  }

  /* Background */
  .stApp, .main .block-container {
    background: var(--bg) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: var(--surface) !important;
  }

  /* Hide Streamlit header/footer chrome */
  #MainMenu, footer, header[data-testid="stHeader"] {
    display: none !important;
  }

  /* Expander styling — match filter-sec pattern */
  .streamlit-expanderHeader {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-2) !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-faint) !important;
    padding: 0.5rem 0.5rem !important;
  }
  .streamlit-expanderHeader:hover {
    color: var(--text) !important;
    background: var(--bg) !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-3) !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
  }

  /* Markdown headers */
  .stMarkdown h1 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
  }
  .stMarkdown h2, [data-testid="stHeadingWithActionElements"] h2 {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
  }
  .stMarkdown h3 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
  }
  .stMarkdown h4 {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--text-2) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .stMarkdown h5 {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: var(--text-3) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* Captions */
  .stMarkdown small, [data-testid="stCaptionContainer"] {
    font-size: 0.72rem !important;
    color: var(--text-3) !important;
    line-height: 1.55 !important;
  }

  /* Buttons — match btn-refresh */
  .stButton > button {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    border-radius: 7px !important;
    padding: 0.35rem 0.75rem !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    background: var(--border) !important;
    color: var(--text) !important;
  }

  /* Select slider */
  [data-testid="stSlider"] label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-3) !important;
  }

  /* Sidebar checkboxes — compact, policy-tracker src-item style */
  section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    padding: 0 !important;
    min-height: 0 !important;
    margin-bottom: -0.55rem !important;
  }
  section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    gap: 0.28rem !important;
    align-items: center !important;
    padding: 0.05rem 0.2rem !important;
    cursor: pointer !important;
    border-radius: 3px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {
    background: var(--bg) !important;
  }
  section[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
    font-size: 0.7rem !important;
    color: var(--text-2) !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
  }

  /* All/None gold text links inside sidebar expanders */
  section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton {
    min-width: 0 !important;
    width: auto !important;
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: #b8852a !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    padding: 0 !important;
    min-height: 0 !important;
    min-width: 0 !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button:hover {
    text-decoration: underline !important;
    background: none !important;
    color: #9e7020 !important;
    border: none !important;
    box-shadow: none !important;
  }

  /* Tighter column gaps inside sidebar expanders */
  section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
    gap: 0.1rem !important;
    align-items: center !important;
  }

  /* Nested expanders — vendor sub-sections, lighter than outer */
  section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpander"] {
    border: none !important;
    border-bottom: 1px solid var(--border-faint) !important;
    margin-bottom: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpander"] .streamlit-expanderHeader {
    font-size: 0.71rem !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: var(--text) !important;
    padding: 0.28rem 0.4rem !important;
    border-bottom: none !important;
    background: transparent !important;
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpander"] .streamlit-expanderContent {
    padding: 0 0.3rem 0.3rem !important;
  }

  /* Dataframes */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
  }

  /* Info/warning boxes */
  [data-testid="stAlert"] {
    font-size: 0.78rem !important;
    border-radius: 8px !important;
  }

  /* Plotly charts — remove excess padding */
  [data-testid="stPlotlyChart"] {
    border: 1px solid var(--border-faint);
    border-radius: 8px;
    background: var(--surface);
    padding: 0.3rem;
  }

  /* Dividers */
  hr {
    border-color: var(--border-faint) !important;
  }

  /* Sidebar section labels */
  section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.75rem !important;
    color: var(--text-2) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown strong {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-3) !important;
  }

  /* Code blocks in sidebar */
  section[data-testid="stSidebar"] .stCodeBlock {
    font-size: 0.7rem !important;
  }

  /* Fix: ensure content area scrolls to full height when expanders toggle */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] {
    overflow: auto !important;
  }
  .main .block-container {
    overflow: visible !important;
    max-height: none !important;
  }
  [data-testid="stExpander"] details[open] {
    overflow: visible !important;
  }
</style>
""", unsafe_allow_html=True)

QUALCOMM_BLUE = "#3253DC"


# ---------------------------------------------------------------------------
# Central color palette
# ---------------------------------------------------------------------------

# Distinct vendor colors — optimized for legibility across line charts
_APPLE_COLOR       = "#A855F7"   # Purple
_GOOGLE_COLOR      = "#22C55E"   # Green
_MICROSOFT_COLOR   = "#06B6D4"   # Cyan
_INTEL_COLOR       = "#0EA5E9"   # Sky blue
_META_COLOR        = "#F97316"   # Orange
_ARM_COLOR         = "#EC4899"   # Pink
_HF_COLOR          = "#FBBF24"   # Amber
_MEDIATEK_COLOR    = "#EF4444"   # Red
_SAMSUNG_COLOR     = "#14B8A6"   # Teal
_HUAWEI_COLOR      = "#D946EF"   # Fuchsia

_NVIDIA_GREY       = "#6B6B6B"
_AMD_GREY          = "#8E8E8E"
_AMBIGUOUS_GREY    = "#BFBFBF"

VENDOR_COLOR: dict[str, str] = {
    "qualcomm":           QUALCOMM_BLUE,
    "Qualcomm":           QUALCOMM_BLUE,
    "apple":              _APPLE_COLOR,
    "Apple":              _APPLE_COLOR,
    "google":             _GOOGLE_COLOR,
    "Google":             _GOOGLE_COLOR,
    "microsoft":          _MICROSOFT_COLOR,
    "Microsoft":          _MICROSOFT_COLOR,
    "intel":              _INTEL_COLOR,
    "Intel":              _INTEL_COLOR,
    "meta":               _META_COLOR,
    "Meta":               _META_COLOR,
    "Meta / cross":       _META_COLOR,
    "MediaTek":           _MEDIATEK_COLOR,
    "samsung":            _SAMSUNG_COLOR,
    "Samsung":            _SAMSUNG_COLOR,
    "Arm":                _ARM_COLOR,
    "Huawei":             _HUAWEI_COLOR,
    "HuggingFace":        _HF_COLOR,
    "nvidia":             _NVIDIA_GREY,
    "NVIDIA":             _NVIDIA_GREY,
    "amd":                _AMD_GREY,
    "AMD":                _AMD_GREY,
    "arduino":            "#00878F",
    "Arduino":            "#00878F",
}

TIER_COLOR: dict[str, str] = {
    "On-device":      QUALCOMM_BLUE,
    "Edge / hybrid":  _GOOGLE_COLOR,
    "Datacenter":     _NVIDIA_GREY,
    "Ambiguous":      _AMBIGUOUS_GREY,
    "Unclassified":   "#DFDFDF",
}

FRAMEWORK_COLOR: dict[str, str] = {
    "QNN (Qualcomm)":       QUALCOMM_BLUE,
    "Core ML (Apple)":      _APPLE_COLOR,
    "MLX (Apple Silicon)":  _APPLE_COLOR,
    "LiteRT / TFLite":      _GOOGLE_COLOR,
    "ExecuTorch":           _META_COLOR,
    "ONNX":                 _MICROSOFT_COLOR,
    "GGUF (llama.cpp)":     _AMBIGUOUS_GREY,
    "OpenVINO (Intel)":     _INTEL_COLOR,
    "CUDA / TensorRT":      _NVIDIA_GREY,
}

BUCKET_COLOR: dict[str, str] = {
    "Qualcomm / Snapdragon": QUALCOMM_BLUE,
    "Apple":                 _APPLE_COLOR,
    "Google":                _GOOGLE_COLOR,
    "Intel":                 _INTEL_COLOR,
    "Generic NPU / Edge":    _AMBIGUOUS_GREY,
    "NVIDIA / CUDA":         _NVIDIA_GREY,
    "AMD":                   _AMD_GREY,
}

PACKAGE_COLOR: dict[str, str] = {
    # Qualcomm
    "qai-hub":                      QUALCOMM_BLUE,
    "qai-hub-models":               QUALCOMM_BLUE,
    # Apple
    "coremltools":                  _APPLE_COLOR,
    "mlx":                          _APPLE_COLOR,
    "mlx-lm":                       _APPLE_COLOR,
    # Google
    "ai-edge-torch":                _GOOGLE_COLOR,
    "tflite-runtime":               _GOOGLE_COLOR,
    "mediapipe":                    _GOOGLE_COLOR,
    "jax":                          _GOOGLE_COLOR,
    # Microsoft
    "onnxruntime":                  _MICROSOFT_COLOR,
    "onnxruntime-gpu":              _MICROSOFT_COLOR,
    "onnxruntime-directml":         _MICROSOFT_COLOR,
    "olive-ai":                     _MICROSOFT_COLOR,
    "torch-directml":               _MICROSOFT_COLOR,
    # Intel
    "openvino":                     _INTEL_COLOR,
    "openvino-dev":                 _INTEL_COLOR,
    "optimum-intel":                _INTEL_COLOR,
    "intel-extension-for-pytorch":  _INTEL_COLOR,
    "neural-compressor":            _INTEL_COLOR,
    # Meta
    "executorch":                   _META_COLOR,
    # Arm
    "ethos-u-vela":                 _ARM_COLOR,
    # HuggingFace
    "optimum":                      _HF_COLOR,
    # NVIDIA
    "tensorrt":                     _NVIDIA_GREY,
    "tensorrt-llm":                 _NVIDIA_GREY,
    "nemo-toolkit":                 _NVIDIA_GREY,
    "tritonclient":                 _NVIDIA_GREY,
    "cuda-python":                  _NVIDIA_GREY,
    # Edge Impulse (Qualcomm-owned)
    "edgeimpulse":                  QUALCOMM_BLUE,
    "edgeimpulse-api":              QUALCOMM_BLUE,
    "edge-impulse-linux":           QUALCOMM_BLUE,
    # Arduino
    "arduino-iot-cloud":            "#00878F",
    "arduino-iot-client":           "#00878F",
}


# ---------------------------------------------------------------------------
# Stage metadata (stages 2-5)
# ---------------------------------------------------------------------------

STAGE_META: dict[str, dict] = {
    "stage2": {
        "title": "1 · Supply / Publishing",
        "proxy": "strong",
        "cadence": "daily",
        "timing": "leading (3–12 mo)",
        "signals": "HF Hub — vendors · runtimes · products · deployment tier",
        "lens": (
            "What vendors actually publish. HF Hub captures every vendor's "
            "'I am building for this target' commitment as a timestamped "
            "model push. The deployment-tier view below shows which targets "
            "are structurally on-device vs datacenter."
        ),
    },
    "stage3": {
        "title": "2 · Developer Intent",
        "proxy": "strong",
        "cadence": "daily",
        "timing": "leading (1–6 mo)",
        "signals": "PyPI downloads · Reddit mentions · GitHub issues · GitHub repos",
        "lens": (
            "What developers actually install and talk about. A `pip "
            "install` is unambiguous build intent; a Reddit post or GitHub "
            "issue mentioning a vendor is organic mindshare — unprompted, "
            "timestamped, and scored by the community."
        ),
    },
    "stage4": {
        "title": "3 · Productionization",
        "proxy": "medium",
        "cadence": "daily",
        "timing": "concurrent",
        "signals": "HF Spaces — deployed demos + keyword mindshare",
        "lens": (
            "What people have actually shipped publicly. HF Spaces is a "
            "weak proxy for full production but a usable leading edge for "
            "ecosystem mindshare — a Space for a keyword means at least "
            "one team stood up a demo for it."
        ),
    },
    "stage5": {
        "title": "4 · End-User Runtime",
        "proxy": "medium",
        "cadence": "on-demand",
        "timing": "lagging",
        "signals": "Play Store installs · OEM design wins",
        "lens": (
            "What's actually running on deployed hardware. The hardest "
            "signal to measure because it happens silently inside apps on "
            "billions of devices — we triangulate from Play Store install "
            "ranges, OEM flagship spec sheets, and curated chipset wins."
        ),
    },
}

PROXY_COLOR = {"strong": "green", "medium": "orange", "weak": "red"}
TIMING_COLOR = {
    "leading (6–24 mo)": "green",
    "leading (3–12 mo)": "green",
    "leading (1–6 mo)": "green",
    "concurrent": "orange",
    "lagging": "red",
}


def stage_header(stage_key: str):
    """Render a stage title + colored signal-quality badges + lens blurb."""
    meta = STAGE_META[stage_key]
    st.subheader(meta["title"])
    proxy_c = PROXY_COLOR.get(meta["proxy"], "gray")
    timing_c = TIMING_COLOR.get(meta["timing"], "gray")
    badges = (
        f":{proxy_c}[**proxy: {meta['proxy']}**]  ·  "
        f":blue[**cadence: {meta['cadence']}**]  ·  "
        f":{timing_c}[**timing: {meta['timing']}**]  ·  "
        f":gray[**signals:** {meta['signals']}]"
    )
    st.markdown(badges)
    st.caption(meta["lens"])
    st.markdown("---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_parquet(name: str, file_mtime: float = 0.0) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _parquet_mtime(name: str) -> float:
    """Return file mtime so cache busts when data is refreshed."""
    p = DATA_DIR / name
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    p = CURATED_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_run_metadata() -> dict:
    p = DATA_DIR / "run_metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def no_data_banner():
    st.warning(
        "No cached data found in `./data`. Run `python collect.py` from the "
        "project root to populate snapshots, then refresh this page."
    )


# ---------------------------------------------------------------------------
# Filter helper functions
# ---------------------------------------------------------------------------

def apply_time_filter(df: pd.DataFrame, date_col: str, window_days: int) -> pd.DataFrame:
    """Filter DataFrame to rows within the time window."""
    if df.empty or date_col not in df.columns:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
    dt = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    return df[dt >= cutoff]


def drop_partial_weeks(df: pd.DataFrame, week_col: str = "week") -> pd.DataFrame:
    """Drop the last week bucket if it's incomplete (fewer than 7 days from today)."""
    if df.empty or week_col not in df.columns:
        return df
    max_week = df[week_col].max()
    if pd.isna(max_week):
        return df
    # A period("W") timestamp is the Monday of that week.
    # If today hasn't reached Sunday, the last week is partial — drop it.
    today = pd.Timestamp.now().normalize()
    week_end = pd.Timestamp(max_week) + pd.Timedelta(days=6)
    if today < week_end:
        return df[df[week_col] < max_week]
    return df


def apply_vendor_filter_source(df: pd.DataFrame, selected_vendors: list[str]) -> pd.DataFrame:
    """Filter vendor_df by its 'source' field (e.g. 'vendor:qualcomm')."""
    if df.empty or not selected_vendors:
        return df.iloc[0:0]
    lc = [v.lower() for v in selected_vendors]
    return df[df["source"].str.replace("vendor:", "", regex=False).isin(lc)]


def apply_vendor_filter(df: pd.DataFrame, vendor_col: str, selected_vendors: list[str]) -> pd.DataFrame:
    """Filter any DataFrame with a capitalized vendor column."""
    if df.empty or not selected_vendors or vendor_col not in df.columns:
        return df if not selected_vendors else df.iloc[0:0]
    return df[df[vendor_col].isin(selected_vendors)]


def apply_tier_filter(
    framework_df: pd.DataFrame,
    fw_counts_df: pd.DataFrame,
    selected_tiers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter framework data to selected deployment tiers."""
    if not selected_tiers:
        return framework_df.iloc[0:0], fw_counts_df.iloc[0:0]
    allowed = set()
    if not framework_df.empty and "framework_label" in framework_df.columns:
        for label in framework_df["framework_label"].unique():
            if analyze.deployment_tier(label) in selected_tiers:
                allowed.add(label)
    if not fw_counts_df.empty and "framework_label" in fw_counts_df.columns:
        for label in fw_counts_df["framework_label"].unique():
            if analyze.deployment_tier(label) in selected_tiers:
                allowed.add(label)
    f_fw = framework_df[framework_df["framework_label"].isin(allowed)] if not framework_df.empty else framework_df
    f_fc = fw_counts_df[fw_counts_df["framework_label"].isin(allowed)] if not fw_counts_df.empty else fw_counts_df
    return f_fw, f_fc


def apply_family_filter(
    vendor_df: pd.DataFrame,
    prod_long: pd.DataFrame,
    selected_vendors: list[str],
    selected_vendor_families: dict[str, list[str]],
    vendor_families_all: dict[str, dict],
) -> pd.DataFrame:
    """Narrow vendor_df to models matching selected silicon families.

    When every family for a vendor is checked, all that vendor's models pass
    through (including unclassified). When only some families are checked,
    only models that product_long classifies into those families are kept.
    """
    if vendor_df.empty or prod_long.empty:
        return vendor_df

    # Check if any vendor has a partial family selection
    needs_filter = False
    for vendor in selected_vendors:
        if vendor in selected_vendor_families and vendor in vendor_families_all:
            if set(selected_vendor_families[vendor]) != set(vendor_families_all[vendor].keys()):
                needs_filter = True
                break

    if not needs_filter:
        return vendor_df

    allowed_ids: set[str] = set()
    for vendor in selected_vendors:
        vendor_lc = vendor.lower()
        vendor_ids = set(
            vendor_df[
                vendor_df["source"].str.replace("vendor:", "", regex=False) == vendor_lc
            ]["repo_id"]
        )

        if vendor not in selected_vendor_families or vendor not in vendor_families_all:
            # No taxonomy — pass all through
            allowed_ids.update(vendor_ids)
        elif set(selected_vendor_families[vendor]) == set(vendor_families_all[vendor].keys()):
            # All families checked — pass all through
            allowed_ids.update(vendor_ids)
        else:
            # Partial selection — include models matching checked families
            # PLUS unclassified models (can't be meaningfully excluded)
            checked = set(selected_vendor_families[vendor])
            checked.add("(Unclassified)")
            mask = (prod_long["vendor"] == vendor) & (prod_long["family"].isin(checked))
            allowed_ids.update(prod_long.loc[mask, "repo_id"].unique())

    return vendor_df[vendor_df["repo_id"].isin(allowed_ids)]


def _set_keys(keys: list[str], val: bool):
    """Set multiple session-state keys and rerun."""
    for k in keys:
        st.session_state[k] = val
    st.rerun()


def _all_none_buttons(all_key: str, none_key: str, cb_keys: list[str]):
    """Render compact All · None gold links for a group of checkboxes."""
    c1, _, c2, _ = st.columns([1, 0.3, 1.2, 5])
    if c1.button("All", key=all_key):
        _set_keys(cb_keys, True)
    if c2.button("None", key=none_key):
        _set_keys(cb_keys, False)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

vendor_df = load_parquet("vendor_models.parquet", file_mtime=_parquet_mtime("vendor_models.parquet"))
framework_df = load_parquet("framework_models.parquet", file_mtime=_parquet_mtime("framework_models.parquet"))
spaces_df = load_parquet("spaces.parquet", file_mtime=_parquet_mtime("spaces.parquet"))
cards_df = load_parquet("model_cards.parquet", file_mtime=_parquet_mtime("model_cards.parquet"))
pypi_recent_df = load_parquet("pypi_recent.parquet", file_mtime=_parquet_mtime("pypi_recent.parquet"))
pypi_daily_df = load_parquet("pypi_downloads.parquet", file_mtime=_parquet_mtime("pypi_downloads.parquet"))
pypi_system_df = load_parquet("pypi_system.parquet", file_mtime=_parquet_mtime("pypi_system.parquet"))
fw_counts_df = load_parquet("framework_counts.parquet", file_mtime=_parquet_mtime("framework_counts.parquet"))
reddit_df = load_parquet("reddit_mentions.parquet", file_mtime=_parquet_mtime("reddit_mentions.parquet"))
gh_counts_df = load_parquet("github_mention_counts.parquet", file_mtime=_parquet_mtime("github_mention_counts.parquet"))
gh_detail_df = load_parquet("github_mentions.parquet", file_mtime=_parquet_mtime("github_mentions.parquet"))
gh_repo_stats_df = load_parquet("github_repo_stats.parquet", file_mtime=_parquet_mtime("github_repo_stats.parquet"))
gh_issue_topics_df = load_parquet("github_issue_topics.parquet", file_mtime=_parquet_mtime("github_issue_topics.parquet"))
gh_stargazers_df = load_parquet("github_stargazers.parquet", file_mtime=_parquet_mtime("github_stargazers.parquet"))
gh_forks_df = load_parquet("github_forks.parquet", file_mtime=_parquet_mtime("github_forks.parquet"))
playstore_df = load_parquet("playstore_apps.parquet", file_mtime=_parquet_mtime("playstore_apps.parquet"))
design_wins_df = load_csv("oem_design_wins.csv")
mlperf_df = load_parquet("mlperf_mobile.parquet", file_mtime=_parquet_mtime("mlperf_mobile.parquet"))
ai_bench_df = load_parquet("ai_benchmark_scores.parquet", file_mtime=_parquet_mtime("ai_benchmark_scores.parquet"))
geekbench_df = load_parquet("geekbench_ai.parquet", file_mtime=_parquet_mtime("geekbench_ai.parquet"))


@st.cache_data(show_spinner="Classifying models into products...")
def build_product_assignments(v_sig: int, c_sig: int) -> pd.DataFrame:
    return products_mod.assign_products(vendor_df, cards_df)


product_long = build_product_assignments(len(vendor_df), len(cards_df))

has_data = any(not df.empty for df in (vendor_df, framework_df, spaces_df))


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

# Build vendor hierarchy from products.py for the filter
CANONICAL_VENDORS = ["Qualcomm", "NVIDIA", "Apple", "Intel", "AMD", "Google", "Microsoft", "Arm", "Meta", "Arduino"]
# Vendors to hide from sidebar (platform / generic — not silicon vendors)
_HIDDEN_VENDORS = {"Generic", "HuggingFace", "community"}

# Normalize vendor names across datasets (before building the vendor list)
_VENDOR_RENAMES = {"Meta / cross": "Meta"}

# Remap edgeimpulse → Qualcomm in vendor_df source field
if not vendor_df.empty and "source" in vendor_df.columns:
    vendor_df["source"] = vendor_df["source"].str.replace(
        "vendor:edgeimpulse", "vendor:qualcomm", regex=False
    )

for _df in (pypi_daily_df, pypi_recent_df):
    if not _df.empty and "vendor" in _df.columns:
        _df["vendor"] = _df["vendor"].replace(_VENDOR_RENAMES)

# Collect vendors that appear in data
_data_vendors: set[str] = set()
if not product_long.empty:
    _data_vendors.update(product_long["vendor"].dropna().unique())
if not pypi_daily_df.empty and "vendor" in pypi_daily_df.columns:
    _data_vendors.update(pypi_daily_df["vendor"].dropna().unique())
if not reddit_df.empty and "vendor" in reddit_df.columns:
    _data_vendors.update(reddit_df["vendor"].dropna().unique())
if not gh_repo_stats_df.empty and "vendor" in gh_repo_stats_df.columns:
    _data_vendors.update(gh_repo_stats_df["vendor"].dropna().unique())
ALL_VENDORS = sorted(
    (v for v in ({v for v in CANONICAL_VENDORS if v in _data_vendors} | _data_vendors)
     if v not in _HIDDEN_VENDORS)
)

# Build family/product hierarchy from ALL_RULES
_vendor_families: dict[str, dict[str, list[str]]] = {}
for rule in products_mod.ALL_RULES:
    vend = rule.vendor
    fam = rule.family
    prod = rule.product
    _vendor_families.setdefault(vend, {}).setdefault(fam, [])
    if prod not in _vendor_families[vend][fam]:
        _vendor_families[vend][fam].append(prod)

TIME_OPTIONS = {
    "1 week": 7,
    "2 weeks": 14,
    "1 month": 30,
    "2 months": 60,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
}

MODEL_TYPE_OPTIONS = [
    "text-generation",
    "image-text-to-text",
    "image-classification",
    "automatic-speech-recognition",
    "text-to-image",
    "sentence-similarity",
    "feature-extraction",
    "text-classification",
    "object-detection",
    "image-segmentation",
    "text-to-speech",
    "token-classification",
]

# ---------------------------------------------------------------------------
# Page header (replaces sidebar)
# ---------------------------------------------------------------------------
_hdr_left, _hdr_right = st.columns([3, 1])
with _hdr_left:
    st.markdown("## Developer Ecosystem")
    st.caption("On-device AI Market Intelligence")
    meta = load_run_metadata()
    _last_run = meta.get("run_finished_utc", "unknown") if meta else "unknown"
    st.caption(f"Last refresh: {_last_run}")
with _hdr_right:
    if st.button("Refresh data", key="refresh_btn"):
        import subprocess
        with st.spinner("Running collectors..."):
            subprocess.run(
                ["python", "collect.py"],
                cwd=str(DATA_DIR.parent),
                capture_output=True,
            )
            subprocess.run(
                ["python", "collect_pypi.py"],
                cwd=str(DATA_DIR.parent),
                capture_output=True,
            )
            subprocess.run(
                ["python", "collect_reddit.py"],
                cwd=str(DATA_DIR.parent),
                capture_output=True,
            )
            subprocess.run(
                ["python", "collect_github_repos.py"],
                cwd=str(DATA_DIR.parent),
                capture_output=True,
            )
            st.cache_data.clear()
            st.rerun()


if not has_data:
    no_data_banner()
    st.stop()

# Default time window for sections without local time controls
window_days = 90
time_label = "3 months"

# ===========================================================================
# 1 · SUPPLY — publishing, runtimes, deployment tier, products
# ===========================================================================

with st.expander("1 — Supply / Publishing", expanded=True):
    stage_header("stage2")

    # ── Publishing Velocity ──────────────────────────────────────────
    with st.expander("Publishing Velocity", expanded=True):
        # --- Local filters ---
        _pv_all_vendors = sorted(
            vendor_df["source"].str.replace("vendor:", "", regex=False).unique()
        ) if not vendor_df.empty else []
        _pv_c1, _pv_c2 = st.columns([1, 3])
        with _pv_c1:
            _pv_time = st.select_slider("Time window", options=list(TIME_OPTIONS.keys()), value="3 months", key="pv_time")
            _pv_window = TIME_OPTIONS[_pv_time]
        with _pv_c2:
            _pv_vendors = st.multiselect("Vendors", options=_pv_all_vendors, default=_pv_all_vendors, key="pv_vendors")
        _pv_c3, _pv_c4 = st.columns(2)
        with _pv_c3:
            _pv_tiers = st.multiselect("Tier", options=["On-device", "Edge / hybrid", "Datacenter"], default=["On-device", "Edge / hybrid", "Datacenter"], key="pv_tiers")
        with _pv_c4:
            _pv_types = st.multiselect("Model type", options=MODEL_TYPE_OPTIONS, default=MODEL_TYPE_OPTIONS, key="pv_types")

        # --- Apply filters ---
        _pv_df = apply_vendor_filter_source(vendor_df, _pv_vendors)
        _pv_df = apply_family_filter(_pv_df, product_long, _pv_vendors, {}, _vendor_families)
        _pv_mt_active = len(_pv_types) < len(MODEL_TYPE_OPTIONS)
        if _pv_mt_active and _pv_types:
            if not _pv_df.empty and "pipeline_tag" in _pv_df.columns:
                _pv_df = _pv_df[_pv_df["pipeline_tag"].isin(_pv_types)]
        elif _pv_mt_active and not _pv_types:
            _pv_df = _pv_df.iloc[0:0]

        st.markdown(f"#### Publishing velocity — last {_pv_window} days")
        vel2 = analyze.vendor_velocity(_pv_df, window_days=_pv_window)
        if vel2.empty:
            st.info("No vendor velocity data for selected vendors.")
        else:
            fig_p = px.bar(
                vel2,
                x="vendor",
                y=f"new_in_{_pv_window}d",
                text=f"new_in_{_pv_window}d",
                title=f"New models by vendor org in last {_pv_window} days",
                color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_p.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="New models"
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("#### Publishing by deployment tier")
        vt = analyze.vendor_tier_breakdown(
            apply_time_filter(_pv_df, "created_at", _pv_window),
            product_long,
        )
        if not vt.empty:
            fig_vt = px.bar(
                vt,
                x="vendor",
                y="models",
                color="tier",
                text="models",
                title=f"New models by vendor × deployment tier (last {_pv_window} days)",
                color_discrete_map=TIER_COLOR,
                barmode="stack",
                category_orders={"tier": ["On-device", "Edge / hybrid", "Datacenter", "Unclassified"]},
            )
            fig_vt.update_traces(textposition="inside")
            fig_vt.update_layout(
                xaxis_title="", yaxis_title="New models",
                legend_title_text="Tier",
            )
            st.plotly_chart(fig_vt, use_container_width=True)
            st.caption(
                "Tier is inferred from product-family classification "
                "(e.g. Snapdragon Mobile → On-device, NVIDIA Data Center → Datacenter). "
                "Unclassified models use a vendor-level default."
            )
        else:
            st.info("No publishing data in this time window.")

        st.markdown("#### Weekly publishing timeline")
        tl = analyze.vendor_publish_timeline(_pv_df, freq="W")
        if not tl.empty:
            # Apply time filter to timeline
            tl["period"] = pd.to_datetime(tl["period"])
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_pv_window)
            tl = tl[tl["period"] >= cutoff.tz_localize(None)]
            if not tl.empty:
                fig_tl = px.line(
                    tl,
                    x="period",
                    y="new_models",
                    color="vendor",
                    title="Weekly new-model publishing by vendor org",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_tl.update_layout(xaxis_title="Week", yaxis_title="New models")
                st.plotly_chart(fig_tl, use_container_width=True)

    # ── Runtimes ─────────────────────────────────────────────────────
    with st.expander("Runtimes", expanded=True):
        st.markdown("#### Share of runtime — model counts by framework tag")
        fw_summ = analyze.framework_summary(framework_df, fw_counts_df)
        if fw_summ.empty:
            st.info("No framework data for selected tiers.")
        else:
            fig_fw = px.bar(
                fw_summ,
                x="framework_label",
                y="models",
                text="models",
                color="framework_label",
                color_discrete_map=FRAMEWORK_COLOR,
                title="Models tagged per runtime (HF Hub, current snapshot)",
            )
            fig_fw.update_traces(textposition="outside")
            fig_fw.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Models"
            )
            st.plotly_chart(fig_fw, use_container_width=True)

            st.markdown("#### Downloads by runtime")
            fig_fw2 = px.bar(
                fw_summ,
                x="framework_label",
                y="downloads",
                text="downloads",
                color="framework_label",
                color_discrete_map=FRAMEWORK_COLOR,
                title="Cumulative downloads per runtime",
            )
            fig_fw2.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Downloads"
            )
            st.plotly_chart(fig_fw2, use_container_width=True)

            st.markdown("#### Framework velocity")
            fw_vel = analyze.framework_velocity(framework_df, window_days=window_days)
            if not fw_vel.empty:
                fig_fwv = px.bar(
                    fw_vel,
                    x="framework_label",
                    y=f"new_in_{window_days}d",
                    text=f"new_in_{window_days}d",
                    color="framework_label",
                    color_discrete_map=FRAMEWORK_COLOR,
                    title=f"New models per runtime in last {window_days} days",
                )
                fig_fwv.update_layout(
                    showlegend=False, xaxis_title="", yaxis_title="New models"
                )
                st.plotly_chart(fig_fwv, use_container_width=True)

        st.markdown("#### Hardware keyword mining (tags + repo names)")
        combined = pd.concat([vendor_df, framework_df], ignore_index=True) \
            if not (vendor_df.empty and framework_df.empty) else pd.DataFrame()
        kw = analyze.tag_keyword_counts(combined)
        if not kw.empty:
            fig_kw = px.bar(
                kw,
                x="bucket",
                y="models",
                text="models",
                title="Models mentioning hardware ecosystem keywords",
                color="bucket",
                color_discrete_map=BUCKET_COLOR,
            )
            fig_kw.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Models"
            )
            st.plotly_chart(fig_kw, use_container_width=True)

    # ── Deployment tier ──────────────────────────────────────────────
    with st.expander("Deployment tier", expanded=False):
        st.markdown("#### HF runtimes mapped to deployment tiers")
        st.caption(
            "Each runtime tag maps to a deployment tier based on where the "
            "compiled artifact is designed to run. **On-device** = the "
            "model's output format runs directly on an accelerator inside "
            "a phone / laptop / edge device (Core ML, MLX, LiteRT, "
            "ExecuTorch, QNN). **Edge / hybrid** = cross-platform runtimes "
            "that commonly hit NPUs via delegates (ONNX, GGUF, OpenVINO). "
            "**Datacenter** = server GPU (CUDA / TensorRT)."
        )
        tier_df2 = analyze.classify_deployment(framework_df, fw_counts_df)
        tier_tot = analyze.tier_totals(framework_df, fw_counts_df)
        if tier_df2.empty:
            st.info("No framework data for selected tiers.")
        else:
            k1, k2, k3 = st.columns(3)
            on_dev = int(tier_tot.loc[tier_tot["tier"] == "On-device", "models"].sum())
            edge = int(tier_tot.loc[tier_tot["tier"] == "Edge / hybrid", "models"].sum())
            dc = int(tier_tot.loc[tier_tot["tier"] == "Datacenter", "models"].sum())
            k1.metric("On-device models", f"{on_dev:,}")
            k2.metric("Edge / hybrid models", f"{edge:,}")
            k3.metric("Datacenter models", f"{dc:,}")

            st.markdown("##### Runtimes inside each tier")
            fig_t = px.bar(
                tier_df2,
                x="framework_label",
                y="models",
                color="tier",
                text="models",
                title="HF model count — runtime broken down by deployment tier",
                color_discrete_map=TIER_COLOR,
            )
            fig_t.update_layout(xaxis_title="", yaxis_title="Models")
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("##### Tier rollup")
            st.dataframe(tier_tot, use_container_width=True, hide_index=True)

    # ── Model types ──────────────────────────────────────────────────
    with st.expander("Model types", expanded=False):
        st.markdown("#### Vendor × model type — supply vs demand")
        st.caption(
            "Model counts show what vendors **publish** (supply). Downloads "
            "show what developers actually **use** (demand). The gap analysis "
            "combines both: a high-download type where Qualcomm has zero models "
            "is an addressable opportunity. This view always shows all types "
            "for comparison."
        )

        # --- Local vendor filter ---
        _mt_all_vendors = sorted(
            vendor_df["source"].str.replace("vendor:", "", regex=False).unique()
        ) if not vendor_df.empty else []
        _mt_vendors = st.multiselect("Vendors", options=_mt_all_vendors, default=_mt_all_vendors, key="mt_vendors")
        _mt_vdf = apply_vendor_filter_source(vendor_df, _mt_vendors)
        _mt_vdf = apply_family_filter(
            _mt_vdf, product_long, _mt_vendors,
            {}, _vendor_families,
        )

        if _mt_vdf.empty or "pipeline_tag" not in _mt_vdf.columns:
            st.info("No vendor model data available.")
        else:
            _mt = _mt_vdf.dropna(subset=["pipeline_tag"]).copy()
            _mt["vendor"] = _mt["source"].str.replace("vendor:", "", regex=False)

            if _mt.empty:
                st.info("No models have pipeline_tag labels.")
            else:
                # ── Demand: total downloads by model type ────────────
                st.markdown("#### Market demand — downloads by model type")
                st.caption(
                    "Total downloads across all selected vendors' models, "
                    "ranked by type. This is the demand signal — which tasks "
                    "developers actually pull models for."
                )
                type_demand = (
                    _mt.groupby("pipeline_tag")
                    .agg(
                        models=("repo_id", "count"),
                        total_downloads=("downloads", "sum"),
                    )
                    .reset_index()
                    .sort_values("total_downloads", ascending=False)
                )
                type_demand["downloads_per_model"] = (
                    type_demand["total_downloads"] / type_demand["models"]
                ).astype(int)

                top_by_dl = type_demand.head(15)

                fig_demand = px.bar(
                    top_by_dl,
                    x="pipeline_tag",
                    y="total_downloads",
                    text="total_downloads",
                    title="Total downloads by model type (top 15)",
                    hover_data=["models", "downloads_per_model"],
                )
                fig_demand.update_traces(
                    texttemplate="%{text:,.0f}", textposition="outside",
                    marker_color=QUALCOMM_BLUE,
                )
                fig_demand.update_layout(
                    xaxis_title="", yaxis_title="Downloads",
                    xaxis_tickangle=-45, height=450,
                )
                st.plotly_chart(fig_demand, use_container_width=True)

                # Usage intensity: downloads per model
                st.markdown("#### Usage intensity — downloads per model")
                st.caption(
                    "High downloads-per-model = strong demand relative to supply. "
                    "Types with few models but many downloads are under-served."
                )
                top_intensity = type_demand.sort_values(
                    "downloads_per_model", ascending=False
                ).head(15)
                fig_intensity = px.bar(
                    top_intensity,
                    x="pipeline_tag",
                    y="downloads_per_model",
                    text="downloads_per_model",
                    title="Downloads per model (usage intensity, top 15)",
                    hover_data=["models", "total_downloads"],
                )
                fig_intensity.update_traces(
                    texttemplate="%{text:,.0f}", textposition="outside",
                    marker_color="#b8852a",
                )
                fig_intensity.update_layout(
                    xaxis_title="", yaxis_title="Downloads / model",
                    xaxis_tickangle=-45, height=450,
                )
                st.plotly_chart(fig_intensity, use_container_width=True)

                # ── Supply: vendor × type grouped bar ────────────────
                st.markdown("#### Supply — model count by type × vendor")

                ct = pd.crosstab(_mt["vendor"], _mt["pipeline_tag"])
                top_types = ct.sum().sort_values(ascending=False).head(15).index.tolist()
                ct_top = ct[top_types]

                ct_melted = ct_top.reset_index().melt(
                    id_vars="vendor", var_name="model_type", value_name="models",
                )
                fig_mt = px.bar(
                    ct_melted,
                    x="model_type",
                    y="models",
                    color="vendor",
                    barmode="group",
                    text="models",
                    title="Model count by type × vendor (top 15 types)",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_mt.update_traces(textposition="outside", textfont_size=9)
                fig_mt.update_layout(
                    xaxis_title="", yaxis_title="Models",
                    legend_title_text="Vendor",
                    xaxis_tickangle=-45, height=500,
                )
                st.plotly_chart(fig_mt, use_container_width=True)

                # ── Downloads by vendor × type ───────────────────────
                st.markdown("#### Demand — downloads by type × vendor")
                dl_pivot = (
                    _mt.groupby(["vendor", "pipeline_tag"])["downloads"]
                    .sum()
                    .reset_index()
                )
                dl_pivot = dl_pivot[dl_pivot["pipeline_tag"].isin(top_types)]
                fig_dl_v = px.bar(
                    dl_pivot,
                    x="pipeline_tag",
                    y="downloads",
                    color="vendor",
                    barmode="group",
                    title="Downloads by type × vendor (top 15 types, log scale)",
                    color_discrete_map=VENDOR_COLOR,
                    log_y=True,
                )
                fig_dl_v.update_layout(
                    xaxis_title="", yaxis_title="Downloads (log)",
                    legend_title_text="Vendor",
                    xaxis_tickangle=-45, height=500,
                )
                st.plotly_chart(fig_dl_v, use_container_width=True)

                # ── Gap analysis: supply + demand combined ───────────
                st.markdown("#### Qualcomm gap analysis — supply × demand")
                st.caption(
                    "Combines model count (supply) and downloads (demand). "
                    "A GAP in a high-download type is the strongest signal of "
                    "an addressable opportunity."
                )

                if "qualcomm" in ct.index:
                    qc_counts = ct.loc["qualcomm"]
                    peer_counts = ct.drop("qualcomm", errors="ignore").sum()
                    n_peers = max(len(ct) - 1, 1)

                    qc_dl = _mt[_mt["vendor"] == "qualcomm"].groupby("pipeline_tag")["downloads"].sum()
                    peer_dl = _mt[_mt["vendor"] != "qualcomm"].groupby("pipeline_tag")["downloads"].sum()

                    # Use all types that have either supply or demand, not just top 15
                    all_types = sorted(
                        set(ct.columns.tolist()) | set(qc_dl.index.tolist()) | set(peer_dl.index.tolist())
                    )
                    gap_rows = []
                    for mtype in all_types:
                        qc_n = int(qc_counts.get(mtype, 0))
                        peer_n = int(peer_counts.get(mtype, 0))
                        qc_d = int(qc_dl.get(mtype, 0))
                        peer_d = int(peer_dl.get(mtype, 0))
                        total_d = qc_d + peer_d

                        if qc_n > 0 and qc_n >= peer_n / n_peers:
                            status = "LEAD"
                        elif qc_n > 0:
                            status = "COVERED"
                        else:
                            status = "GAP"

                        gap_rows.append({
                            "model_type": mtype,
                            "qc_models": qc_n,
                            "peer_models": peer_n,
                            "qc_downloads": qc_d,
                            "peer_downloads": peer_d,
                            "total_downloads": total_d,
                            "status": status,
                        })

                    gap_df = pd.DataFrame(gap_rows).sort_values(
                        "total_downloads", ascending=False
                    )

                    st.dataframe(
                        gap_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "qc_downloads": st.column_config.NumberColumn(
                                "QC downloads", format="%,d",
                            ),
                            "peer_downloads": st.column_config.NumberColumn(
                                "Peer downloads", format="%,d",
                            ),
                            "total_downloads": st.column_config.NumberColumn(
                                "Total downloads", format="%,d",
                            ),
                            "status": st.column_config.TextColumn(
                                "Status",
                                help="LEAD = QC above peer avg, COVERED = present, GAP = zero QC models",
                            ),
                        },
                    )

                    n_lead = sum(1 for r in gap_rows if r["status"] == "LEAD")
                    n_covered = sum(1 for r in gap_rows if r["status"] == "COVERED")
                    n_gap = sum(1 for r in gap_rows if r["status"] == "GAP")

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Leading types", f"{n_lead:,}")
                    k2.metric("Covered types", f"{n_covered:,}")
                    k3.metric("Gap types (opportunities)", f"{n_gap:,}")

                    # Highlight top opportunity gaps
                    gap_opps = [r for r in gap_rows if r["status"] == "GAP"]
                    gap_opps.sort(key=lambda r: r["total_downloads"], reverse=True)
                    if gap_opps:
                        st.markdown("##### Top opportunity gaps (by download demand)")
                        for r in gap_opps[:5]:
                            st.markdown(
                                f"- **{r['model_type']}** — "
                                f"{r['total_downloads']:,} downloads across "
                                f"{r['peer_models']} peer models, "
                                f"0 from Qualcomm"
                            )
                else:
                    st.info("Qualcomm not in selected vendors — gap analysis unavailable.")

                # ── Qualcomm QAI Hub vs HF ecosystem ──────────────
                st.markdown("#### Qualcomm QAI Hub portfolio vs HF ecosystem")
                st.caption(
                    "Compares models Qualcomm publishes directly (QAI Hub / HF qualcomm org) "
                    "against what the broader HF ecosystem supplies per model type. "
                    "Types where Qualcomm self-supplies heavily signal strategic focus; "
                    "types where HF ecosystem dominates may indicate reliance on community."
                )

                _qc_models = _mt[_mt["vendor"] == "qualcomm"].copy()
                _hf_others = _mt[_mt["vendor"] != "qualcomm"].copy()

                if not _qc_models.empty:
                    # Count by type for Qualcomm vs rest-of-HF
                    _qc_type = (
                        _qc_models.groupby("pipeline_tag")
                        .agg(qc_models=("repo_id", "count"))
                        .reset_index()
                    )
                    _hf_type = (
                        _hf_others.groupby("pipeline_tag")
                        .agg(hf_models=("repo_id", "count"))
                        .reset_index()
                    )
                    _compare = _qc_type.merge(_hf_type, on="pipeline_tag", how="outer").fillna(0)
                    for c in ["qc_models", "hf_models"]:
                        _compare[c] = _compare[c].astype(int)
                    _compare = _compare.sort_values("qc_models", ascending=False)

                    # Bar chart: Qualcomm model count vs HF ecosystem, by type
                    _cmp_top = _compare[(_compare["qc_models"] > 0) | (_compare["hf_models"] > 0)].head(20)
                    _cmp_melt = _cmp_top.melt(
                        id_vars="pipeline_tag",
                        value_vars=["qc_models", "hf_models"],
                        var_name="source",
                        value_name="models",
                    )
                    _cmp_melt["source"] = _cmp_melt["source"].map({
                        "qc_models": "Qualcomm (QAI Hub)",
                        "hf_models": "Other HF vendors",
                    })
                    fig_qc_hf = px.bar(
                        _cmp_melt,
                        x="pipeline_tag",
                        y="models",
                        color="source",
                        barmode="group",
                        text="models",
                        color_discrete_map={
                            "Qualcomm (QAI Hub)": QUALCOMM_BLUE,
                            "Other HF vendors": _AMBIGUOUS_GREY,
                        },
                    )
                    fig_qc_hf.update_traces(textposition="outside", textfont_size=9)
                    fig_qc_hf.update_layout(
                        xaxis_title="", yaxis_title="Models",
                        legend_title_text="",
                        xaxis_tickangle=-45, height=500,
                    )
                    st.plotly_chart(fig_qc_hf, use_container_width=True)

                    # Table with full detail
                    _compare["qc_share"] = (
                        _compare["qc_models"] / (_compare["qc_models"] + _compare["hf_models"]) * 100
                    ).round(1)
                    _compare_display = _compare.rename(columns={
                        "pipeline_tag": "Model type",
                        "qc_models": "QC models",
                        "hf_models": "HF ecosystem models",
                        "qc_share": "QC supply share %",
                    })
                    st.dataframe(
                        _compare_display,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Qualcomm not in selected vendors.")

                # Full cross-tab
                st.markdown("#### Full cross-tab (model counts)")
                st.dataframe(ct_top, use_container_width=True)


# ===========================================================================
# 2 · DEVELOPER INTENT — PyPI, Reddit, GitHub issues, GitHub repos
# ===========================================================================

with st.expander("2 — Developer Intent", expanded=True):
    stage_header("stage3")

    # ── PyPI downloads ───────────────────────────────────────────────
    with st.expander("PyPI downloads", expanded=True):
        if pypi_daily_df.empty:
            st.info("No PyPI data for selected vendors.")
        else:
            _daily = pypi_daily_df.copy()
            _daily["date"] = pd.to_datetime(_daily["date"])

            # Apply time window
            _cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
            _daily_windowed = _daily[_daily["date"] >= _cutoff.tz_localize(None)]

            max_date = _daily["date"].max()
            # Use window_days for growth comparison
            last_start = max_date - pd.Timedelta(days=window_days - 1)
            prior_end = max_date - pd.Timedelta(days=window_days)
            prior_start = max_date - pd.Timedelta(days=2 * window_days - 1)

            QCOM_PKGS = ["qai-hub", "qai-hub-models"]

            def _win_sum(packages: list[str], start, end) -> int:
                m = (
                    _daily["package"].isin(packages)
                    & (_daily["date"] >= start)
                    & (_daily["date"] <= end)
                )
                return int(_daily.loc[m, "downloads"].sum())

            def _growth_pct(last: int, prior: int) -> float | None:
                if prior <= 0:
                    return None
                return (last - prior) / prior * 100

            qcom_last = _win_sum(QCOM_PKGS, last_start, max_date)
            qcom_prior = _win_sum(QCOM_PKGS, prior_start, prior_end)
            qcom_growth = _growth_pct(qcom_last, qcom_prior)

            mask_last = (_daily["date"] >= last_start) & (_daily["date"] <= max_date)
            vendor_window = (
                _daily.loc[mask_last]
                .groupby("vendor", as_index=False)["downloads"]
                .sum()
                .sort_values("downloads", ascending=False)
                .reset_index(drop=True)
            )
            qcom_rank_row = vendor_window[vendor_window["vendor"] == "Qualcomm"]
            qcom_rank = int(qcom_rank_row.index[0]) + 1 if not qcom_rank_row.empty else None

            aet_last = _win_sum(["ai-edge-torch"], last_start, max_date)
            vs_aet = (qcom_last / aet_last) if aet_last else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                f"Qualcomm SDK downloads (last {window_days}d)",
                f"{qcom_last:,}",
            )
            k2.metric(
                f"Growth vs prior {window_days}d",
                f"{qcom_growth:+.0f}%" if qcom_growth is not None else "—",
            )
            k3.metric(
                f"Vendor rank (last {window_days}d)",
                f"#{qcom_rank} / {len(vendor_window)}" if qcom_rank else "—",
            )
            k4.metric(
                "vs ai-edge-torch (Google)",
                f"{vs_aet:.1f}x" if vs_aet else "—",
            )

            st.markdown(f"#### Vendor SDK intent — weekly downloads by vendor (last {time_label})")
            if not _daily_windowed.empty:
                _vt = _daily_windowed.copy()
                _vt_all_vendors = sorted(_vt["vendor"].unique())
                _vt_vendors = st.multiselect(
                    "Vendors", options=_vt_all_vendors, default=_vt_all_vendors, key="pypi_vendor_ts_vendors",
                )
                _vt = _vt[_vt["vendor"].isin(_vt_vendors)]
                _vt["week"] = _vt["date"].dt.to_period("W").dt.to_timestamp()
                _vt = drop_partial_weeks(_vt)
                vendor_ts = (
                    _vt.groupby(["week", "vendor"], as_index=False)["downloads"]
                    .sum()
                )
                fig_vts = px.line(
                    vendor_ts, x="week", y="downloads", color="vendor",
                    color_discrete_map=VENDOR_COLOR, log_y=True,
                )
                fig_vts.update_layout(
                    xaxis_title="", yaxis_title="Downloads / week (log)",
                    legend_title_text="Vendor",
                )
                st.plotly_chart(fig_vts, use_container_width=True)

            st.markdown(f"##### Vendor SDK intent — total downloads, last {window_days} days")
            if not vendor_window.empty:
                fig_v = px.bar(
                    vendor_window,
                    x="vendor",
                    y="downloads",
                    text="downloads",
                    log_y=True,
                    color="vendor",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_v.update_traces(texttemplate="%{text:,}", textposition="outside")
                fig_v.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title=f"Downloads — last {window_days}d (log scale)",
                )
                st.plotly_chart(fig_v, use_container_width=True)

            st.markdown("#### Direct-analog comparison — daily downloads (7-day rolling mean)")
            DEFAULT_COMPARE = [
                "qai-hub", "qai-hub-models", "coremltools",
                "openvino", "ai-edge-torch", "executorch", "onnxruntime-directml",
            ]
            all_pkgs = sorted(_daily_windowed["package"].unique())
            selected_pkgs = st.multiselect(
                "Packages to compare",
                options=all_pkgs,
                default=[p for p in DEFAULT_COMPARE if p in all_pkgs],
                key="pypi_compare",
            )
            if selected_pkgs:
                sel = _daily_windowed[_daily_windowed["package"].isin(selected_pkgs)].copy()
                sel = sel.sort_values(["package", "date"])
                sel["rolling_7d"] = (
                    sel.groupby("package")["downloads"]
                    .transform(lambda s: s.rolling(7, min_periods=1).mean())
                )
                fig_t = px.line(
                    sel,
                    x="date",
                    y="rolling_7d",
                    color="package",
                    log_y=True,
                    color_discrete_map=PACKAGE_COLOR,
                )
                fig_t.update_layout(
                    xaxis_title="",
                    yaxis_title="Downloads / day — 7d avg (log)",
                    legend_title_text="Package",
                )
                st.plotly_chart(fig_t, use_container_width=True)

            st.markdown(f"#### Qualcomm vs direct analogs — {window_days}d vs prior {window_days}d")
            ANALOGS = [
                "qai-hub", "qai-hub-models", "coremltools", "openvino",
                "ai-edge-torch", "executorch", "onnxruntime-directml", "mlx", "mlx-lm",
            ]
            analog_rows: list[dict] = []
            pkg_to_vendor = dict(zip(pypi_recent_df["package"], pypi_recent_df["vendor"]))
            for pkg in ANALOGS:
                if pkg not in pkg_to_vendor:
                    continue
                last = _win_sum([pkg], last_start, max_date)
                prior = _win_sum([pkg], prior_start, prior_end)
                growth = _growth_pct(last, prior)
                analog_rows.append({
                    "package": pkg,
                    "vendor": pkg_to_vendor[pkg],
                    f"last_{window_days}d": last,
                    f"prior_{window_days}d": prior,
                    "growth_pct": round(growth, 1) if growth is not None else None,
                    "ratio_to_qai_hub_combined": (
                        round(last / qcom_last, 2) if qcom_last else None
                    ),
                })
            if analog_rows:
                st.dataframe(
                    pd.DataFrame(analog_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("#### All tracked packages")
            if not pypi_recent_df.empty:
                show = pypi_recent_df[
                    ["package", "vendor", "product", "last_day", "last_week", "last_month", "note"]
                ].copy()
                show = show.sort_values("last_month", ascending=False).reset_index(drop=True)
                st.dataframe(show, use_container_width=True, hide_index=True)

    # ── PyPI developer profile (OS breakdown) ───────────────────────
    with st.expander("Developer profile (OS breakdown)", expanded=False):
        st.markdown("#### Who's installing? — Linux vs Windows vs macOS")
        st.caption(
            "These are **developer workstation / CI installs** — where models get *built*, "
            "not where they *run*. Linux-heavy = CI/CD pipelines and enterprise build servers. "
            "macOS/Windows-heavy = individual developers on local machines. "
            "All packages tracked are AI/ML toolchain SDKs. Use the affinity filter below to "
            "isolate on-device-specific tools (native) vs cross-platform SDKs (abstracted). "
            "Source: pypistats.org /system endpoint."
        )
        if pypi_system_df.empty:
            st.info("No PyPI system data. Run `python collect_pypi.py --system-only` to collect.")
        else:
            _sys = pypi_system_df.copy()
            _sys["date"] = pd.to_datetime(_sys["date"], errors="coerce")

            # Affinity filter
            _sys_affinities = sorted(_sys["affinity"].dropna().unique()) if "affinity" in _sys.columns else []
            if _sys_affinities:
                _sys_aff = st.multiselect(
                    "Hardware affinity",
                    options=_sys_affinities,
                    default=_sys_affinities,
                    key="sys_affinity",
                    help="native = on-device-specific SDKs, abstracted = cross-platform, mixed = both",
                )
                _sys = _sys[_sys["affinity"].isin(_sys_aff)]

            # Normalize OS names
            _sys["os"] = _sys["category"].fillna("unknown").replace({
                "Darwin": "macOS", "null": "unknown",
            })

            # Aggregate last 30 days per vendor + OS
            _sys_cutoff = _sys["date"].max() - pd.Timedelta(days=30)
            _sys_recent = _sys[_sys["date"] >= _sys_cutoff]

            # --- Per-vendor OS split (stacked bar) ---
            _vendor_os = (
                _sys_recent.groupby(["vendor", "os"], as_index=False)["downloads"]
                .sum()
            )
            # Calculate percentages
            _vendor_totals = _vendor_os.groupby("vendor")["downloads"].transform("sum")
            _vendor_os["pct"] = (_vendor_os["downloads"] / _vendor_totals * 100).round(1)

            _os_colors = {"Linux": "#F4A460", "Windows": "#4682B4", "macOS": "#A9A9A9", "other": "#D3D3D3", "unknown": "#EEEEEE"}

            fig_os = px.bar(
                _vendor_os, x="vendor", y="downloads", color="os",
                color_discrete_map=_os_colors,
                barmode="stack",
                text="pct",
            )
            fig_os.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
            fig_os.update_layout(
                xaxis_title="", yaxis_title="Downloads (last 30d)",
                legend_title_text="OS", yaxis_type="log",
            )
            st.plotly_chart(fig_os, use_container_width=True)

            # --- Linux % summary table ---
            st.markdown("##### Linux share by vendor (proxy for enterprise/CI adoption)")
            _linux_pct = (
                _vendor_os.pivot_table(index="vendor", columns="os", values="downloads", aggfunc="sum", fill_value=0)
                .reset_index()
            )
            _os_cols = [c for c in ["Linux", "Windows", "macOS", "other", "unknown"] if c in _linux_pct.columns]
            _linux_pct["total"] = _linux_pct[_os_cols].sum(axis=1)
            if "Linux" in _linux_pct.columns:
                _linux_pct["linux_pct"] = (_linux_pct["Linux"] / _linux_pct["total"] * 100).round(1)
            else:
                _linux_pct["linux_pct"] = 0.0
            _linux_pct = _linux_pct.sort_values("linux_pct", ascending=False)
            _show_cols = ["vendor", "total"] + _os_cols + ["linux_pct"]
            _show_cols = [c for c in _show_cols if c in _linux_pct.columns]
            st.dataframe(
                _linux_pct[_show_cols].rename(columns={
                    "vendor": "Vendor", "total": "Total (30d)", "linux_pct": "Linux %",
                }),
                use_container_width=True, hide_index=True,
            )

            # --- Per-package breakdown (top packages) ---
            st.markdown("##### Per-package OS split (top 15 by volume)")
            _pkg_os = (
                _sys_recent.groupby(["package", "vendor", "os"], as_index=False)["downloads"]
                .sum()
            )
            _pkg_totals = _pkg_os.groupby("package")["downloads"].transform("sum")
            _pkg_os["pct"] = (_pkg_os["downloads"] / _pkg_totals * 100).round(1)
            _top_pkgs = (
                _pkg_os.groupby("package")["downloads"].sum()
                .nlargest(15).index.tolist()
            )
            _pkg_os_top = _pkg_os[_pkg_os["package"].isin(_top_pkgs)]
            fig_pkg_os = px.bar(
                _pkg_os_top, x="package", y="downloads", color="os",
                color_discrete_map=_os_colors,
                barmode="stack",
                text="pct",
            )
            fig_pkg_os.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
            fig_pkg_os.update_layout(
                xaxis_title="", yaxis_title="Downloads (last 30d, log)",
                legend_title_text="OS", yaxis_type="log",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_pkg_os, use_container_width=True)

    # ── Reddit mentions ──────────────────────────────────────────────
    with st.expander("Reddit mentions", expanded=True):
        st.markdown("#### Reddit — vendor share of voice in on-device AI")
        st.caption(
            "r/LocalLLaMA is the largest community of people running LLMs "
            "on local hardware. When someone posts about Snapdragon there, "
            "it's an organic mindshare signal — unprompted, timestamped, and "
            "scored by the community."
        )

        if reddit_df.empty:
            st.info("No Reddit data for selected vendors.")
        else:
            rdf = reddit_df.copy()
            rdf["created_utc"] = pd.to_datetime(
                rdf["created_utc"], utc=True, errors="coerce"
            )
            # Apply time window
            _cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
            rdf = rdf[rdf["created_utc"] >= _cutoff]

            vendor_roll = (
                rdf.drop_duplicates(subset=["post_id", "vendor"])
                .groupby("vendor", as_index=False)
                .agg(
                    posts=("post_id", "nunique"),
                    total_score=("score", "sum"),
                    total_comments=("num_comments", "sum"),
                )
                .sort_values("posts", ascending=False)
            )

            k1, k2, k3 = st.columns(3)
            k1.metric("Unique posts", f"{rdf['post_id'].nunique():,}")
            k2.metric("Subreddits searched", f"{rdf['subreddit'].nunique():,}")
            qcom_posts = int(
                vendor_roll.loc[vendor_roll["vendor"] == "Qualcomm", "posts"].sum()
            )
            k3.metric("Qualcomm-mention posts", f"{qcom_posts:,}")

            # --- Weekly time-series: posts per week by vendor ---
            st.markdown(f"##### Posts per week by vendor (last {time_label})")
            _rdf_ts = rdf.drop_duplicates(subset=["post_id", "vendor"]).copy()
            _rdf_ts["week"] = _rdf_ts["created_utc"].dt.to_period("W").dt.to_timestamp()
            _rdf_ts = drop_partial_weeks(_rdf_ts)
            reddit_ts = _rdf_ts.groupby(["week", "vendor"], as_index=False).agg(
                posts=("post_id", "nunique"),
            )
            fig_rt = px.line(
                reddit_ts, x="week", y="posts", color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_rt.update_layout(
                xaxis_title="", yaxis_title="Posts / week",
                legend_title_text="Vendor",
            )
            st.plotly_chart(fig_rt, use_container_width=True)

            # --- Weekly time-series: engagement score per week by vendor ---
            st.markdown(f"##### Community engagement — weekly score by vendor (last {time_label})")
            reddit_score_ts = _rdf_ts.groupby(["week", "vendor"], as_index=False).agg(
                total_score=("score", "sum"),
            )
            fig_rsc = px.line(
                reddit_score_ts, x="week", y="total_score", color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_rsc.update_layout(
                xaxis_title="", yaxis_title="Score / week",
                legend_title_text="Vendor",
            )
            st.plotly_chart(fig_rsc, use_container_width=True)

            # --- Totals summaries (bar charts) ---
            st.markdown(f"##### Posts mentioning each vendor — totals (last {time_label})")
            fig_rv = px.bar(
                vendor_roll,
                x="vendor",
                y="posts",
                text="posts",
                color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_rv.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Posts"
            )
            st.plotly_chart(fig_rv, use_container_width=True)

            st.markdown("##### Community engagement — total score")
            fig_rs = px.bar(
                vendor_roll,
                x="vendor",
                y="total_score",
                text="total_score",
                color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_rs.update_layout(
                showlegend=False, xaxis_title="",
                yaxis_title="Total score (upvotes - downvotes)",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

            # Subreddit x vendor heatmap
            st.markdown("##### Mentions by subreddit x vendor")
            heat = (
                rdf.drop_duplicates(subset=["post_id", "vendor", "subreddit"])
                .groupby(["subreddit", "vendor"], as_index=False)
                .agg(posts=("post_id", "nunique"))
            )
            if not heat.empty:
                heat_pivot = heat.pivot(
                    index="subreddit", columns="vendor", values="posts"
                ).fillna(0).astype(int)
                st.dataframe(heat_pivot, use_container_width=True)

            # Recent Qualcomm posts
            st.markdown("##### Recent Qualcomm-mention posts")
            qcom_df = (
                rdf[rdf["vendor"] == "Qualcomm"]
                .drop_duplicates(subset=["post_id"])
                .sort_values("created_utc", ascending=False)
                .head(20)
            )
            if not qcom_df.empty:
                st.dataframe(
                    qcom_df[[
                        "created_utc", "subreddit", "keyword",
                        "title", "score", "num_comments", "url",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "url": st.column_config.LinkColumn("Link"),
                        "created_utc": st.column_config.DatetimeColumn(
                            "Date", format="YYYY-MM-DD"
                        ),
                    },
                )

    # ── GitHub issues ────────────────────────────────────────────────
    with st.expander("GitHub issues", expanded=True):
        st.markdown(
            "#### GitHub issues + discussions — vendor mentions in "
            "on-device AI repos"
        )
        st.caption(
            "When a developer files an issue mentioning 'Qualcomm' or "
            "'Snapdragon' in llama.cpp, that's a concrete signal: someone "
            "tried to run a model on Qualcomm hardware."
        )

        if gh_counts_df.empty:
            st.info("No GitHub data for selected vendors.")
        else:
            ghc = gh_counts_df.copy()

            vendor_totals = (
                ghc.groupby("vendor", as_index=False)["total_count"]
                .sum()
                .sort_values("total_count", ascending=False)
            )

            k1, k2, k3 = st.columns(3)
            k1.metric("Repos searched", f"{ghc['repo'].nunique():,}")
            k2.metric(
                "Total issue/discussion matches",
                f"{int(ghc['total_count'].sum()):,}",
            )
            qcom_gh = int(
                vendor_totals.loc[
                    vendor_totals["vendor"] == "Qualcomm", "total_count"
                ].sum()
            )
            k3.metric("Qualcomm mentions", f"{qcom_gh:,}")

            # --- Weekly time-series: new issues by vendor ---
            st.markdown("##### Vendor share of voice — weekly new issues")
            if not gh_detail_df.empty:
                _ghd = gh_detail_df.copy()
                _ghd["created_at"] = pd.to_datetime(_ghd["created_at"], utc=True, errors="coerce")
                _cutoff_gh = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
                _ghd = _ghd[_ghd["created_at"] >= _cutoff_gh]
                _ghd = _ghd.drop_duplicates(subset=["repo", "number", "vendor"])
                _ghd["week"] = _ghd["created_at"].dt.to_period("W").dt.to_timestamp()
                _ghd = drop_partial_weeks(_ghd)
                gh_ts = _ghd.groupby(["week", "vendor"], as_index=False).agg(
                    issues=("number", "nunique"),
                )
                fig_ght = px.line(
                    gh_ts, x="week", y="issues", color="vendor",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_ght.update_layout(
                    xaxis_title="", yaxis_title="New issues / week",
                    legend_title_text="Vendor",
                )
                st.plotly_chart(fig_ght, use_container_width=True)

            # --- Totals summary (bar chart) ---
            st.markdown("##### Vendor share of voice — totals across repos")
            fig_gv = px.bar(
                vendor_totals,
                x="vendor",
                y="total_count",
                text="total_count",
                color="vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_gv.update_layout(
                showlegend=False, xaxis_title="",
                yaxis_title="Issues + discussions mentioning vendor",
            )
            st.plotly_chart(fig_gv, use_container_width=True)

            # Per-repo breakdown
            st.markdown("##### Mentions by repo x vendor")
            repo_vendor = (
                ghc.groupby(["repo", "vendor"], as_index=False)["total_count"]
                .sum()
                .sort_values("total_count", ascending=False)
            )
            fig_gr = px.bar(
                repo_vendor,
                x="repo",
                y="total_count",
                color="vendor",
                text="total_count",
                barmode="group",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_gr.update_layout(
                xaxis_title="", yaxis_title="Mentions",
                legend_title_text="Vendor",
            )
            st.plotly_chart(fig_gr, use_container_width=True)

            # Keyword detail table
            st.markdown("##### Keyword x repo detail")
            show_ghc = ghc[["repo", "vendor", "keyword", "total_count"]].copy()
            show_ghc = show_ghc.sort_values("total_count", ascending=False)
            st.dataframe(
                show_ghc, use_container_width=True, hide_index=True,
                column_config={"total_count": st.column_config.NumberColumn(format="%,d")},
            )

            # Recent Qualcomm issues
            if not gh_detail_df.empty:
                _gh_det = gh_detail_df.copy()
                _gh_det["created_at"] = pd.to_datetime(
                    _gh_det["created_at"], utc=True, errors="coerce"
                )
                _cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
                _gh_det = _gh_det[_gh_det["created_at"] >= _cutoff]

                st.markdown("##### Recent Qualcomm-mention issues")
                qcom_gh_detail = (
                    _gh_det[_gh_det["vendor"] == "Qualcomm"]
                    .drop_duplicates(subset=["repo", "number"])
                    .sort_values("created_at", ascending=False)
                    .head(20)
                )
                if not qcom_gh_detail.empty:
                    st.dataframe(
                        qcom_gh_detail[[
                            "created_at", "repo", "title",
                            "state", "comments", "url",
                        ]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "url": st.column_config.LinkColumn("Link"),
                            "created_at": st.column_config.DatetimeColumn(
                                "Date", format="YYYY-MM-DD"
                            ),
                        },
                    )

    # ── GitHub repos ─────────────────────────────────────────────────
    with st.expander("GitHub repos", expanded=True):
        st.markdown("#### Toolchain Repo Health — Cross-Vendor Comparison")
        st.caption(
            "Stars and forks on vendor SDK repos are direct signals of "
            "developer awareness and active experimentation. Unlike PyPI "
            "downloads (which measure install intent), GitHub stars reflect "
            "discovery and bookmarking, while forks indicate hands-on usage."
        )

        if gh_repo_stats_df.empty:
            st.info("No repo stats data for selected vendors.")
        else:
            _grs = gh_repo_stats_df.copy()
            _grs["fetched_at"] = pd.to_datetime(_grs["fetched_at"], utc=True)
            _grs = _grs.sort_values("fetched_at").drop_duplicates(
                subset=["repo"], keep="last"
            )

            # --- Local controls: time window + vendor filter ---
            _gh_all_vendors = sorted(_grs["vendor"].unique())
            _fc1, _fc2 = st.columns([1, 3])
            with _fc1:
                _gh_time = st.select_slider(
                    "Time window",
                    options=list(TIME_OPTIONS.keys()),
                    value="6 months",
                    key="gh_repo_time",
                )
                _gh_window = TIME_OPTIONS[_gh_time]
            with _fc2:
                _gh_vendors = st.multiselect(
                    "Vendors",
                    options=_gh_all_vendors,
                    default=_gh_all_vendors,
                    key="gh_repo_vendors",
                )
            _gh_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_gh_window)

            # Filter repo stats by selected vendors
            _grs = _grs[_grs["vendor"].isin(_gh_vendors)]

            _vendor_repo = _grs.groupby("vendor", as_index=False).agg(
                stars=("stars", "sum"),
                forks=("forks", "sum"),
                open_issues=("open_issues", "sum"),
                repos=("repo", "count"),
            ).sort_values("stars", ascending=False)

            qcom_row = _vendor_repo[_vendor_repo["vendor"] == "Qualcomm"]
            qcom_stars = int(qcom_row["stars"].sum()) if not qcom_row.empty else 0
            qcom_forks = int(qcom_row["forks"].sum()) if not qcom_row.empty else 0
            top_vendor = _vendor_repo.iloc[0]["vendor"] if not _vendor_repo.empty else "—"
            top_stars = int(_vendor_repo.iloc[0]["stars"]) if not _vendor_repo.empty else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Qualcomm toolchain stars", f"{qcom_stars:,}")
            k2.metric("Qualcomm toolchain forks", f"{qcom_forks:,}")
            k3.metric("Top vendor (stars)", f"{top_vendor} ({top_stars:,})")

            # -- Stars over time (from stargazer history) --
            # Use unfiltered data — the local _gh_vendors multiselect handles vendor filtering
            _has_star_history = not gh_stargazers_df.empty
            _has_fork_history = not gh_forks_df.empty

            # Warn about vendors with missing history data
            _sg_available = set(gh_stargazers_df["vendor"].unique()) if _has_star_history else set()
            _fk_available = set(gh_forks_df["vendor"].unique()) if _has_fork_history else set()
            _missing_sg = set(_gh_vendors) - _sg_available
            _missing_fk = set(_gh_vendors) - _fk_available
            if _missing_sg or _missing_fk:
                _missing_all = _missing_sg | _missing_fk
                st.caption(
                    f"Star/fork history incomplete for: {', '.join(sorted(_missing_all))}. "
                    f"Re-run `collect_github_repos.py` with a GITHUB_TOKEN for full data."
                )

            if _has_star_history:
                _sg = gh_stargazers_df.copy()
                _sg["starred_at"] = pd.to_datetime(_sg["starred_at"], utc=True)
                _sg = _sg[(_sg["starred_at"] >= _gh_cutoff) & (_sg["vendor"].isin(_gh_vendors))]
                _sg["week"] = _sg["starred_at"].dt.to_period("W").dt.to_timestamp()
                _sg = drop_partial_weeks(_sg)
                _sg_weekly = _sg.groupby(["week", "vendor"], as_index=False).size()
                _sg_weekly = _sg_weekly.rename(columns={"size": "new_stars"})
                _sg_weekly = _sg_weekly.sort_values("week")

                st.markdown("##### New GitHub stars per week by vendor")
                if _sg_weekly.empty:
                    st.info("No star history for selected vendors.")
                else:
                    fig_rs2 = px.line(
                        _sg_weekly, x="week", y="new_stars", color="vendor",
                        color_discrete_map=VENDOR_COLOR,
                    )
                    fig_rs2.update_layout(
                        xaxis_title="", yaxis_title="New stars per week",
                        legend_title_text="Vendor",
                    )
                    st.plotly_chart(fig_rs2, use_container_width=True)

            if _has_fork_history:
                _fk = gh_forks_df.copy()
                _fk["forked_at"] = pd.to_datetime(_fk["forked_at"], utc=True)
                _fk = _fk[(_fk["forked_at"] >= _gh_cutoff) & (_fk["vendor"].isin(_gh_vendors))]
                _fk["week"] = _fk["forked_at"].dt.to_period("W").dt.to_timestamp()
                _fk = drop_partial_weeks(_fk)
                _fk_weekly = _fk.groupby(["week", "vendor"], as_index=False).size()
                _fk_weekly = _fk_weekly.rename(columns={"size": "new_forks"})
                _fk_weekly = _fk_weekly.sort_values("week")

                st.markdown("##### New forks per week by vendor")
                if _fk_weekly.empty:
                    st.info("No fork history for selected vendors.")
                else:
                    fig_rf = px.line(
                        _fk_weekly, x="week", y="new_forks", color="vendor",
                        color_discrete_map=VENDOR_COLOR,
                    )
                    fig_rf.update_layout(
                        xaxis_title="", yaxis_title="New forks per week",
                        legend_title_text="Vendor",
                    )
                    st.plotly_chart(fig_rf, use_container_width=True)

            # -- Open issues over time (from existing issue data) --
            if not gh_issue_topics_df.empty:
                _iss = gh_issue_topics_df.copy()
                _iss["created_at"] = pd.to_datetime(_iss["created_at"], utc=True)
                _iss = _iss[(_iss["created_at"] >= _gh_cutoff) & (_iss["vendor"].isin(_gh_vendors))]
                _iss["week_created"] = _iss["created_at"].dt.to_period("W").dt.to_timestamp()
                _iss = drop_partial_weeks(_iss, week_col="week_created")

                _created = _iss.groupby(["week_created", "vendor"], as_index=False).size()
                _created = _created.rename(columns={"week_created": "week", "size": "created"})

                _closed_iss = _iss[_iss["state"] == "closed"].copy()
                if "updated_at" in _closed_iss.columns:
                    _closed_iss["closed_at"] = pd.to_datetime(_closed_iss["updated_at"], utc=True)
                    _closed_iss["week_closed"] = _closed_iss["closed_at"].dt.to_period("W").dt.to_timestamp()
                    _closed_iss = drop_partial_weeks(_closed_iss, week_col="week_closed")
                    _closed = _closed_iss.groupby(["week_closed", "vendor"], as_index=False).size()
                    _closed = _closed.rename(columns={"week_closed": "week", "size": "closed"})
                else:
                    _closed = pd.DataFrame(columns=["week", "vendor", "closed"])

                _all_weeks = pd.merge(_created, _closed, on=["week", "vendor"], how="outer").fillna(0)
                _all_weeks = _all_weeks.sort_values("week")
                _all_weeks["net"] = _all_weeks["created"] - _all_weeks["closed"]
                _all_weeks["open_issues"] = _all_weeks.groupby("vendor")["net"].cumsum()

                st.markdown("##### New issues filed per week by vendor")
                fig_ri = px.line(
                    _created, x="week", y="created", color="vendor",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_ri.update_layout(
                    xaxis_title="", yaxis_title="Issues filed per week",
                    legend_title_text="Vendor",
                )
                st.plotly_chart(fig_ri, use_container_width=True)

            st.markdown("##### Individual repo detail")
            _repo_show = _grs[
                ["vendor", "repo", "role", "stars", "forks", "open_issues"]
            ].copy()
            _repo_show = _repo_show.sort_values("stars", ascending=False)
            st.dataframe(
                _repo_show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "stars": st.column_config.NumberColumn(format="%,d"),
                    "forks": st.column_config.NumberColumn(format="%,d"),
                    "open_issues": st.column_config.NumberColumn(format="%,d"),
                },
            )

            # -- Developer Engagement Ratio ----------------------------------
            # Compares time-aligned GitHub engagement (forks + production issues)
            # against PyPI downloads to measure actual usage per download.
            st.markdown("##### Developer Engagement Ratio")
            st.caption(
                "Measures **actual hands-on usage per download**. "
                "Forks (experimentation) and production-keyword issues (deployment) "
                "from the selected time window are divided by PyPI downloads over "
                "the same period. A higher ratio means a larger share of downloaders "
                "are actively building — not just installing."
            )

            # Normalize vendor names between PyPI ("Meta / cross") and GitHub ("Meta")
            _vendor_norm = {"Meta / cross": "Meta"}

            # PyPI downloads in the time window, by vendor
            _er_has_pypi = not pypi_daily_df.empty and "date" in pypi_daily_df.columns
            if _er_has_pypi:
                _er_pypi = pypi_daily_df.copy()
                _er_pypi["date"] = pd.to_datetime(_er_pypi["date"], utc=True, errors="coerce")
                _er_pypi = _er_pypi[_er_pypi["date"] >= _gh_cutoff]
                _er_pypi["vendor"] = _er_pypi["vendor"].replace(_vendor_norm)
                _er_dl = _er_pypi.groupby("vendor", as_index=False)["downloads"].sum()
                _er_dl = _er_dl.rename(columns={"downloads": "downloads"})
            else:
                _er_dl = pd.DataFrame(columns=["vendor", "downloads"])

            # Forks in the time window, by vendor
            if not gh_forks_df.empty:
                _er_fk = gh_forks_df.copy()
                _er_fk["forked_at"] = pd.to_datetime(_er_fk["forked_at"], utc=True)
                _er_fk = _er_fk[_er_fk["forked_at"] >= _gh_cutoff]
                _er_forks = _er_fk.groupby("vendor", as_index=False).size()
                _er_forks = _er_forks.rename(columns={"size": "forks"})
            else:
                _er_forks = pd.DataFrame(columns=["vendor", "forks"])

            # Production-keyword issues in the time window, by vendor
            if not gh_issue_topics_df.empty:
                _er_iss = gh_issue_topics_df.copy()
                _er_iss["created_at"] = pd.to_datetime(_er_iss["created_at"], utc=True, errors="coerce")
                _er_iss = _er_iss[(_er_iss["created_at"] >= _gh_cutoff) & (_er_iss["topic_count"] > 0)]
                _er_prod = _er_iss.groupby("vendor", as_index=False).size()
                _er_prod = _er_prod.rename(columns={"size": "prod_issues"})
            else:
                _er_prod = pd.DataFrame(columns=["vendor", "prod_issues"])

            # Merge all three
            _er = _er_dl.merge(_er_forks, on="vendor", how="outer") \
                        .merge(_er_prod, on="vendor", how="outer") \
                        .fillna(0)
            for _c in ["downloads", "forks", "prod_issues"]:
                _er[_c] = _er[_c].astype(int)

            # Filter to selected vendors and require both download + engagement data
            _er = _er[_er["vendor"].isin(_gh_vendors)]
            _er = _er[(_er["downloads"] > 0) & ((_er["forks"] + _er["prod_issues"]) > 0)]

            if not _er.empty:
                _er["engagement"] = _er["forks"] + _er["prod_issues"]
                _er["ratio"] = (_er["engagement"] / _er["downloads"] * 10_000).round(1)
                _er = _er.sort_values("ratio", ascending=False)

                fig_er = px.bar(
                    _er,
                    x="vendor",
                    y="ratio",
                    color="vendor",
                    text="ratio",
                    color_discrete_map=VENDOR_COLOR,
                )
                fig_er.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_er.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title=f"Engagement per 10K downloads (last {_gh_time})",
                    height=400,
                )
                st.plotly_chart(fig_er, use_container_width=True)

                st.dataframe(
                    _er[["vendor", "downloads", "forks", "prod_issues", "engagement", "ratio"]].rename(
                        columns={
                            "downloads": f"PyPI downloads ({_gh_time})",
                            "forks": f"Forks ({_gh_time})",
                            "prod_issues": f"Prod issues ({_gh_time})",
                            "engagement": "Engagement (forks + issues)",
                            "ratio": "Per 10K downloads",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        f"PyPI downloads ({_gh_time})": st.column_config.NumberColumn(format="%,d"),
                        f"Forks ({_gh_time})": st.column_config.NumberColumn(format="%,d"),
                        f"Prod issues ({_gh_time})": st.column_config.NumberColumn(format="%,d"),
                        "Engagement (forks + issues)": st.column_config.NumberColumn(format="%,d"),
                    },
                )
            else:
                st.info("Not enough overlapping PyPI + GitHub data for selected vendors.")

    # ── Developer Lock-in Analysis ────────────────────────────────────
    with st.expander("Developer Lock-in Analysis", expanded=True):
        st.markdown("#### Native vs Abstracted Developer Ecosystem")
        st.caption(
            "**Native** toolchains (QNN, CoreML, TensorRT, OpenVINO) create hardware "
            "lock-in — developers who adopt them are tied to that vendor's silicon. "
            "**Abstracted** runtimes (ONNX Runtime, LiteRT, ExecuTorch, DirectML) "
            "create NPU utilization without lock-in. Both help justify the current "
            "premium, but only native adoption protects future premium."
        )

        _AFFINITY_COLOR = {
            "native": QUALCOMM_BLUE,
            "abstracted": "#BFBFBF",
            "mixed": _APPLE_COLOR,
        }

        # --- PyPI Lock-in Split ---
        _has_pypi_affinity = (
            not pypi_daily_df.empty and "affinity" in pypi_daily_df.columns
        )

        if _has_pypi_affinity:
            _daily_aff = pypi_daily_df.copy()
            _daily_aff["date"] = pd.to_datetime(_daily_aff["date"], errors="coerce")

            # --- Local controls: time window + vendor filter ---
            _li_all_vendors = sorted(_daily_aff["vendor"].unique())
            _lc1, _lc2 = st.columns([1, 3])
            with _lc1:
                _li_time = st.select_slider(
                    "Time window",
                    options=list(TIME_OPTIONS.keys()),
                    value="3 months",
                    key="lockin_time",
                )
                _li_window = TIME_OPTIONS[_li_time]
            with _lc2:
                _li_vendors = st.multiselect(
                    "Vendors",
                    options=_li_all_vendors,
                    default=_li_all_vendors,
                    key="lockin_vendors",
                )
            _li_cutoff = _daily_aff["date"].max() - pd.Timedelta(days=_li_window)
            _daily_aff = _daily_aff[
                (_daily_aff["date"] >= _li_cutoff) & (_daily_aff["vendor"].isin(_li_vendors))
            ]

            _daily_aff["week"] = _daily_aff["date"].dt.to_period("W").dt.to_timestamp()
            _daily_aff = drop_partial_weeks(_daily_aff)

            # --- Per-vendor native vs abstracted time-series ---
            _aff_vendor_wk = _daily_aff.groupby(
                ["week", "vendor", "affinity"], as_index=False
            )["downloads"].sum().sort_values("week")

            st.markdown("##### PyPI downloads — native vs abstracted by vendor (weekly)")
            fig_aff_v = px.line(
                _aff_vendor_wk,
                x="week",
                y="downloads",
                color="vendor",
                line_dash="affinity",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_aff_v.update_layout(
                xaxis_title="",
                yaxis_title="Weekly downloads",
                legend_title_text="Vendor / Affinity",
            )
            st.plotly_chart(fig_aff_v, use_container_width=True)

            # --- Lock-in Index metrics ---
            _last_30d = _daily_aff[
                _daily_aff["date"] >= _daily_aff["date"].max() - pd.Timedelta(days=30)
            ]
            _prev_30d = _daily_aff[
                (_daily_aff["date"] >= _daily_aff["date"].max() - pd.Timedelta(days=60))
                & (_daily_aff["date"] < _daily_aff["date"].max() - pd.Timedelta(days=30))
            ]

            def _calc_lockin(chunk: pd.DataFrame) -> float:
                nat = chunk[chunk["affinity"] == "native"]["downloads"].sum()
                abst = chunk[chunk["affinity"] == "abstracted"]["downloads"].sum()
                total = nat + abst
                return (nat / total * 100) if total > 0 else 0.0

            li_current = _calc_lockin(_last_30d)
            li_prior = _calc_lockin(_prev_30d)
            li_delta = li_current - li_prior

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Lock-in Index (last 30d)",
                f"{li_current:.1f}%",
                delta=f"{li_delta:+.1f}pp vs prior 30d",
            )
            _native_dl = int(
                _last_30d[_last_30d["affinity"] == "native"]["downloads"].sum()
            )
            _abst_dl = int(
                _last_30d[_last_30d["affinity"] == "abstracted"]["downloads"].sum()
            )
            m2.metric("Native downloads (30d)", f"{_native_dl:,}")
            m3.metric("Abstracted downloads (30d)", f"{_abst_dl:,}")

            # --- Per-vendor lock-in breakdown table ---
            st.markdown("##### Lock-in Index by vendor")
            _vendor_aff = _last_30d.groupby(
                ["vendor", "affinity"], as_index=False
            )["downloads"].sum()
            _vendor_nat = _vendor_aff[_vendor_aff["affinity"] == "native"].set_index("vendor")["downloads"]
            _vendor_abs = _vendor_aff[_vendor_aff["affinity"] == "abstracted"].set_index("vendor")["downloads"]
            _vendor_mix = _vendor_aff[_vendor_aff["affinity"] == "mixed"].set_index("vendor")["downloads"]
            _li_all = sorted(set(_vendor_nat.index) | set(_vendor_abs.index) | set(_vendor_mix.index))
            _li_table = pd.DataFrame({
                "vendor": _li_all,
                "native_dl": [int(_vendor_nat.get(v, 0)) for v in _li_all],
                "abstracted_dl": [int(_vendor_abs.get(v, 0)) for v in _li_all],
                "mixed_dl": [int(_vendor_mix.get(v, 0)) for v in _li_all],
            })
            _li_table["lock_in_pct"] = (
                _li_table["native_dl"]
                / (_li_table["native_dl"] + _li_table["abstracted_dl"]).replace(0, 1)
                * 100
            ).round(1)
            _li_table = _li_table.sort_values("lock_in_pct", ascending=False)
            st.dataframe(
                _li_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "native_dl": st.column_config.NumberColumn("Native DL", format="%,d"),
                    "abstracted_dl": st.column_config.NumberColumn("Abstracted DL", format="%,d"),
                    "mixed_dl": st.column_config.NumberColumn("Mixed DL", format="%,d"),
                },
            )

        else:
            st.info("No affinity data available. Run collect_pypi.py to tag packages with native/abstracted affinity.")

        # --- GitHub repo affinity split (time-series) ---
        _repo_affinity_map = {}
        if not gh_repo_stats_df.empty and "affinity" in gh_repo_stats_df.columns:
            _latest_stats = gh_repo_stats_df.copy()
            _latest_stats["fetched_at"] = pd.to_datetime(_latest_stats["fetched_at"], utc=True)
            _latest_stats = _latest_stats.sort_values("fetched_at").drop_duplicates(
                subset=["repo"], keep="last"
            )
            _repo_affinity_map = dict(zip(_latest_stats["repo"], _latest_stats["affinity"]))

        _can_show_sg = not gh_stargazers_df.empty and _repo_affinity_map
        _can_show_fk = not gh_forks_df.empty and _repo_affinity_map
        # Re-use lock-in time window if available, else default 6 months
        _li_gh_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=TIME_OPTIONS.get(_li_time, 90) if _has_pypi_affinity else 180
        )
        _li_gh_vendors = _li_vendors if _has_pypi_affinity else []

        if _can_show_sg:
            st.markdown("##### New GitHub stars per week — native vs abstracted")
            _sg_aff = gh_stargazers_df.copy()
            if "affinity" not in _sg_aff.columns:
                _sg_aff["affinity"] = _sg_aff["repo"].map(_repo_affinity_map).fillna("native")
            _sg_aff["starred_at"] = pd.to_datetime(_sg_aff["starred_at"], utc=True)
            _sg_aff = _sg_aff[_sg_aff["starred_at"] >= _li_gh_cutoff]
            if _li_gh_vendors:
                _sg_aff = _sg_aff[_sg_aff["vendor"].isin(_li_gh_vendors)]
            _sg_aff["week"] = _sg_aff["starred_at"].dt.to_period("W").dt.to_timestamp()
            _sg_aff = drop_partial_weeks(_sg_aff)
            _sg_aff_wk = _sg_aff.groupby(
                ["week", "affinity"], as_index=False
            ).size().rename(columns={"size": "new_stars"}).sort_values("week")
            fig_sg_aff = px.line(
                _sg_aff_wk,
                x="week",
                y="new_stars",
                color="affinity",
                color_discrete_map=_AFFINITY_COLOR,
            )
            fig_sg_aff.update_layout(
                xaxis_title="",
                yaxis_title="New stars per week",
                legend_title_text="Affinity",
            )
            st.plotly_chart(fig_sg_aff, use_container_width=True)

        if _can_show_fk:
            st.markdown("##### New forks per week — native vs abstracted")
            _fk_aff = gh_forks_df.copy()
            if "affinity" not in _fk_aff.columns:
                _fk_aff["affinity"] = _fk_aff["repo"].map(_repo_affinity_map).fillna("native")
            _fk_aff["forked_at"] = pd.to_datetime(_fk_aff["forked_at"], utc=True)
            _fk_aff = _fk_aff[_fk_aff["forked_at"] >= _li_gh_cutoff]
            if _li_gh_vendors:
                _fk_aff = _fk_aff[_fk_aff["vendor"].isin(_li_gh_vendors)]
            _fk_aff["week"] = _fk_aff["forked_at"].dt.to_period("W").dt.to_timestamp()
            _fk_aff = drop_partial_weeks(_fk_aff)
            _fk_aff_wk = _fk_aff.groupby(
                ["week", "affinity"], as_index=False
            ).size().rename(columns={"size": "new_forks"}).sort_values("week")
            fig_fk_aff = px.line(
                _fk_aff_wk,
                x="week",
                y="new_forks",
                color="affinity",
                color_discrete_map=_AFFINITY_COLOR,
            )
            fig_fk_aff.update_layout(
                xaxis_title="",
                yaxis_title="New forks per week",
                legend_title_text="Affinity",
            )
            st.plotly_chart(fig_fk_aff, use_container_width=True)

    # ── NPU Performance Benchmarks ───────────────────────────────────
    with st.expander("NPU Performance Benchmarks", expanded=False):
        st.markdown("#### Hardware Performance Comparison")
        st.caption(
            "Benchmark data from MLPerf Mobile, AI Benchmark, and Geekbench ML. "
            "Shows whether Qualcomm's hardware lead is growing or converging with competitors."
        )

        _bench_tabs = []
        _bench_labels = []
        if not mlperf_df.empty and len(mlperf_df) > 0:
            _bench_labels.append("MLPerf Mobile")
        if not ai_bench_df.empty and len(ai_bench_df) > 0:
            _bench_labels.append("AI Benchmark")
        if not geekbench_df.empty and len(geekbench_df) > 0:
            _bench_labels.append("Geekbench ML")

        if not _bench_labels:
            st.info(
                "No benchmark data available yet. Run the collectors:\n"
                "- `python collect_mlperf.py`\n"
                "- `python collect_ai_benchmark.py`\n"
                "- `python collect_geekbench.py`"
            )
        else:
            _bench_tabs = st.tabs(_bench_labels)
            _tab_idx = 0

            # --- MLPerf Mobile ---
            if not mlperf_df.empty and len(mlperf_df) > 0:
                with _bench_tabs[_tab_idx]:
                    st.markdown("##### MLPerf Mobile — on-device inference throughput")
                    st.markdown(
                        "**Source:** [MLCommons Mobile Inference Benchmark](https://mlcommons.org/benchmarks/inference-mobile/) "
                        "(Apache 2.0, [GitHub results repos](https://github.com/mlcommons))  \n"
                        "**Metric:** Queries Per Second (QPS) — how many inference requests "
                        "the device completes per second on standardized tasks "
                        "(image classification, object detection, NLP, segmentation, super-resolution, stable diffusion). "
                        "**Higher = faster.** Results are single-stream (one query at a time), "
                        "so QPS is effectively 1/latency.  \n"
                        "**Coverage note:** Only vendors who voluntarily submit appear. "
                        "Apple, Google, and MediaTek do not currently submit mobile results."
                    )
                    _mp = mlperf_df.copy()

                    if "result" in _mp.columns and _mp["result"].notna().any():
                        # Line chart: scores over versions
                        _mp_agg = _mp.groupby(
                            ["version", "vendor", "benchmark"], as_index=False
                        )["result"].max()

                        fig_mp_line = px.line(
                            _mp_agg,
                            x="version",
                            y="result",
                            color="vendor",
                            facet_col="benchmark",
                            facet_col_wrap=3,
                            markers=True,
                            color_discrete_map=VENDOR_COLOR,
                        )
                        fig_mp_line.update_layout(
                            height=500,
                            yaxis_title="Score (higher is better)",
                        )
                        st.plotly_chart(fig_mp_line, use_container_width=True)

                        # Bar chart: latest version head-to-head
                        _latest_ver = _mp["version"].max()
                        _mp_latest = _mp[_mp["version"] == _latest_ver]
                        if not _mp_latest.empty:
                            st.markdown(f"##### Head-to-head — {_latest_ver}")
                            _mp_h2h = _mp_latest.groupby(
                                ["vendor", "benchmark"], as_index=False
                            )["result"].max()
                            fig_mp_bar = px.bar(
                                _mp_h2h,
                                x="benchmark",
                                y="result",
                                color="vendor",
                                barmode="group",
                                color_discrete_map=VENDOR_COLOR,
                            )
                            fig_mp_bar.update_layout(
                                xaxis_title="",
                                yaxis_title="Score",
                                height=400,
                            )
                            st.plotly_chart(fig_mp_bar, use_container_width=True)
                    else:
                        st.info("MLPerf data loaded but no scores found.")
                _tab_idx += 1

            # --- AI Benchmark ---
            if not ai_bench_df.empty and len(ai_bench_df) > 0:
                with _bench_tabs[_tab_idx]:
                    st.markdown("##### AI Benchmark — mobile SoC AI performance")
                    st.markdown(
                        "**Source:** [AI Benchmark](https://ai-benchmark.com/ranking_processors) "
                        "by ETH Zurich — **mobile/edge SoC rankings** (Android benchmark app)  \n"
                        "**Metric:** Composite AI Score — a weighted sum of real neural network "
                        "inference workloads run on each mobile processor's CPU, GPU, and NPU/DSP. "
                        "Sub-scores cover INT8 and FP16 inference on CNNs and Transformers, "
                        "quantized CPU-only inference, and parallel multi-model execution. "
                        "**Higher = faster inference.** Covers 275 mobile SoCs across all major vendors "
                        "(Qualcomm, MediaTek, Samsung, Apple, Google, Huawei).  \n"
                        "**Key signal:** The scatter plot shows whether the performance gap between "
                        "Qualcomm and MediaTek is growing (premium justified) or converging (premium at risk)."
                    )
                    _ab = ai_bench_df.copy()

                    if "ai_score" in _ab.columns and _ab["ai_score"].notna().any():
                        # --- Local vendor filter ---
                        _ab_all_vendors = sorted(_ab["vendor"].unique())
                        _ab_default = [v for v in ["Qualcomm", "MediaTek", "Samsung", "Apple", "Google"] if v in _ab_all_vendors]
                        _ab_vendors = st.multiselect(
                            "Vendors",
                            options=_ab_all_vendors,
                            default=_ab_default or _ab_all_vendors,
                            key="ai_bench_vendors",
                        )
                        _ab = _ab[_ab["vendor"].isin(_ab_vendors)]

                        # Scatter: AI Score by vendor over release year
                        if "release_year" in _ab.columns and _ab["release_year"].notna().any():
                            fig_ab_scatter = px.scatter(
                                _ab.dropna(subset=["ai_score", "release_year"]),
                                x="release_year",
                                y="ai_score",
                                color="vendor",
                                hover_data=["processor"],
                                color_discrete_map=VENDOR_COLOR,
                            )
                            fig_ab_scatter.update_layout(
                                xaxis_title="Release year",
                                yaxis_title="AI Score",
                                height=450,
                            )
                            st.plotly_chart(fig_ab_scatter, use_container_width=True)

                        # Bar chart: top scores by vendor
                        _ab_top = (
                            _ab.groupby("vendor", as_index=False)["ai_score"]
                            .max()
                            .sort_values("ai_score", ascending=False)
                        )
                        st.markdown("##### Peak AI Score by vendor")
                        fig_ab_bar = px.bar(
                            _ab_top,
                            x="vendor",
                            y="ai_score",
                            text="ai_score",
                            color="vendor",
                            color_discrete_map=VENDOR_COLOR,
                        )
                        fig_ab_bar.update_traces(
                            texttemplate="%{text:.0f}", textposition="outside"
                        )
                        fig_ab_bar.update_layout(
                            showlegend=False,
                            xaxis_title="",
                            yaxis_title="AI Score",
                            height=380,
                        )
                        st.plotly_chart(fig_ab_bar, use_container_width=True)
                    else:
                        st.dataframe(
                            _ab, use_container_width=True, hide_index=True,
                            column_config={"ai_score": st.column_config.NumberColumn(format="%,d")},
                        )
                _tab_idx += 1

            # --- Geekbench ML ---
            if not geekbench_df.empty and len(geekbench_df) > 0:
                with _bench_tabs[_tab_idx]:
                    st.markdown("##### Geekbench ML — real-world device scores")
                    _gb = geekbench_df.copy()

                    if "score" in _gb.columns and _gb["score"].notna().any():
                        _gb_valid = _gb[_gb["score"] > 0]

                        # Box plot: score distributions by vendor
                        if not _gb_valid.empty:
                            fig_gb_box = px.box(
                                _gb_valid,
                                x="vendor",
                                y="score",
                                color="vendor",
                                color_discrete_map=VENDOR_COLOR,
                                points="outliers",
                            )
                            fig_gb_box.update_layout(
                                showlegend=False,
                                xaxis_title="",
                                yaxis_title="Geekbench ML Score",
                                height=450,
                            )
                            st.plotly_chart(fig_gb_box, use_container_width=True)

                        # Median score trend by vendor over time
                        if "test_date" in _gb.columns:
                            _gb_ts = _gb_valid.copy()
                            _gb_ts["test_date"] = pd.to_datetime(
                                _gb_ts["test_date"], errors="coerce"
                            )
                            _gb_ts = _gb_ts.dropna(subset=["test_date"])
                            if not _gb_ts.empty:
                                _gb_ts["month"] = _gb_ts["test_date"].dt.to_period("M").dt.to_timestamp()
                                _gb_monthly = _gb_ts.groupby(
                                    ["month", "vendor"], as_index=False
                                )["score"].median()
                                st.markdown("##### Median score trend by vendor")
                                fig_gb_line = px.line(
                                    _gb_monthly,
                                    x="month",
                                    y="score",
                                    color="vendor",
                                    markers=True,
                                    color_discrete_map=VENDOR_COLOR,
                                )
                                fig_gb_line.update_layout(
                                    xaxis_title="",
                                    yaxis_title="Median score",
                                    height=400,
                                )
                                st.plotly_chart(fig_gb_line, use_container_width=True)
                    else:
                        st.dataframe(
                            _gb, use_container_width=True, hide_index=True,
                            column_config={"score": st.column_config.NumberColumn(format="%,d")},
                        )
                _tab_idx += 1


# ===========================================================================
# 3 · PRODUCTIONIZATION — HF Spaces
# ===========================================================================

with st.expander("3 — Productionization", expanded=False):
    stage_header("stage4")

    sp = analyze.spaces_summary(spaces_df)
    if sp.empty:
        st.info("No spaces data.")
    else:
        st.caption(
            "Each bar = Spaces returned by HF text search for that keyword. "
            "Not a perfect attribution of 'which silicon the app runs on', "
            "but a useful directional signal for ecosystem mindshare."
        )
        fig_sp = px.bar(
            sp, x="keyword", y="spaces", text="spaces", title="Spaces per keyword"
        )
        fig_sp.update_layout(xaxis_title="", yaxis_title="Spaces")
        st.plotly_chart(fig_sp, use_container_width=True)

        st.markdown("#### SDK mix per keyword")
        sdk_mix = analyze.spaces_sdk_mix(spaces_df)
        fig_sdk = px.bar(
            sdk_mix,
            x="keyword",
            y="spaces",
            color="sdk",
            title="Space SDK breakdown (Gradio / Streamlit / Docker / Static)",
            barmode="stack",
        )
        fig_sdk.update_layout(xaxis_title="", yaxis_title="Spaces")
        st.plotly_chart(fig_sdk, use_container_width=True)


# ===========================================================================
# 4 · END-USER RUNTIME — Play Store + OEM design wins
# ===========================================================================

with st.expander("4 — End-User Runtime", expanded=False):
    stage_header("stage5")

    # ── Play Store ───────────────────────────────────────────────────
    with st.expander("Play Store apps", expanded=True):
        st.markdown("### Android apps with on-device AI — Play Store scrape")
        if playstore_df.empty:
            st.info("No Play Store data for selected vendors.")
        else:
            ps = playstore_df.copy()
            ps["play_min_installs"] = pd.to_numeric(
                ps["play_min_installs"], errors="coerce"
            ).fillna(0)
            ps["installs_label"] = ps["play_installs_str"]
            ps = ps.sort_values("play_min_installs", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Apps tracked", f"{len(ps):,}")
            total_min_installs = int(ps["play_min_installs"].sum())
            k2.metric("Combined floor installs", f"{total_min_installs:,}")
            snap_backend_mask = ps["backend_attribution"].str.contains(
                "QNN|Hexagon|Snapdragon|NNAPI", case=False, na=False
            )
            k3.metric(
                "Apps with Snapdragon-path backend",
                f"{int(snap_backend_mask.sum()):,} / {len(ps):,}",
            )

            st.markdown("#### Installs by app (log scale)")
            fig_inst = px.bar(
                ps,
                x="display_name",
                y="play_min_installs",
                color="category",
                log_y=True,
                text="installs_label",
                hover_data={
                    "play_min_installs": False,
                    "installs_label": True,
                    "play_score": True,
                    "vendor": True,
                    "backend_attribution": True,
                    "ai_features": True,
                },
            )
            fig_inst.update_layout(
                xaxis_title="",
                yaxis_title="Play Store install floor (log)",
                legend_title_text="Category",
            )
            fig_inst.update_traces(textposition="outside")
            st.plotly_chart(fig_inst, use_container_width=True)

            st.markdown("#### App detail")
            show_cols = [
                "display_name", "vendor", "category", "installs_label",
                "play_score", "play_last_updated", "backend_attribution",
                "ai_features", "play_url",
            ]
            st.dataframe(
                ps[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "play_url": st.column_config.LinkColumn("Play Store"),
                    "play_score": st.column_config.NumberColumn("Rating", format="%.2f"),
                    "play_last_updated": st.column_config.DatetimeColumn(
                        "Last updated", format="YYYY-MM-DD"
                    ),
                },
            )

    # ── Design wins ──────────────────────────────────────────────────
    with st.expander("OEM design wins", expanded=True):
        st.markdown("### OEM flagship design wins by chipset")
        if design_wins_df.empty:
            st.info("No design wins data for selected vendors.")
        else:
            dw = design_wins_df.copy()
            dw["launch_date"] = pd.to_datetime(dw["launch_date"], errors="coerce")
            dw = dw.dropna(subset=["launch_date"]).sort_values("launch_date")

            # Apply time filter
            _cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
            dw_filtered = dw[dw["launch_date"] >= _cutoff.tz_localize(None)]
            if dw_filtered.empty:
                st.info(f"No design wins in the last {time_label}. Showing all data.")
                dw_filtered = dw

            k1, k2, k3, k4 = st.columns(4)
            qcom_wins = int((dw_filtered["chipset_vendor"] == "Qualcomm").sum())
            k1.metric("Devices tracked", f"{len(dw_filtered):,}")
            k2.metric("Qualcomm chipset wins", f"{qcom_wins:,} / {len(dw_filtered):,}")
            copilot_wins = int((dw_filtered["category"] == "copilot_plus_pc").sum())
            k3.metric("Copilot+ PCs", f"{copilot_wins:,}")
            flagship_android = int((dw_filtered["category"] == "flagship_android").sum())
            k4.metric("Flagship Android", f"{flagship_android:,}")

            st.markdown("#### Flagship launch timeline by chipset vendor")
            fig_dw = px.scatter(
                dw_filtered,
                x="launch_date",
                y="chipset_vendor",
                color="chipset_vendor",
                symbol="category",
                hover_data=["device", "oem", "chipset", "ai_features", "source"],
                color_discrete_map=VENDOR_COLOR,
            )
            fig_dw.update_traces(marker=dict(size=14))
            fig_dw.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_dw, use_container_width=True)

            st.markdown("#### Chipset vendor share of tracked design wins")
            share = (
                dw_filtered.groupby("chipset_vendor", as_index=False)
                .size()
                .rename(columns={"size": "devices"})
                .sort_values("devices", ascending=False)
            )
            fig_share = px.bar(
                share,
                x="chipset_vendor",
                y="devices",
                text="devices",
                color="chipset_vendor",
                color_discrete_map=VENDOR_COLOR,
            )
            fig_share.update_layout(showlegend=False, xaxis_title="", yaxis_title="Devices")
            st.plotly_chart(fig_share, use_container_width=True)

            st.markdown("#### Design wins table")
            st.dataframe(
                dw_filtered[[
                    "launch_date", "device", "oem", "chipset", "chipset_vendor",
                    "category", "ai_features", "source", "verified",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "source": st.column_config.LinkColumn("Source"),
                    "launch_date": st.column_config.DatetimeColumn(
                        "Launch", format="YYYY-MM-DD"
                    ),
                    "verified": st.column_config.CheckboxColumn("Verified"),
                },
            )
