"""
analyze.py — Transforms that turn raw HF snapshots into MI-ready views.

Kept deliberately small. Every function takes a DataFrame in, returns a
DataFrame out, and is called from app.py. No I/O here.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

# Keyword buckets used for model-card / tag text mining.
HARDWARE_KEYWORDS: dict[str, list[str]] = {
    "Qualcomm / Snapdragon": ["snapdragon", "hexagon", "qualcomm", "qnn", "aimet"],
    "NVIDIA / CUDA":         ["cuda", "tensorrt", "nvidia", "nim", "triton"],
    "Apple":                 ["coreml", "core-ml", "mlx", "neural engine", "ane", "mps"],
    "Intel":                 ["openvino", "ipex", "intel"],
    "AMD":                   ["rocm", "ryzen ai", "amd"],
    "Google":                ["tpu", "litert", "tflite", "edgetpu"],
    "Generic NPU / Edge":    ["npu", "on-device", "edge", "mobile"],
}


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _window(df: pd.DataFrame, column: str, days: int) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df.iloc[0:0]
    cutoff = _now_utc() - pd.Timedelta(days=days)
    s = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df[s >= cutoff]


# ---------------------------------------------------------------------------
# Vendor-level rollups
# ---------------------------------------------------------------------------

def vendor_summary(vendor_df: pd.DataFrame) -> pd.DataFrame:
    """One row per vendor: total models, downloads, likes, median age."""
    if vendor_df.empty:
        return pd.DataFrame(
            columns=["vendor", "models", "downloads", "likes", "median_days_since_update"]
        )

    df = vendor_df.copy()
    df["vendor"] = df["source"].str.replace("vendor:", "", regex=False)
    now = _now_utc()
    df["days_since_update"] = (now - pd.to_datetime(df["last_modified"], utc=True, errors="coerce")).dt.days

    out = (
        df.groupby("vendor")
        .agg(
            models=("repo_id", "count"),
            downloads=("downloads", "sum"),
            likes=("likes", "sum"),
            median_days_since_update=("days_since_update", "median"),
        )
        .reset_index()
        .sort_values("downloads", ascending=False)
    )
    return out


def vendor_velocity(vendor_df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """New models per vendor in the last N days (uses created_at)."""
    if vendor_df.empty:
        return pd.DataFrame(columns=["vendor", f"new_in_{window_days}d"])

    recent = _window(vendor_df, "created_at", window_days).copy()
    recent["vendor"] = recent["source"].str.replace("vendor:", "", regex=False)
    out = (
        recent.groupby("vendor")
        .size()
        .reset_index(name=f"new_in_{window_days}d")
        .sort_values(f"new_in_{window_days}d", ascending=False)
    )
    return out


def vendor_publish_timeline(vendor_df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Long-format time series: vendor x period x new_models."""
    if vendor_df.empty:
        return pd.DataFrame(columns=["period", "vendor", "new_models"])

    df = vendor_df.copy()
    df["vendor"] = df["source"].str.replace("vendor:", "", regex=False)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["period"] = df["created_at"].dt.to_period(freq).dt.to_timestamp()
    out = df.groupby(["period", "vendor"]).size().reset_index(name="new_models")
    return out


# ---------------------------------------------------------------------------
# Framework / runtime share of voice
# ---------------------------------------------------------------------------


# framework_summary() moved below classify_deployment() so it can share
# the counts_df parameter pattern.


def framework_velocity(framework_df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    if framework_df.empty:
        return pd.DataFrame(columns=["framework_label", f"new_in_{window_days}d"])
    recent = _window(framework_df, "created_at", window_days)
    out = (
        recent.groupby("framework_label")
        .size()
        .reset_index(name=f"new_in_{window_days}d")
        .sort_values(f"new_in_{window_days}d", ascending=False)
    )
    return out


# ---------------------------------------------------------------------------
# Spaces (productionization proxy)
# ---------------------------------------------------------------------------

def spaces_summary(spaces_df: pd.DataFrame) -> pd.DataFrame:
    if spaces_df.empty:
        return pd.DataFrame(columns=["keyword", "spaces", "likes", "median_days_since_update"])

    df = spaces_df.copy()
    now = _now_utc()
    df["days_since_update"] = (now - pd.to_datetime(df["last_modified"], utc=True, errors="coerce")).dt.days
    out = (
        df.groupby("keyword")
        .agg(
            spaces=("repo_id", "count"),
            likes=("likes", "sum"),
            median_days_since_update=("days_since_update", "median"),
        )
        .reset_index()
        .sort_values("spaces", ascending=False)
    )
    return out


def spaces_sdk_mix(spaces_df: pd.DataFrame) -> pd.DataFrame:
    if spaces_df.empty:
        return pd.DataFrame(columns=["keyword", "sdk", "spaces"])
    out = (
        spaces_df.groupby(["keyword", "sdk"], dropna=False)
        .size()
        .reset_index(name="spaces")
        .sort_values(["keyword", "spaces"], ascending=[True, False])
    )
    return out


# ---------------------------------------------------------------------------
# Keyword mining over tags
# ---------------------------------------------------------------------------

def tag_keyword_counts(models_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each hardware bucket, count models whose `tags` string matches
    any of the bucket's keywords. This runs against whatever DataFrame
    you pass — typically the union of vendor + framework pulls.
    """
    if models_df.empty:
        return pd.DataFrame(columns=["bucket", "models"])

    text = (models_df["tags"].fillna("") + " " + models_df["repo_id"].fillna("")).str.lower()
    rows = []
    for bucket, kws in HARDWARE_KEYWORDS.items():
        pattern = "|".join(re.escape(k) for k in kws)
        count = int(text.str.contains(pattern, regex=True, na=False).sum())
        rows.append({"bucket": bucket, "models": count})
    return pd.DataFrame(rows).sort_values("models", ascending=False)


# ---------------------------------------------------------------------------
# Deployment-tier classifier
#
# HF's runtime tags tell you *which toolchain* a model was published against.
# That's a direct proxy for the deployment target a vendor believes is
# structurally worth publishing to. The mapping below buckets each runtime
# into one of three tiers (plus ambiguous) so the dashboard can show
# Qualcomm's opportunity surface at a glance.
#
# The tiers are deliberately opinionated — defended in the dashboard caption.
# ---------------------------------------------------------------------------

TIER_ON_DEVICE = {
    "Core ML (Apple)",
    "MLX (Apple Silicon)",
    "LiteRT / TFLite",
    "ExecuTorch",
    "QNN (Qualcomm)",
}
TIER_EDGE_HYBRID = {
    "ONNX",
    "GGUF (llama.cpp)",
    "OpenVINO (Intel)",
}
TIER_DATACENTER = {
    "CUDA / TensorRT",
}


def deployment_tier(framework_label: str) -> str:
    if framework_label in TIER_ON_DEVICE:
        return "On-device"
    if framework_label in TIER_EDGE_HYBRID:
        return "Edge / hybrid"
    if framework_label in TIER_DATACENTER:
        return "Datacenter"
    return "Ambiguous"


def classify_deployment(
    framework_df: pd.DataFrame,
    counts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bucket HF models by deployment tier based on their runtime tag.

    If *counts_df* (from framework_counts.parquet) is provided, its
    ``total_models`` / ``total_downloads`` columns are used instead of the
    capped detail data — giving accurate numbers even for tags that exceed
    the per-query cap (e.g. GGUF 169k, ONNX 40k).
    """
    if counts_df is not None and not counts_df.empty:
        df = counts_df.copy()
        df["tier"] = df["framework_label"].map(deployment_tier)
        return (
            df.rename(columns={
                "total_models": "models",
                "total_downloads": "downloads",
            })[["tier", "framework_label", "models", "downloads"]]
            .sort_values(["tier", "models"], ascending=[True, False])
        )

    # Fallback: derive from the (potentially capped) detail data.
    if framework_df.empty:
        return pd.DataFrame(
            columns=["tier", "framework_label", "models", "downloads"]
        )
    df = framework_df.copy()
    df["tier"] = df["framework_label"].map(deployment_tier)
    out = (
        df.groupby(["tier", "framework_label"], as_index=False)
        .agg(
            models=("repo_id", "count"),
            downloads=("downloads", "sum"),
        )
        .sort_values(["tier", "models"], ascending=[True, False])
    )
    return out


def tier_totals(
    framework_df: pd.DataFrame,
    counts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per tier — for the Overview funnel + Stage 2 rollup."""
    tier_df = classify_deployment(framework_df, counts_df)
    if tier_df.empty:
        return pd.DataFrame(columns=["tier", "models", "downloads"])
    return (
        tier_df.groupby("tier", as_index=False)
        .agg(models=("models", "sum"), downloads=("downloads", "sum"))
        .sort_values("models", ascending=False)
    )


def framework_summary(
    framework_df: pd.DataFrame,
    counts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per runtime tag: total tagged models, downloads, likes.

    If *counts_df* is provided, model counts and downloads come from the
    uncapped count pass instead of the capped detail data.
    """
    if counts_df is not None and not counts_df.empty:
        df = counts_df.rename(columns={
            "total_models": "models",
            "total_downloads": "downloads",
        })[["framework_label", "models", "downloads"]].copy()
        df = df.sort_values("models", ascending=False)
        return df

    if framework_df.empty:
        return pd.DataFrame(
            columns=["framework_label", "models", "downloads", "likes",
                      "median_days_since_update"]
        )

    df = framework_df.copy()
    now = _now_utc()
    df["days_since_update"] = (
        now - pd.to_datetime(df["last_modified"], utc=True, errors="coerce")
    ).dt.days
    out = (
        df.groupby("framework_label")
        .agg(
            models=("repo_id", "count"),
            downloads=("downloads", "sum"),
            likes=("likes", "sum"),
            median_days_since_update=("days_since_update", "median"),
        )
        .reset_index()
        .sort_values("models", ascending=False)
    )
    return out


# ---------------------------------------------------------------------------
# Qualcomm competitive lens
# ---------------------------------------------------------------------------

def qualcomm_vs_peers(vendor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Qualcomm-centric comparison row vs each peer:
    model count ratio, download ratio, velocity ratio (90d).
    Output is for a single Qualcomm-highlight table on the Overview tab.
    """
    if vendor_df.empty:
        return pd.DataFrame()

    summ = vendor_summary(vendor_df).set_index("vendor")
    vel = vendor_velocity(vendor_df, 90).set_index("vendor")
    merged = summ.join(vel, how="left").fillna(0)

    if "qualcomm" not in merged.index:
        return merged.reset_index()

    q = merged.loc["qualcomm"]

    def _ratio(numer, denom_col):
        # Convert to float so pd.NA becomes NaN (round-safe).
        denom = merged[denom_col].replace(0, pd.NA).astype("Float64")
        return (numer / denom).astype("Float64").round(2)

    merged["qualcomm_model_ratio"] = _ratio(q["models"], "models")
    merged["qualcomm_downloads_ratio"] = _ratio(q["downloads"], "downloads")
    merged["qualcomm_velocity_ratio"] = _ratio(q["new_in_90d"], "new_in_90d")
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Vendor × deployment-tier cross-classification
# ---------------------------------------------------------------------------

# Map product families (from products.py) to deployment tier.
_FAMILY_TIER: dict[str, str] = {
    # Qualcomm
    "Snapdragon Mobile (Premium)": "On-device",
    "Snapdragon Mobile (Mid)":     "On-device",
    "Snapdragon Compute (PC)":     "Edge / hybrid",
    "Snapdragon Automotive":       "On-device",
    "Qualcomm IoT / Robotics":     "On-device",
    "Qualcomm Cloud":              "Datacenter",
    # AMD
    "Ryzen AI (Client NPU)":       "On-device",
    "Instinct (Data Center)":      "Datacenter",
    # Apple
    "Apple Silicon":               "On-device",
    "Apple Vision":                "On-device",
    # NVIDIA
    "NVIDIA Data Center":          "Datacenter",
    "NVIDIA Edge / Robotics":      "On-device",
    "NVIDIA Research / Foundation": "Datacenter",
    # Intel
    "Intel Client NPU":            "On-device",
    "Intel Data Center AI":        "Datacenter",
    "Intel GPU":                   "Edge / hybrid",
    "Intel (Quantized / OpenVINO)":"Edge / hybrid",
    # Google
    "Google TPU":                  "Datacenter",
    "Google Edge / Mobile":        "On-device",
    # Microsoft
    "Microsoft Client / Copilot+ PC": "Edge / hybrid",
    # Arm
    "Arm (IP)":                    "On-device",
    # Qualcomm (Edge Impulse)
    "Qualcomm Edge Impulse":       "On-device",
    # Arduino
    "Arduino Boards":              "On-device",
}

# Fallback tier when a model has no product-family classification.
_VENDOR_DEFAULT_TIER: dict[str, str] = {
    "qualcomm":  "On-device",
    "nvidia":    "Datacenter",
    "apple":     "On-device",
    "intel":     "Edge / hybrid",
    "amd":       "On-device",
    "google":    "Datacenter",
    "microsoft": "Edge / hybrid",
    "arm":       "On-device",
    "meta":      "On-device",
    "arduino":   "On-device",
}

_TIER_PRIORITY = ["On-device", "Edge / hybrid", "Datacenter"]


def vendor_tier_breakdown(
    vendor_df: pd.DataFrame,
    product_long: pd.DataFrame,
) -> pd.DataFrame:
    """Vendor × tier model counts.

    Tier is derived from product-family classification (most reliable) with
    vendor-level defaults for unclassified models.  Each model gets exactly
    one tier — when multiple families map to different tiers, the most
    on-device tier wins (priority: On-device > Edge > Datacenter).
    """
    if vendor_df.empty:
        return pd.DataFrame(columns=["vendor", "tier", "models"])

    df = vendor_df.copy()
    df["vendor"] = df["source"].str.replace("vendor:", "", regex=False)

    # Build repo_id → primary tier from product_long
    if not product_long.empty:
        pl = product_long.copy()
        pl["tier"] = pl["family"].map(_FAMILY_TIER)
        pl = pl.dropna(subset=["tier"])
        tier_pri = {t: i for i, t in enumerate(_TIER_PRIORITY)}
        pl["_pri"] = pl["tier"].map(tier_pri)
        repo_tier = (
            pl.sort_values("_pri")
            .drop_duplicates(subset=["repo_id"], keep="first")
            .set_index("repo_id")["tier"]
        )
    else:
        repo_tier = pd.Series(dtype=str)

    df["tier"] = df["repo_id"].map(repo_tier)
    vendor_default = df["vendor"].str.lower().map(_VENDOR_DEFAULT_TIER)
    df["tier"] = df["tier"].fillna(vendor_default).fillna("Unclassified")

    return (
        df.groupby(["vendor", "tier"])
        .size()
        .reset_index(name="models")
        .sort_values(["vendor", "models"], ascending=[True, False])
    )
