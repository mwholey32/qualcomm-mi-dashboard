"""
collect.py — Pull signals from the Hugging Face Hub and cache them locally.

Usage:
    python collect.py                 # full refresh
    python collect.py --vendors-only  # only re-pull vendor org models
    python collect.py --spaces-only   # only re-pull spaces

Outputs (in ./data):
    vendor_models.parquet    one row per model in a silicon-vendor org
    framework_models.parquet one row per model matching a framework tag
    spaces.parquet           one row per space matching a vendor keyword
    run_metadata.json        timestamp + row counts for the last run

Design notes:
 - HF Hub API is public / read-only / no auth required for listing.
 - We cache to parquet so the Streamlit app reads snapshots, not the live API.
 - Re-run this script on a schedule (cron / task scheduler) to build a
   longitudinal series. Each run overwrites the current snapshot; if you want
   deltas, copy the parquet file to data/history/<timestamp>/ before re-running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Target set — edit here to tune what we track.
# ---------------------------------------------------------------------------

# Silicon / platform vendor HF orgs. These are the primary "publishing activity"
# proxies: net-new models, downloads, likes, cadence.
VENDOR_ORGS: list[str] = [
    "qualcomm",     # Qualcomm AI Hub — Snapdragon / Hexagon NPU optimized
    "nvidia",       # NVIDIA — CUDA / TensorRT / NIM
    "apple",        # Apple — Core ML / MLX / Neural Engine
    "Intel",        # Intel — OpenVINO / IPEX
    "amd",          # AMD — ROCm / Ryzen AI
    "google",       # Google — TPU / LiteRT / Gemma
    "microsoft",    # Microsoft — DirectML / ONNX Runtime / Phi
    "Arm",          # Arm — Cortex / Ethos NPU / Kleidi
    "arduino",      # Arduino — Qualcomm subsidiary (separate vendor)
    "edgeimpulse",  # Edge Impulse — Qualcomm acquisition (remapped → Qualcomm in app.py)
]

# Runtime / framework tags. HF models carry a `library_name` and free-form
# `tags`. These are the "share of runtime" view across the whole Hub.
FRAMEWORK_TAGS: dict[str, str] = {
    # label shown in UI            :  HF filter string
    "CUDA / TensorRT":              "tensorrt",
    "ONNX":                         "onnx",
    "OpenVINO (Intel)":             "openvino",
    "Core ML (Apple)":              "coreml",
    "MLX (Apple Silicon)":          "mlx",
    "LiteRT / TFLite":              "tflite",
    "QNN (Qualcomm)":               "qnn",
    "ExecuTorch":                   "executorch",
    "GGUF (llama.cpp)":             "gguf",
}

# Keywords used to scan HF Spaces for ecosystem mentions. Spaces text search
# is the closest thing to a "what's getting deployed" signal on the Hub.
SPACE_KEYWORDS: list[str] = [
    "snapdragon",
    "qualcomm",
    "hexagon",
    "tensorrt",
    "openvino",
    "coreml",
    "mlx",
    "npu",
    "on-device",
    "arduino",
    "edge impulse",
    "uno q",
]

# Cap how many results we pull per query. The public API is generous but we
# don't need more than this for MI-style trend analysis.
MAX_PER_QUERY = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

api = HfApi()


def _safe_dt(value) -> pd.Timestamp | None:
    """HF returns datetimes, ISO strings, or None. Normalize to UTC Timestamp."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _model_row(m, source_label: str) -> dict:
    """Project a ModelInfo into a flat record we can stick in a DataFrame.

    Note: `last_modified` is not always populated by the listing endpoint.
    Callers should prefer `created_at` as the primary temporal signal and
    treat `last_modified` as best-effort.
    """
    return {
        "source": source_label,
        "repo_id": m.id,
        "author": getattr(m, "author", None) or (m.id.split("/")[0] if "/" in m.id else None),
        "created_at": _safe_dt(getattr(m, "created_at", None)),
        "last_modified": _safe_dt(getattr(m, "last_modified", None)),
        "downloads": getattr(m, "downloads", 0) or 0,
        "likes": getattr(m, "likes", 0) or 0,
        "library_name": getattr(m, "library_name", None),
        "pipeline_tag": getattr(m, "pipeline_tag", None),
        "tags": ";".join(getattr(m, "tags", []) or []),
        "private": bool(getattr(m, "private", False)),
    }


# Fields to explicitly request from the listing API. Without `expand`, a
# default subset is returned; with `expand`, we get exactly these.
MODEL_EXPAND = [
    "createdAt",
    "lastModified",
    "downloads",
    "likes",
    "tags",
    "library_name",
    "pipeline_tag",
    "private",
]

SPACE_EXPAND = [
    "createdAt",
    "lastModified",
    "likes",
    "tags",
    "sdk",
    "private",
]


def _space_row(s, keyword: str) -> dict:
    return {
        "keyword": keyword,
        "repo_id": s.id,
        "author": getattr(s, "author", None) or (s.id.split("/")[0] if "/" in s.id else None),
        "sdk": getattr(s, "sdk", None),
        "created_at": _safe_dt(getattr(s, "created_at", None)),
        "last_modified": _safe_dt(getattr(s, "last_modified", None)),
        "likes": getattr(s, "likes", 0) or 0,
        "tags": ";".join(getattr(s, "tags", []) or []),
    }


def _iter_with_limit(it: Iterable, limit: int) -> Iterable:
    for i, x in enumerate(it):
        if i >= limit:
            return
        yield x


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_vendor_models() -> pd.DataFrame:
    """One row per model published by each vendor org."""
    rows: list[dict] = []
    for org in VENDOR_ORGS:
        t0 = time.time()
        try:
            it = api.list_models(
                author=org,
                limit=MAX_PER_QUERY,
                sort="downloads",
                expand=MODEL_EXPAND,
            )
            count = 0
            for m in _iter_with_limit(it, MAX_PER_QUERY):
                rows.append(_model_row(m, source_label=f"vendor:{org}"))
                count += 1
            print(f"  [vendor] {org:<14} {count:>5} models  ({time.time() - t0:4.1f}s)")
        except HfHubHTTPError as e:
            print(f"  [vendor] {org:<14} ERROR: {e}")
    return pd.DataFrame(rows)


def collect_framework_counts() -> pd.DataFrame:
    """Fast count-only pass: real model count per runtime tag (no 2k cap).

    Uses a lightweight expand (downloads only) and iterates without storing
    rows, so even tags with 100k+ models finish in under 30 seconds.
    """
    rows: list[dict] = []
    for label, tag in FRAMEWORK_TAGS.items():
        t0 = time.time()
        try:
            count = 0
            total_dl = 0
            for m in api.list_models(filter=tag, limit=None, expand=["downloads"]):
                count += 1
                total_dl += getattr(m, "downloads", 0) or 0
            rows.append({
                "framework_label": label,
                "framework_tag": tag,
                "total_models": count,
                "total_downloads": total_dl,
            })
            print(f"  [count]  {label:<22} {count:>7,} models  ({time.time() - t0:4.1f}s)")
        except HfHubHTTPError as e:
            print(f"  [count]  {label:<22} ERROR: {e}")
    return pd.DataFrame(rows)


def collect_framework_models() -> pd.DataFrame:
    """Top models per runtime tag (capped at MAX_PER_QUERY for detail data)."""
    rows: list[dict] = []
    for label, tag in FRAMEWORK_TAGS.items():
        t0 = time.time()
        try:
            it = api.list_models(
                filter=tag,
                limit=MAX_PER_QUERY,
                sort="downloads",
                expand=MODEL_EXPAND,
            )
            count = 0
            for m in _iter_with_limit(it, MAX_PER_QUERY):
                r = _model_row(m, source_label=f"framework:{label}")
                r["framework_label"] = label
                r["framework_tag"] = tag
                rows.append(r)
                count += 1
            print(f"  [fw]     {label:<22} {count:>5} models  ({time.time() - t0:4.1f}s)")
        except HfHubHTTPError as e:
            print(f"  [fw]     {label:<22} ERROR: {e}")
    return pd.DataFrame(rows)


def collect_spaces() -> pd.DataFrame:
    """One row per Space matching a hardware / vendor keyword."""
    rows: list[dict] = []
    for kw in SPACE_KEYWORDS:
        t0 = time.time()
        try:
            it = api.list_spaces(
                search=kw,
                limit=MAX_PER_QUERY,
                sort="likes",
                expand=SPACE_EXPAND,
            )
            count = 0
            for s in _iter_with_limit(it, MAX_PER_QUERY):
                rows.append(_space_row(s, keyword=kw))
                count += 1
            print(f"  [spaces] {kw:<14} {count:>5} spaces  ({time.time() - t0:4.1f}s)")
        except HfHubHTTPError as e:
            print(f"  [spaces] {kw:<14} ERROR: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendors-only", action="store_true")
    parser.add_argument("--frameworks-only", action="store_true")
    parser.add_argument("--spaces-only", action="store_true")
    args = parser.parse_args()

    do_all = not (args.vendors_only or args.frameworks_only or args.spaces_only)
    started = datetime.now(timezone.utc)
    summary: dict[str, int] = {}

    print(f"\n=== HF MI collection run @ {started.isoformat()} ===")

    if do_all or args.vendors_only:
        print("\n[1/3] Vendor org models")
        df = collect_vendor_models()
        df.to_parquet(DATA_DIR / "vendor_models.parquet", index=False)
        summary["vendor_models"] = len(df)
        print(f"  -> saved {len(df)} rows to data/vendor_models.parquet")

    if do_all or args.frameworks_only:
        print("\n[2/3a] Framework / runtime tag — full counts (uncapped)")
        counts_df = collect_framework_counts()
        counts_df.to_parquet(DATA_DIR / "framework_counts.parquet", index=False)
        summary["framework_counts"] = len(counts_df)
        print(f"  -> saved {len(counts_df)} rows to data/framework_counts.parquet")

        print("\n[2/3b] Framework / runtime tag — top models (detail)")
        df = collect_framework_models()
        df.to_parquet(DATA_DIR / "framework_models.parquet", index=False)
        summary["framework_models"] = len(df)
        print(f"  -> saved {len(df)} rows to data/framework_models.parquet")

    if do_all or args.spaces_only:
        print("\n[3/3] Spaces keyword search")
        df = collect_spaces()
        df.to_parquet(DATA_DIR / "spaces.parquet", index=False)
        summary["spaces"] = len(df)
        print(f"  -> saved {len(df)} rows to data/spaces.parquet")

    meta = {
        "run_started_utc": started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": summary,
        "vendor_orgs": VENDOR_ORGS,
        "framework_tags": FRAMEWORK_TAGS,
        "space_keywords": SPACE_KEYWORDS,
    }
    (DATA_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nDone. Metadata written to data/run_metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
