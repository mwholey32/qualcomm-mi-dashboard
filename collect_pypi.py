"""
collect_pypi.py — Pull package download stats from pypistats.org.

Usage:
    python collect_pypi.py                  # recent + daily, all packages
    python collect_pypi.py --recent-only    # just rolling totals
    python collect_pypi.py --daily-only     # just daily time-series

Outputs:
    data/pypi_recent.parquet     last-day / last-week / last-month per package
    data/pypi_downloads.parquet  daily time-series per package (~180 days)

Why PyPI:
    A `pip install` is an unambiguous act of developer intent. You don't
    passively install a package the way you might star a GitHub repo. And
    pypistats.org exposes ~180 days of daily data with no auth, so we get a
    ready-made trend line for every vendor's SDK.

Curated package list below — edit to add/remove targets. Each package is
mapped to a vendor + product label so the dashboard can roll up by vendor
and compare "SDK intent" head-to-head.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RECENT_PATH = DATA_DIR / "pypi_recent.parquet"
DAILY_PATH = DATA_DIR / "pypi_downloads.parquet"


@dataclass(frozen=True)
class PyPIPackage:
    package: str        # PyPI name (case-insensitive but we use canonical)
    vendor: str         # bucket for vendor rollup
    product: str        # human-readable SDK / runtime name
    note: str = ""      # one-line description (shown in tooltip)
    affinity: str = "native"  # "native" | "abstracted" | "mixed"


# ---------------------------------------------------------------------------
# Curated target list. Edit here to tune the comparison set.
#
# Criteria for inclusion:
#   - The package is the *primary distribution channel* for a vendor's
#     hardware-targeted toolchain (compiler, runtime, optimizer, model zoo).
#   - Downloads can reasonably be interpreted as developer intent to target
#     that hardware. General-purpose ML libs (e.g. torch itself) are out.
# ---------------------------------------------------------------------------
PACKAGES: list[PyPIPackage] = [
    # ── Qualcomm ────────────────────────────────────────────────────────────
    PyPIPackage("qai-hub",         "Qualcomm", "Qualcomm AI Hub (client)",
                "CLI + Python client to compile/profile models on Snapdragon",
                affinity="native"),
    PyPIPackage("qai-hub-models",  "Qualcomm", "Qualcomm AI Hub Models",
                "Model zoo that maps to hf.co/qualcomm",
                affinity="native"),

    # ── Qualcomm (Edge Impulse acquisition) ──────────────────────────────
    PyPIPackage("edgeimpulse",        "Qualcomm", "Edge Impulse SDK",
                "Python SDK for Edge Impulse ML pipelines",
                affinity="mixed"),
    PyPIPackage("edgeimpulse-api",    "Qualcomm", "Edge Impulse API client",
                "API bindings for Edge Impulse web platform",
                affinity="mixed"),
    PyPIPackage("edge-impulse-linux", "Qualcomm", "Edge Impulse Linux runner",
                "Real-time ML classification on Linux devices",
                affinity="mixed"),

    # ── Arduino (separate vendor — Qualcomm subsidiary) ─────────────────
    PyPIPackage("arduino-iot-cloud",  "Arduino", "Arduino IoT Cloud SDK",
                "Device connectivity for Arduino IoT Cloud",
                affinity="mixed"),
    PyPIPackage("arduino-iot-client", "Arduino", "Arduino IoT Cloud API client",
                "REST API client for Arduino Cloud resources",
                affinity="mixed"),

    # ── NVIDIA ──────────────────────────────────────────────────────────────
    PyPIPackage("tensorrt",        "NVIDIA",   "TensorRT",
                "Optimized inference runtime",
                affinity="native"),
    PyPIPackage("tensorrt-llm",    "NVIDIA",   "TensorRT-LLM",
                "LLM compiler on top of TensorRT",
                affinity="native"),
    PyPIPackage("nemo-toolkit",    "NVIDIA",   "NeMo",
                "NVIDIA's LLM / speech training + fine-tuning toolkit",
                affinity="mixed"),
    PyPIPackage("tritonclient",    "NVIDIA",   "Triton Inference Server client",
                "Client SDK for NVIDIA Triton serving",
                affinity="abstracted"),
    PyPIPackage("cuda-python",     "NVIDIA",   "CUDA Python",
                "Python bindings for the CUDA driver",
                affinity="native"),

    # ── Apple ───────────────────────────────────────────────────────────────
    PyPIPackage("coremltools",     "Apple",    "Core ML Tools",
                "Convert models to .mlmodel / .mlpackage",
                affinity="native"),
    PyPIPackage("mlx",             "Apple",    "MLX",
                "Apple's array framework for Apple Silicon",
                affinity="native"),
    PyPIPackage("mlx-lm",          "Apple",    "MLX-LM",
                "LLM utilities on top of MLX",
                affinity="native"),

    # ── Intel ───────────────────────────────────────────────────────────────
    PyPIPackage("openvino",                    "Intel", "OpenVINO runtime",
                "Cross-HW inference runtime",
                affinity="native"),
    PyPIPackage("openvino-dev",                "Intel", "OpenVINO dev tools",
                "Model Optimizer + tooling (legacy umbrella)",
                affinity="native"),
    PyPIPackage("optimum-intel",               "Intel", "Optimum-Intel",
                "HF Optimum integration for OpenVINO / IPEX",
                affinity="native"),
    PyPIPackage("intel-extension-for-pytorch", "Intel", "IPEX",
                "Intel CPU/GPU PyTorch extension",
                affinity="native"),
    PyPIPackage("neural-compressor",           "Intel", "Neural Compressor",
                "Quantization / pruning toolkit",
                affinity="native"),

    # ── Microsoft ───────────────────────────────────────────────────────────
    PyPIPackage("onnxruntime",          "Microsoft", "ONNX Runtime (CPU)",
                "Cross-platform inference runtime",
                affinity="abstracted"),
    PyPIPackage("onnxruntime-gpu",      "Microsoft", "ONNX Runtime (CUDA)",
                "ONNX Runtime built against CUDA",
                affinity="abstracted"),
    PyPIPackage("onnxruntime-directml", "Microsoft", "ONNX Runtime (DirectML)",
                "ONNX Runtime via DirectML — hits Qualcomm/AMD/Intel NPUs on Windows",
                affinity="abstracted"),
    PyPIPackage("olive-ai",             "Microsoft", "Olive",
                "Model optimization pipeline",
                affinity="abstracted"),
    PyPIPackage("torch-directml",       "Microsoft", "torch-directml",
                "PyTorch on DirectML backend",
                affinity="abstracted"),

    # ── Google ──────────────────────────────────────────────────────────────
    PyPIPackage("ai-edge-torch",  "Google", "AI Edge Torch",
                "Google's new PyTorch → LiteRT converter — direct qai-hub analog",
                affinity="abstracted"),
    PyPIPackage("mediapipe",      "Google", "MediaPipe",
                "On-device ML solutions",
                affinity="abstracted"),
    PyPIPackage("tflite-runtime", "Google", "LiteRT runtime",
                "Slim TensorFlow Lite / LiteRT runtime",
                affinity="abstracted"),
    PyPIPackage("jax",            "Google", "JAX",
                "Array + autodiff / XLA",
                affinity="abstracted"),

    # ── Arm ─────────────────────────────────────────────────────────────────
    PyPIPackage("ethos-u-vela",   "Arm",    "Ethos-U Vela",
                "Compiler for the Ethos-U NPU",
                affinity="native"),

    # ── Cross-vendor frameworks (included as reference points) ─────────────
    PyPIPackage("executorch",     "Meta / cross", "ExecuTorch",
                "PyTorch runtime for mobile + embedded",
                affinity="abstracted"),
    PyPIPackage("optimum",        "HuggingFace",  "Optimum",
                "HF's unified hardware-backend router",
                affinity="abstracted"),
]


RECENT_URL = "https://pypistats.org/api/packages/{package}/recent"
DAILY_URL = "https://pypistats.org/api/packages/{package}/overall"
REQUEST_DELAY_SEC = 1.0          # Baseline: pypistats is polite but not generous.
RATE_LIMIT_BACKOFF_SEC = 15.0    # Wait this long after a 429 before retrying.
MAX_RETRIES = 3
HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}


def _get(url: str) -> dict | None:
    """GET with retry-on-429. Returns None on 404 or on terminal failure."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = RATE_LIMIT_BACKOFF_SEC * (attempt + 1)
                print(f"    429 rate-limited, sleeping {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            print(f"    HTTP error {url}: {e}")
            return None
        except Exception as e:
            print(f"    ERROR {url}: {e}")
            return None
    print(f"    GAVE UP after {MAX_RETRIES} attempts: {url}")
    return None


def fetch_recent(pkg: PyPIPackage) -> dict | None:
    """Rolling last-day / last-week / last-month counts."""
    data = _get(RECENT_URL.format(package=pkg.package))
    if not data:
        return None
    row = data.get("data") or {}
    return {
        "package": pkg.package,
        "vendor": pkg.vendor,
        "product": pkg.product,
        "note": pkg.note,
        "affinity": pkg.affinity,
        "last_day": int(row.get("last_day") or 0),
        "last_week": int(row.get("last_week") or 0),
        "last_month": int(row.get("last_month") or 0),
        "fetched_at": datetime.now(timezone.utc),
    }


def fetch_daily(pkg: PyPIPackage) -> pd.DataFrame:
    """Daily downloads (excluding mirrors/CI bots) back ~180 days."""
    data = _get(DAILY_URL.format(package=pkg.package))
    if not data:
        return pd.DataFrame()
    rows = data.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # pypistats returns both `with_mirrors` and `without_mirrors` categories.
    # Mirrors are CI farms, pip cache mirrors, etc. — we want organic intent,
    # so drop them.
    if "category" in df.columns:
        df = df[df["category"] == "without_mirrors"]
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0).astype(int)
    df["package"] = pkg.package
    df["vendor"] = pkg.vendor
    df["product"] = pkg.product
    df["affinity"] = pkg.affinity
    df["fetched_at"] = datetime.now(timezone.utc)
    return df[["package", "vendor", "product", "affinity", "date", "downloads", "fetched_at"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent-only", action="store_true")
    parser.add_argument("--daily-only", action="store_true")
    args = parser.parse_args()

    do_recent = not args.daily_only
    do_daily = not args.recent_only

    print(f"\n=== PyPI collection run @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Packages: {len(PACKAGES)}")

    if do_recent:
        print("\n[1/2] Recent (last day / week / month)")
        recent_rows: list[dict] = []
        for i, pkg in enumerate(PACKAGES, 1):
            row = fetch_recent(pkg)
            if row:
                print(
                    f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<32} "
                    f"day={row['last_day']:>8,}  wk={row['last_week']:>9,}  mo={row['last_month']:>10,}"
                )
                recent_rows.append(row)
            else:
                print(f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<32} NOT FOUND")
            time.sleep(REQUEST_DELAY_SEC)
        if recent_rows:
            pd.DataFrame(recent_rows).to_parquet(RECENT_PATH, index=False)
            print(f"  -> saved {len(recent_rows)} rows to {RECENT_PATH}")

    if do_daily:
        print("\n[2/2] Daily historical (~180 days per package)")
        frames: list[pd.DataFrame] = []
        for i, pkg in enumerate(PACKAGES, 1):
            df = fetch_daily(pkg)
            if df.empty:
                print(f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<32} no data")
            else:
                print(
                    f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<32} "
                    f"{len(df):>4} days  "
                    f"({df['date'].min().date()} .. {df['date'].max().date()})  "
                    f"total={df['downloads'].sum():>10,}"
                )
                frames.append(df)
            time.sleep(REQUEST_DELAY_SEC)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(DAILY_PATH, index=False)
            print(f"\n  -> saved {len(combined)} rows to {DAILY_PATH}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
