"""
collect_npm.py — Pull download stats for on-device AI JS packages from npm.

Usage:
    python collect_npm.py

Outputs:
    data/npm_downloads.parquet   daily time-series per package (~180 days)

Why npm:
    JavaScript is where browser-based ML and React Native mobile apps live.
    MediaPipe Tasks, ONNX Runtime Web, and TensorFlow.js are the primary
    SDKs for client-side AI in web + React Native on iOS/Android.

    npm's public API is free, unauthenticated, and gives daily download
    counts with 180 days of history.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "npm_downloads.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}
REQUEST_DELAY = 0.5


@dataclass(frozen=True)
class NpmPackage:
    package: str
    vendor: str
    product: str
    affinity: str = "abstracted"


PACKAGES: list[NpmPackage] = [
    # ── Google — MediaPipe web + TFJS ────────────────────────────────
    NpmPackage("@mediapipe/tasks-vision", "Google", "MediaPipe Tasks — Vision (web)",   "abstracted"),
    NpmPackage("@mediapipe/tasks-text",   "Google", "MediaPipe Tasks — Text (web)",     "abstracted"),
    NpmPackage("@mediapipe/tasks-audio",  "Google", "MediaPipe Tasks — Audio (web)",    "abstracted"),
    NpmPackage("@mediapipe/tasks-genai",  "Google", "MediaPipe Tasks — GenAI (web)",    "abstracted"),
    NpmPackage("@tensorflow/tfjs",                 "Google", "TensorFlow.js",             "abstracted"),
    NpmPackage("@tensorflow/tfjs-backend-webgpu",  "Google", "TFJS — WebGPU backend",     "abstracted"),
    NpmPackage("@tensorflow/tfjs-backend-webgl",   "Google", "TFJS — WebGL backend",      "abstracted"),
    NpmPackage("@tensorflow/tfjs-react-native",    "Google", "TFJS — React Native",       "abstracted"),

    # ── Microsoft — ONNX Runtime Web / React Native ──────────────────
    NpmPackage("onnxruntime-web",           "Microsoft", "ONNX Runtime (Web)",          "abstracted"),
    NpmPackage("onnxruntime-react-native",  "Microsoft", "ONNX Runtime (React Native)", "abstracted"),
    NpmPackage("onnxruntime-node",          "Microsoft", "ONNX Runtime (Node)",         "abstracted"),

    # ── Meta — ExecuTorch mobile ─────────────────────────────────────
    NpmPackage("react-native-executorch",   "Meta", "ExecuTorch React Native",          "abstracted"),
]

DOWNLOADS_URL = "https://api.npmjs.org/downloads/range/{start}:{end}/{package}"


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ERROR {url}: {e}")
        return None


def fetch_daily(pkg: NpmPackage, days: int = 180) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days)
    url = DOWNLOADS_URL.format(start=start.isoformat(), end=end.isoformat(), package=pkg.package)
    data = _get(url)
    if not data:
        return pd.DataFrame()
    rows = data.get("downloads") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["day"], errors="coerce")
    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0).astype(int)
    df["package"] = pkg.package
    df["vendor"] = pkg.vendor
    df["product"] = pkg.product
    df["affinity"] = pkg.affinity
    df["fetched_at"] = datetime.now(timezone.utc)
    return df[["package", "vendor", "product", "affinity", "date", "downloads", "fetched_at"]]


def main() -> int:
    print(f"\n=== npm collection @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Packages: {len(PACKAGES)}")

    frames: list[pd.DataFrame] = []
    for i, pkg in enumerate(PACKAGES, 1):
        df = fetch_daily(pkg)
        if df.empty:
            print(f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<40} no data")
        else:
            print(
                f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<40} "
                f"{len(df):>4} days  total={df['downloads'].sum():>12,}"
            )
            frames.append(df)
        time.sleep(REQUEST_DELAY)

    if not frames:
        print("\nNo data collected.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(combined)} rows to {OUT_PATH}")

    print("\nBy vendor (last 30d):")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    recent = combined[combined["date"] >= cutoff]
    summary = recent.groupby("vendor")["downloads"].sum().sort_values(ascending=False)
    for v, d in summary.items():
        print(f"  {v:<12} {d:>12,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
