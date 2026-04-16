"""
collect_nuget.py — Pull NuGet download counts for .NET AI packages.

Usage:
    python collect_nuget.py

Output:
    data/nuget_downloads.parquet   one row per package — cumulative downloads

Why NuGet:
    .NET is Microsoft's native developer ecosystem and a large non-PyPI
    audience for ML runtimes. ONNX Runtime and DirectML both ship as
    first-class NuGet packages. This is the biggest PyPI blind spot for
    Microsoft.

    NuGet only exposes total download counts (no time series).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "nuget_downloads.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}
REQUEST_DELAY = 0.5


@dataclass(frozen=True)
class NugetPackage:
    package: str
    vendor: str
    product: str
    affinity: str = "abstracted"


PACKAGES: list[NugetPackage] = [
    # ── Microsoft — ONNX Runtime family ──────────────────────────────
    NugetPackage("Microsoft.ML.OnnxRuntime",            "Microsoft", "ONNX Runtime (CPU)",          "abstracted"),
    NugetPackage("Microsoft.ML.OnnxRuntime.Gpu",        "Microsoft", "ONNX Runtime (CUDA)",         "abstracted"),
    NugetPackage("Microsoft.ML.OnnxRuntime.DirectML",   "Microsoft", "ONNX Runtime (DirectML)",     "abstracted"),
    NugetPackage("Microsoft.ML.OnnxRuntime.QNN",        "Microsoft", "ONNX Runtime (Qualcomm QNN)", "native"),
    NugetPackage("Microsoft.ML.OnnxRuntime.OpenVINO",   "Microsoft", "ONNX Runtime (OpenVINO)",     "native"),
    NugetPackage("Microsoft.ML.OnnxRuntime.Training",   "Microsoft", "ONNX Runtime (Training)",     "abstracted"),
    NugetPackage("Microsoft.ML.OnnxRuntime.Managed",    "Microsoft", "ONNX Runtime (Managed)",      "abstracted"),
    NugetPackage("Microsoft.AI.DirectML",               "Microsoft", "DirectML",                    "abstracted"),
    NugetPackage("Microsoft.ML",                         "Microsoft", "ML.NET",                     "abstracted"),

    # ── Intel — OpenVINO .NET (less common but exists) ───────────────
    # None widely-distributed on NuGet under Intel's own org

    # ── NVIDIA — CUDA .NET bindings ──────────────────────────────────
    # None widely distributed; most CUDA .NET use comes via ManagedCuda community package

    # ── Qualcomm / Apple / Google — no NuGet presence ────────────────
]

# NuGet v3 search API returns totalDownloads
SEARCH_URL = "https://azuresearch-usnc.nuget.org/query"


def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params or {}, timeout=30, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ERROR {url}: {e}")
        return None


def fetch_package(pkg: NugetPackage) -> dict | None:
    data = _get(SEARCH_URL, {"q": f"packageid:{pkg.package}", "prerelease": "true"})
    if not data:
        return None
    results = data.get("data") or []
    # Match exact package id (case-insensitive)
    match = next((r for r in results if r.get("id", "").lower() == pkg.package.lower()), None)
    if not match:
        return None
    return {
        "package": pkg.package,
        "vendor": pkg.vendor,
        "product": pkg.product,
        "affinity": pkg.affinity,
        "total_downloads": int(match.get("totalDownloads") or 0),
        "version_count": len(match.get("versions") or []),
        "latest_version": match.get("version"),
        "fetched_at": datetime.now(timezone.utc),
    }


def main() -> int:
    print(f"\n=== NuGet collection @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Packages: {len(PACKAGES)}\n")

    rows: list[dict] = []
    for i, pkg in enumerate(PACKAGES, 1):
        row = fetch_package(pkg)
        if row:
            print(
                f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<48} "
                f"downloads={row['total_downloads']:>14,}  v={row['version_count']:>3}"
            )
            rows.append(row)
        else:
            print(f"  [{i:>2}/{len(PACKAGES)}] {pkg.package:<48} NOT FOUND")
        time.sleep(REQUEST_DELAY)

    if not rows:
        print("\nNo data collected.")
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(df)} rows to {OUT_PATH}")

    print("\nBy vendor:")
    summary = df.groupby("vendor")["total_downloads"].sum().sort_values(ascending=False)
    for v, d in summary.items():
        print(f"  {v:<12} {d:>14,} cumulative downloads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
