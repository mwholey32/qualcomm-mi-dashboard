"""
collect_mlperf.py — Pull MLPerf Mobile benchmark results from mlcommons GitHub.

Usage:
    python collect_mlperf.py

Output:
    data/mlperf_mobile.parquet   one row per benchmark result per SoC per version

Data source:
    GitHub repos: mlcommons/mobile_results_v5.0, v4.0, v3.1, v3.0
    Apache 2.0 licensed, structured text/JSON, no auth needed.
    Results live in mlperf_log_summary.txt files; system descriptions in JSON
    with trailing commas (non-standard).

Why this matters:
    MLPerf Mobile is the industry-standard benchmark for on-device AI inference.
    Tracking scores over time shows whether Qualcomm's hardware lead is growing
    or converging with competitors (MediaTek, Samsung, Google).
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "mlperf_mobile.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}

# ---------------------------------------------------------------------------
# MLPerf Mobile result versions to fetch — newest first
# ---------------------------------------------------------------------------
VERSIONS = [
    {"repo": "mlcommons/mobile_results_v5.0", "version": "v5.0"},
    {"repo": "mlcommons/mobile_results_v4.0", "version": "v4.0"},
    {"repo": "mlcommons/mobile_results_v3.1", "version": "v3.1"},
    {"repo": "mlcommons/mobile_results_v3.0", "version": "v3.0"},
]

# Map SoC name fragments / submitter names to vendor taxonomy
SOC_VENDOR_MAP: list[tuple[str, str]] = [
    ("qualcomm", "Qualcomm"),
    ("snapdragon", "Qualcomm"),
    ("qrd", "Qualcomm"),
    ("qcs", "Qualcomm"),
    ("sm8", "Qualcomm"),
    ("sm7", "Qualcomm"),
    ("sd8", "Qualcomm"),
    ("sd7", "Qualcomm"),
    ("dimensity", "MediaTek"),
    ("mediatek", "MediaTek"),
    ("mt6", "MediaTek"),
    ("tensor", "Google"),
    ("pixel", "Google"),
    ("exynos", "Samsung"),
    ("samsung", "Samsung"),
    ("galaxy", "Samsung"),
    ("a17", "Apple"),
    ("a18", "Apple"),
    ("a16", "Apple"),
    ("a15", "Apple"),
    ("m1", "Apple"),
    ("m2", "Apple"),
    ("m3", "Apple"),
    ("m4", "Apple"),
    ("apple", "Apple"),
    ("iphone", "Apple"),
    ("ipad", "Apple"),
    ("kirin", "Huawei"),
    ("intel", "Intel"),
    ("nvidia", "NVIDIA"),
]


def classify_vendor(text: str) -> str:
    """Map a system/SoC/submitter name to a vendor."""
    lower = text.lower()
    for fragment, vendor in SOC_VENDOR_MAP:
        if fragment in lower:
            return vendor
    return "Other"


def fetch_github_tree(repo: str) -> list[dict] | None:
    """Fetch the full Git tree for a repo's default branch."""
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
            r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {repo} tree")
            return None
        return r.json().get("tree", [])
    except Exception as e:
        print(f"  ERROR fetching tree for {repo}: {e}")
        return None


def fetch_raw_text(repo: str, path: str) -> str | None:
    """Fetch a raw text file from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def parse_json_lenient(text: str) -> dict | None:
    """Parse JSON that may have trailing commas (MLPerf system files)."""
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def parse_summary_txt(text: str) -> dict:
    """Extract key metrics from mlperf_log_summary.txt."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "Scenario":
                result["scenario"] = val
            elif key == "90th percentile latency (ns)":
                result["latency_90_ns"] = int(val)
            elif key == "Result is":
                result["valid"] = val == "VALID"
            elif key == "QPS w/o loadgen overhead":
                result["qps"] = float(val)
            elif key == "Mean latency (ns)":
                result["mean_latency_ns"] = int(val)
            elif key == "50.00 percentile latency (ns)":
                result["latency_50_ns"] = int(val)
    return result


def parse_mlperf_results(repo: str, version: str) -> list[dict]:
    """Parse MLPerf Mobile results from a version repo."""
    rows: list[dict] = []

    tree = fetch_github_tree(repo)
    if not tree:
        print(f"  Could not fetch tree for {repo}")
        return rows

    # Find summary text files (performance runs only — skip accuracy)
    summary_files = [
        f["path"] for f in tree
        if f["type"] == "blob"
        and f["path"].endswith("mlperf_log_summary.txt")
        and "-performance/" in f["path"]
    ]

    # Find system description JSON files
    system_files = [
        f["path"] for f in tree
        if f["type"] == "blob"
        and f["path"].endswith(".json")
        and "/systems/" in f["path"].lower()
    ]

    print(f"  {repo}: {len(summary_files)} perf summaries, {len(system_files)} system files")

    # Parse system descriptions first — build SoC info map
    systems: dict[str, dict] = {}
    for spath in system_files:
        text = fetch_raw_text(repo, spath)
        if not text:
            continue
        data = parse_json_lenient(text)
        if not data:
            continue

        sys_name = Path(spath).stem
        # MLPerf system JSON has nested "system" key
        sys_block = data.get("system", data)
        notes_block = data.get("notes", {})

        soc = (
            sys_block.get("Processor Name", "")
            or sys_block.get("processor", "")
            or notes_block.get("SoC Product information", "")
            or sys_name
        )
        submitter = (
            # Derive submitter from path: closed/<submitter>/systems/...
            spath.split("/")[1] if len(spath.split("/")) > 2 else ""
        )
        vendor = classify_vendor(f"{soc} {submitter} {sys_name}")

        systems[sys_name] = {
            "soc": soc,
            "vendor": vendor,
            "submitter": submitter,
        }
        time.sleep(0.2)

    print(f"    Parsed {len(systems)} system descriptions")

    # Parse performance summary files
    for spath in summary_files:
        text = fetch_raw_text(repo, spath)
        if not text:
            continue

        metrics = parse_summary_txt(text)
        if not metrics.get("qps"):
            continue

        # Extract system name and benchmark from path
        # Path format: closed/<submitter>/results/<system>/<benchmark>-performance/mlperf_log_summary.txt
        parts = spath.split("/")
        system_name = ""
        benchmark = ""
        submitter_from_path = parts[1] if len(parts) > 2 else ""
        for i, part in enumerate(parts):
            if part == "results" and i + 1 < len(parts):
                system_name = parts[i + 1]
            if part.endswith("-performance"):
                benchmark = part.replace("-performance", "")

        sys_info = systems.get(system_name, {})
        soc = sys_info.get("soc", system_name)
        vendor = sys_info.get("vendor", classify_vendor(f"{system_name} {submitter_from_path}"))
        submitter = sys_info.get("submitter", submitter_from_path)

        rows.append({
            "vendor": vendor,
            "soc": soc,
            "benchmark": benchmark,
            "result": metrics["qps"],
            "units": "QPS",
            "latency_90_ns": metrics.get("latency_90_ns"),
            "mean_latency_ns": metrics.get("mean_latency_ns"),
            "scenario": metrics.get("scenario", ""),
            "valid": metrics.get("valid", False),
            "version": version,
            "submitter": submitter,
            "system": system_name,
        })

        time.sleep(0.2)

    return rows


def main() -> int:
    print(f"\n=== MLPerf Mobile collection @ {datetime.now(timezone.utc).isoformat()} ===")

    all_rows: list[dict] = []
    for ver in VERSIONS:
        print(f"\nProcessing {ver['repo']} ({ver['version']})...")
        rows = parse_mlperf_results(ver["repo"], ver["version"])
        all_rows.extend(rows)
        print(f"  -> {len(rows)} results extracted")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\n-> saved {len(df)} rows to {OUTPUT_PATH}")
        print(f"\nVendor breakdown:")
        for vendor in sorted(df["vendor"].unique()):
            v_df = df[df["vendor"] == vendor]
            print(f"  {vendor}: {len(v_df)} results")
        print(f"\nBenchmark breakdown:")
        for bench in sorted(df["benchmark"].unique()):
            print(f"  {bench}: {len(df[df['benchmark'] == bench])} results")
    else:
        print("\nNo results collected. The MLPerf repo structure may have changed.")
        print("Creating empty parquet with correct schema...")
        df = pd.DataFrame(columns=[
            "vendor", "soc", "benchmark", "result", "units",
            "latency_90_ns", "mean_latency_ns", "scenario", "valid",
            "version", "submitter", "system",
        ])
        df.to_parquet(OUTPUT_PATH, index=False)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
