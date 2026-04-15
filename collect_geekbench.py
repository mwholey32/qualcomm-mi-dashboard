"""
collect_geekbench.py — Pull Geekbench ML / AI benchmark scores.

Usage:
    python collect_geekbench.py

Output:
    data/geekbench_ai.parquet   one row per benchmark result

Data source:
    browser.geekbench.com — JSON API (set Accept: application/json header)

Why this matters:
    Geekbench ML captures real-world device performance across CPU, GPU, and NPU
    backends. Unlike synthetic benchmarks, these scores reflect actual user
    experiences. Score distributions by vendor reveal real-world performance gaps.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "geekbench_ai.parquet"

BASE_URL = "https://browser.geekbench.com"
HEADERS = {
    "User-Agent": "qualcomm-mi-dashboard (research prototype)",
    "Accept": "application/json",
}

REQUEST_DELAY = 2.0  # Be polite to Geekbench servers
MAX_PAGES = 5        # Pages per search query

# Search queries to find relevant devices
SEARCH_QUERIES: list[dict[str, str]] = [
    {"query": "Snapdragon 8 Elite",   "vendor": "Qualcomm"},
    {"query": "Snapdragon 8 Gen 3",   "vendor": "Qualcomm"},
    {"query": "Snapdragon 8 Gen 2",   "vendor": "Qualcomm"},
    {"query": "Snapdragon 8 Gen 1",   "vendor": "Qualcomm"},
    {"query": "Snapdragon X Elite",   "vendor": "Qualcomm"},
    {"query": "Snapdragon X Plus",    "vendor": "Qualcomm"},
    {"query": "Dimensity 9400",       "vendor": "MediaTek"},
    {"query": "Dimensity 9300",       "vendor": "MediaTek"},
    {"query": "Dimensity 9200",       "vendor": "MediaTek"},
    {"query": "Tensor G4",            "vendor": "Google"},
    {"query": "Tensor G3",            "vendor": "Google"},
    {"query": "A18 Pro",              "vendor": "Apple"},
    {"query": "A17 Pro",              "vendor": "Apple"},
    {"query": "M4",                   "vendor": "Apple"},
    {"query": "M3",                   "vendor": "Apple"},
    {"query": "Exynos 2400",          "vendor": "Samsung"},
    {"query": "Exynos 2500",          "vendor": "Samsung"},
]


def fetch_geekbench_search(query: str, page: int = 1) -> dict | None:
    """Search Geekbench browser for AI/ML benchmark results."""
    # Try the AI benchmark endpoint first, then ML
    for endpoint in ["/ai/v1/search", "/ml/v1/search", "/ai/search"]:
        url = f"{BASE_URL}{endpoint}"
        try:
            r = requests.get(
                url,
                params={"q": query, "page": page},
                headers=HEADERS,
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                if data and (isinstance(data, dict) and data.get("devices", data.get("results"))):
                    return data
                if isinstance(data, list) and data:
                    return {"results": data}
        except requests.exceptions.JSONDecodeError:
            continue
        except Exception as e:
            print(f"    ERROR {url}: {e}")
            continue
    return None


def fetch_geekbench_chart(query: str) -> dict | None:
    """Try the chart/ranking endpoints."""
    for endpoint in ["/ai/v1/chart", "/ml/v1/chart"]:
        url = f"{BASE_URL}{endpoint}"
        try:
            r = requests.get(
                url,
                params={"q": query},
                headers=HEADERS,
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                if data:
                    return data
        except Exception:
            continue
    return None


def parse_results(data: dict, vendor_hint: str) -> list[dict]:
    """Parse Geekbench JSON response into rows."""
    rows = []

    # Handle different response formats
    items = data.get("devices", []) or data.get("results", []) or data.get("data", [])
    if isinstance(data, list):
        items = data

    for item in items:
        if not isinstance(item, dict):
            continue

        device = item.get("device", "") or item.get("model", "") or item.get("name", "")
        soc = item.get("processor", "") or item.get("soc", "") or item.get("chip", "")
        score = item.get("score", 0) or item.get("ai_score", 0) or item.get("ml_score", 0)
        backend = item.get("backend", "") or item.get("runtime", "") or item.get("framework", "")

        # Parse test date
        test_date = item.get("uploaded", "") or item.get("date", "") or item.get("created", "")

        if device or soc:
            rows.append({
                "device": device,
                "soc": soc if soc else device,
                "vendor": vendor_hint,
                "score": float(score) if score else 0.0,
                "backend": backend,
                "test_date": test_date,
            })

    return rows


def main() -> int:
    print(f"\n=== Geekbench AI/ML collection @ {datetime.now(timezone.utc).isoformat()} ===")

    all_rows: list[dict] = []

    for sq in SEARCH_QUERIES:
        query, vendor = sq["query"], sq["vendor"]
        print(f"\n  Searching: {query} ({vendor})...")

        # Try search endpoint
        for page in range(1, MAX_PAGES + 1):
            data = fetch_geekbench_search(query, page=page)
            if not data:
                if page == 1:
                    # Try chart endpoint as fallback
                    data = fetch_geekbench_chart(query)
                    if data:
                        rows = parse_results(data, vendor)
                        all_rows.extend(rows)
                        print(f"    chart -> {len(rows)} results")
                else:
                    break

            if data:
                rows = parse_results(data, vendor)
                if not rows:
                    break
                all_rows.extend(rows)
                print(f"    page {page} -> {len(rows)} results")

            time.sleep(REQUEST_DELAY)

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Parse dates
        if "test_date" in df.columns:
            df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
        # Deduplicate
        df = df.drop_duplicates(subset=["device", "soc", "score", "backend"], keep="first")
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\n-> saved {len(df)} rows to {OUTPUT_PATH}")
        print(f"\nVendor breakdown:")
        for vendor in sorted(df["vendor"].unique()):
            v_df = df[df["vendor"] == vendor]
            avg = v_df["score"].mean()
            print(f"  {vendor}: {len(v_df)} results, avg score={avg:.0f}")
    else:
        print("\nNo results collected. Geekbench API may have changed.")
        print("Creating empty parquet with correct schema...")
        df = pd.DataFrame(columns=[
            "device", "soc", "vendor", "score", "backend", "test_date",
        ])
        df.to_parquet(OUTPUT_PATH, index=False)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
