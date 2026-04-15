"""
collect_ai_benchmark.py — Scrape processor AI scores from ai-benchmark.com.

Usage:
    python collect_ai_benchmark.py

Output:
    data/ai_benchmark_scores.parquet   one row per processor with AI scores

Data source:
    https://ai-benchmark.com/ranking_processors — HTML table

Why this matters:
    AI Benchmark tests real-world neural network inference across a wide range
    of mobile and edge processors. Tracking scores by vendor over time reveals
    whether the NPU performance gap between Qualcomm and competitors is growing
    or converging — critical for premium pricing justification.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "ai_benchmark_scores.parquet"

URL = "https://ai-benchmark.com/ranking_processors"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# Map processor name fragments to vendor taxonomy
PROCESSOR_VENDOR_MAP: list[tuple[str, str]] = [
    ("snapdragon", "Qualcomm"),
    ("adreno", "Qualcomm"),
    ("qcs", "Qualcomm"),
    ("dimensity", "MediaTek"),
    ("helio", "MediaTek"),
    ("mediatek", "MediaTek"),
    ("mt6", "MediaTek"),
    ("exynos", "Samsung"),
    ("tensor", "Google"),
    ("a18", "Apple"),
    ("a17", "Apple"),
    ("a16", "Apple"),
    ("a15", "Apple"),
    ("a14", "Apple"),
    ("a13", "Apple"),
    ("a12", "Apple"),
    ("apple m", "Apple"),
    ("m1", "Apple"),
    ("m2", "Apple"),
    ("m3", "Apple"),
    ("m4", "Apple"),
    ("kirin", "Huawei"),
    ("intel", "Intel"),
    ("nvidia", "NVIDIA"),
    ("jetson", "NVIDIA"),
    ("ethos", "Arm"),
    ("unisoc", "Unisoc"),
    ("tiger", "Unisoc"),
]

# Column layout of the ai-benchmark.com/ranking_processors table:
# 0: Processor  1: CPU Cores  2: AI Accelerator  3: Year  4: Lib
# 5: CPU-Q  6: CPU-F  7: INT8 CNNs  8: INT8 Transformer  9: INT8 Accuracy
# 10: FP16 CNNs  11: FP16 Transformer  12: FP16 Accuracy
# 13: INT16 CNNs  14: INT8 Parallel  15: FP16 Parallel  16: AI Score


def classify_vendor(processor_name: str) -> str:
    """Map a processor name to a vendor."""
    lower = processor_name.lower()
    for fragment, vendor in PROCESSOR_VENDOR_MAP:
        if fragment in lower:
            return vendor
    return "Other"


def _safe_int(text: str) -> int | None:
    """Parse an integer from cell text, ignoring non-numeric chars."""
    cleaned = text.strip().replace(",", "")
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def scrape_ai_benchmark() -> pd.DataFrame:
    """Scrape the AI Benchmark processor rankings table."""
    print(f"Fetching {URL}...")
    try:
        r = requests.get(URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching page: {e}")
        return pd.DataFrame()

    print(f"  Received {len(r.text):,} bytes")
    soup = BeautifulSoup(r.text, "lxml")

    # The data table is the second <table> on the page (index 1)
    tables = soup.find_all("table")
    if len(tables) < 2:
        print(f"  Expected 2+ tables, found {len(tables)}")
        return pd.DataFrame()

    data_table = tables[1]
    all_rows = data_table.find_all("tr")
    # First row is header, rest are data
    data_rows = [tr for tr in all_rows if tr.find("td", class_="column_5s")]
    print(f"  Found {len(data_rows)} processor rows")

    rows = []
    for tr in data_rows:
        cells = tr.find_all("td")
        if len(cells) < 17:
            continue

        processor = cells[0].get_text(strip=True)
        npu_type = cells[2].get_text(strip=True)
        release_year = _safe_int(cells[3].get_text(strip=True))
        cpu_q = _safe_int(cells[5].get_text(strip=True))
        cpu_f = _safe_int(cells[6].get_text(strip=True))
        int8_cnns = _safe_int(cells[7].get_text(strip=True))
        int8_transformer = _safe_int(cells[8].get_text(strip=True))
        fp16_cnns = _safe_int(cells[10].get_text(strip=True))
        fp16_transformer = _safe_int(cells[11].get_text(strip=True))
        ai_score = _safe_int(cells[16].get_text(strip=True))

        if not processor:
            continue

        rows.append({
            "processor": processor,
            "vendor": classify_vendor(processor),
            "ai_score": ai_score,
            "npu_type": npu_type,
            "release_year": release_year,
            "cpu_q_score": cpu_q,
            "cpu_f_score": cpu_f,
            "int8_cnns": int8_cnns,
            "int8_transformer": int8_transformer,
            "fp16_cnns": fp16_cnns,
            "fp16_transformer": fp16_transformer,
        })

    return pd.DataFrame(rows)


def main() -> int:
    print(f"\n=== AI Benchmark collection @ {datetime.now(timezone.utc).isoformat()} ===")

    df = scrape_ai_benchmark()

    if df.empty:
        print("\nNo data scraped. Creating empty parquet with correct schema...")
        df = pd.DataFrame(columns=[
            "processor", "vendor", "ai_score", "npu_type", "release_year",
            "cpu_q_score", "cpu_f_score", "int8_cnns", "int8_transformer",
            "fp16_cnns", "fp16_transformer",
        ])
        df.to_parquet(OUTPUT_PATH, index=False)
        return 0

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n-> saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"\nVendor breakdown:")
    for vendor in sorted(df["vendor"].unique()):
        v_df = df[df["vendor"] == vendor]
        avg = v_df["ai_score"].mean()
        top = v_df["ai_score"].max()
        avg_str = f", avg={avg:.0f}, top={top:.0f}" if pd.notna(avg) else ""
        print(f"  {vendor}: {len(v_df)} processors{avg_str}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
