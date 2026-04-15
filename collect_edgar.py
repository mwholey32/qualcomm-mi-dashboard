"""
collect_edgar.py — SEC EDGAR full-text search for hardware / on-device AI
mentions in public-company filings. Free, no API key.

Usage:
    python collect_edgar.py

Output:
    data/edgar_mentions.parquet

Design notes:
 - Uses the EDGAR full-text search endpoint at https://efts.sec.gov/LATEST/search-index
 - SEC requires a descriptive User-Agent; set CONTACT_EMAIL below.
 - SEC rate limit is ~10 req/sec. We sleep 0.2s between requests to be polite.
 - Returns one row per (keyword, filing) match. A filing may appear once per
   keyword if it mentions multiple keywords — this is intentional; the Streamlit
   tab deduplicates when it needs to.
 - Quote-exact keywords reduce false positives (Hexagon AB, Hexagon Composites,
   etc.) — use 'Hexagon NPU' not bare 'hexagon'.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CONTACT_EMAIL = "qcom-mi-research@example.com"  # update to real contact before productionizing
USER_AGENT = f"Qualcomm MI Dashboard research {CONTACT_EMAIL}"

ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
FORMS = "10-K,10-Q"
START_DATE = "2022-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Quote-exact phrases. Bucketed by theme so we can show grouped counts.
QUERIES: dict[str, list[str]] = {
    "Qualcomm-specific": [
        '"Snapdragon"',
        '"Hexagon NPU"',
        '"Qualcomm AI Hub"',
    ],
    "On-device / NPU generic": [
        '"on-device AI"',
        '"neural processing unit"',
        '"edge AI"',
        '"AI PC"',
    ],
    "Competitor platforms": [
        '"Apple Neural Engine"',
        '"Core ML"',
        '"TensorRT"',
        '"OpenVINO"',
        '"Ryzen AI"',
    ],
}

PAGE_SIZE = 100  # EDGAR returns 100 per page
MAX_PAGES_PER_QUERY = 20  # 20 * 100 = 2,000 filings cap per keyword
REQUEST_DELAY_SEC = 0.2


def _get(params: dict) -> dict:
    r = requests.get(
        ENDPOINT,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _parse_hit(hit: dict, keyword: str, bucket: str) -> dict:
    src = hit.get("_source", {}) or {}
    hit_id = hit.get("_id", "")  # "ACCESSION:FILENAME"
    accession, _, filename = hit_id.partition(":")
    # Accession in "0000804328-24-000075" form; EDGAR filing URLs drop the dashes.
    cik = (src.get("ciks") or [""])[0].lstrip("0") or ""
    acc_nodash = accession.replace("-", "")
    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{filename}"
        if cik and accession and filename
        else ""
    )
    display = (src.get("display_names") or [""])[0]
    # "COMPANY NAME/STATE  (TICKER)  (CIK 0000123456)"
    ticker = ""
    name = display
    if "(" in display:
        name = display.split("(")[0].strip()
        try:
            ticker = display.split("(")[1].split(")")[0].strip()
        except IndexError:
            pass
    return {
        "bucket": bucket,
        "keyword": keyword,
        "company": name,
        "ticker": ticker,
        "cik": cik,
        "form": src.get("form"),
        "file_date": pd.to_datetime(src.get("file_date"), errors="coerce"),
        "period_ending": pd.to_datetime(src.get("period_ending"), errors="coerce"),
        "accession": accession,
        "filename": filename,
        "filing_url": filing_url,
    }


def search_keyword(keyword: str, bucket: str) -> list[dict]:
    rows: list[dict] = []
    total_value: int | None = None

    for page in range(MAX_PAGES_PER_QUERY):
        params = {
            "q": keyword,
            "forms": FORMS,
            "dateRange": "custom",
            "startdt": START_DATE,
            "enddt": END_DATE,
            "from": page * PAGE_SIZE,
        }
        try:
            data = _get(params)
        except requests.HTTPError as e:
            print(f"    HTTP error on '{keyword}' page {page}: {e}")
            break
        except Exception as e:
            print(f"    Error on '{keyword}' page {page}: {e}")
            break

        hits_wrap = data.get("hits", {})
        if total_value is None:
            total_value = (hits_wrap.get("total") or {}).get("value", 0)
        hits = hits_wrap.get("hits", []) or []
        if not hits:
            break

        for h in hits:
            rows.append(_parse_hit(h, keyword=keyword, bucket=bucket))

        if (page + 1) * PAGE_SIZE >= (total_value or 0):
            break

        time.sleep(REQUEST_DELAY_SEC)

    print(f"  [edgar] {keyword:<28} total={total_value or 0:>4}  pulled={len(rows):>4}")
    return rows


def main() -> int:
    print(f"\n=== SEC EDGAR collection run @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Forms: {FORMS} | Dates: {START_DATE} .. {END_DATE}")
    all_rows: list[dict] = []

    for bucket, keywords in QUERIES.items():
        print(f"\n[bucket] {bucket}")
        for kw in keywords:
            all_rows.extend(search_keyword(kw, bucket))
            time.sleep(REQUEST_DELAY_SEC)

    df = pd.DataFrame(all_rows)
    before = len(df)
    if not df.empty:
        df = df.drop_duplicates(subset=["keyword", "accession", "filename"])
    out = DATA_DIR / "edgar_mentions.parquet"
    df.to_parquet(out, index=False)
    print(f"\n-> saved {len(df)} rows to {out}  (deduped from {before})")
    if not df.empty:
        top_cos = (
            df.groupby("company").size().sort_values(ascending=False).head(10)
        )
        print("\nTop 10 filers by mention count:")
        for co, n in top_cos.items():
            print(f"  {co:<50} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
