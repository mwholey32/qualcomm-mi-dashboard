"""
collect_cards.py — Fetch model card READMEs and cache them locally.

Usage:
    python collect_cards.py                        # fetch for all vendors (slow)
    python collect_cards.py --vendors qualcomm     # just one vendor
    python collect_cards.py --vendors qualcomm amd apple
    python collect_cards.py --top-per-vendor 100   # cap per vendor to top N by downloads

Why:
    HF's list_models endpoint doesn't return README text. Silicon-vendor models
    encode their actual hardware targets in the README (chipset tables, narrative
    mentions, benchmark tables). This collector fetches READMEs once and caches
    them so products.py can do pattern matching offline.

Output:
    data/model_cards.parquet  — columns: repo_id, fetched_at, readme_text

Incremental:
    Rows already in the cache are skipped unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CARDS_PATH = DATA_DIR / "model_cards.parquet"


def _load_cache() -> pd.DataFrame:
    if CARDS_PATH.exists():
        return pd.read_parquet(CARDS_PATH)
    return pd.DataFrame(columns=["repo_id", "fetched_at", "readme_text"])


def _save_cache(df: pd.DataFrame) -> None:
    # Stable ordering helps diffs across runs.
    df = df.drop_duplicates(subset=["repo_id"], keep="last").sort_values("repo_id")
    df.to_parquet(CARDS_PATH, index=False)


def fetch_readme(repo_id: str) -> str | None:
    """Download README.md for a model repo. None on any access failure."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md")
    except (EntryNotFoundError, RepositoryNotFoundError, GatedRepoError):
        return None
    except HfHubHTTPError as e:
        # 401/404/etc
        return None
    except Exception:
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendors",
        nargs="*",
        default=None,
        help="Subset of vendor orgs to fetch. Default: all vendors in vendor_models.parquet.",
    )
    parser.add_argument(
        "--top-per-vendor",
        type=int,
        default=0,
        help="If > 0, only fetch top N models per vendor (by downloads). Default 0 = all.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch even cached models.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between requests (default 0.05).",
    )
    args = parser.parse_args()

    vendor_path = DATA_DIR / "vendor_models.parquet"
    if not vendor_path.exists():
        print("Run `python collect.py` first to produce data/vendor_models.parquet.")
        return 1
    vendor_df = pd.read_parquet(vendor_path)

    # Which vendors to pull
    if args.vendors:
        sources = [f"vendor:{v}" for v in args.vendors]
        df = vendor_df[vendor_df["source"].isin(sources)].copy()
    else:
        df = vendor_df.copy()

    if args.top_per_vendor > 0:
        df = (
            df.sort_values("downloads", ascending=False)
            .groupby("source", group_keys=False)
            .head(args.top_per_vendor)
        )

    targets = df["repo_id"].dropna().unique().tolist()

    cache = _load_cache()
    already = set(cache["repo_id"].tolist()) if not args.refresh else set()
    todo = [r for r in targets if r not in already]

    print(f"\n=== Model card fetch @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Target repos:  {len(targets)}")
    print(f"Already cached: {len(targets) - len(todo)}")
    print(f"To fetch:       {len(todo)}")

    new_rows: list[dict] = []
    t0 = time.time()
    ok = 0
    fail = 0
    for i, repo_id in enumerate(todo, start=1):
        text = fetch_readme(repo_id)
        if text is None:
            fail += 1
        else:
            ok += 1
            new_rows.append({
                "repo_id": repo_id,
                "fetched_at": datetime.now(timezone.utc),
                "readme_text": text,
            })
        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            print(f"  [{i:>5}/{len(todo):<5}] ok={ok} fail={fail}  {rate:.1f} req/s")
        time.sleep(args.sleep)

        # Flush to disk every 100 so we don't lose progress.
        if len(new_rows) >= 100:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            _save_cache(cache)
            new_rows = []

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    _save_cache(cache)
    print(f"\n-> cache now contains {len(cache)} rows at {CARDS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
