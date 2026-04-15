"""
collect_reddit.py — Track hardware vendor mentions in on-device AI subreddits.

Usage:
    python collect_reddit.py

Output:
    data/reddit_mentions.parquet   one row per post mentioning a keyword

Why Reddit:
    r/LocalLLaMA is the largest community of people actively running LLMs
    on local hardware. When a user posts about Snapdragon or Qualcomm there,
    it's an organic signal of developer/enthusiast mindshare — unprompted,
    unsolicited, and timestamped. Comparing Qualcomm mention volume against
    NVIDIA, Apple, Intel, and AMD gives a direct competitive share-of-voice.

    Reddit's JSON API is free, unauthenticated, and returns post metadata
    including score (upvotes - downvotes) and comment count — proxies for
    how much the community cares about that topic.

Limitations:
    - Reddit caps search results at ~250 per query (pagination via `after`).
    - Only post titles/selftext are searched — comments are not indexed.
    - Score and num_comments are point-in-time snapshots.
    - Rate limited to ~60 requests/minute without OAuth.
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
OUT_PATH = DATA_DIR / "reddit_mentions.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard/0.1 (research prototype)"}
REQUEST_DELAY = 1.2  # Reddit asks for ~1 req/sec without OAuth


@dataclass(frozen=True)
class RedditTarget:
    subreddit: str
    keyword: str
    vendor: str


# ---------------------------------------------------------------------------
# Curated targets: subreddit × keyword × vendor
#
# We search each subreddit for vendor-specific keywords. This gives a
# share-of-voice matrix: which vendors get talked about, where, and how much.
# ---------------------------------------------------------------------------

SUBREDDITS = [
    "LocalLLaMA",
    "StableDiffusion",
    "MachineLearning",
    "comfyui",
    "arduino",
    "embedded",
    "androiddev",
]

VENDOR_KEYWORDS: dict[str, list[str]] = {
    "Qualcomm":  ["qualcomm", "snapdragon", "hexagon", "edge impulse"],
    "NVIDIA":    ["nvidia", "cuda"],
    "Apple":     ["apple silicon", "mlx"],
    "Intel":     ["intel", "openvino"],
    "AMD":       ["amd", "rocm"],
    "Google":    ["google ai", "litert", "tflite", "mediapipe"],
    "Arduino":   ["arduino", "uno q", "ventuno"],
    "Generic":   ["npu", "on-device"],
}


def search_subreddit(
    subreddit: str, keyword: str, max_pages: int = 3
) -> list[dict]:
    """Search a subreddit for a keyword. Paginates up to max_pages."""
    results: list[dict] = []
    after: str | None = None

    for page in range(max_pages):
        params: dict = {
            "q": keyword,
            "sort": "new",
            "limit": 100,
            "restrict_sr": "on",
            "t": "year",
        }
        if after:
            params["after"] = after

        try:
            r = requests.get(
                f"https://www.reddit.com/r/{subreddit}/search.json",
                params=params,
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code == 429:
                print(f"    429 rate-limited, sleeping 10s")
                time.sleep(10)
                continue
            if r.status_code != 200:
                break

            data = r.json().get("data", {})
            children = data.get("children", [])
            if not children:
                break

            for child in children:
                d = child.get("data", {})
                results.append({
                    "subreddit": subreddit,
                    "keyword": keyword,
                    "post_id": d.get("id"),
                    "title": d.get("title"),
                    "author": d.get("author"),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "created_utc": d.get("created_utc"),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                })

            after = data.get("after")
            if not after:
                break

        except Exception as e:
            print(f"    ERROR r/{subreddit} '{keyword}' page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    return results


def main() -> int:
    print(f"\n=== Reddit collection @ {datetime.now(timezone.utc).isoformat()} ===")

    all_rows: list[dict] = []
    targets = [
        (sub, kw, vendor)
        for sub in SUBREDDITS
        for vendor, kws in VENDOR_KEYWORDS.items()
        for kw in kws
    ]
    print(f"Targets: {len(targets)} (subreddit × keyword)")

    for i, (sub, kw, vendor) in enumerate(targets, 1):
        posts = search_subreddit(sub, kw)
        for p in posts:
            p["vendor"] = vendor
        all_rows.extend(posts)
        total_score = sum(p["score"] for p in posts)
        print(
            f"  [{i:>3}/{len(targets)}] r/{sub:<20} \"{kw:<16}\" "
            f"-> {len(posts):>3} posts  score_sum={total_score:>5}"
        )
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        print("\nNo data collected.")
        return 1

    df = pd.DataFrame(all_rows)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["fetched_at"] = datetime.now(timezone.utc)

    # Deduplicate: same post can match multiple keywords
    df = df.drop_duplicates(subset=["post_id", "keyword"])

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(df)} rows to {OUT_PATH}")
    print(f"   unique posts: {df['post_id'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
