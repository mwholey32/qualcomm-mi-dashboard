"""
collect_github_discussions.py — Track vendor mentions in on-device AI repos.

Usage:
    python collect_github_discussions.py

Output:
    data/github_mentions.parquet   one row per issue/discussion per keyword

Why GitHub issues + discussions:
    When a developer files an issue mentioning "Qualcomm" or "Snapdragon" in
    llama.cpp, that's a concrete signal: someone tried to run an LLM on
    Qualcomm hardware and hit something worth reporting. Comparing vendor
    mention counts across repos gives a share-of-voice metric for which
    hardware vendors are getting real developer attention in the on-device
    AI ecosystem.

    The GitHub search API is free (unauthenticated: 10 req/min, authenticated:
    30 req/min) and returns total_count + paginated results with timestamps.

Limitations:
    - Unauthenticated rate limit: 10 requests/minute. We pace at 6s/request.
    - GitHub search indexes issue/PR/discussion titles + bodies, not comments.
    - total_count may be approximate for large result sets.
    - Without a token, we get 10 req/min. Set GITHUB_TOKEN env var for 30/min.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "github_mentions.parquet"

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "qualcomm-mi-dashboard",
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

# Pacing: 6s unauthenticated (10 req/min), 2.5s authenticated (30 req/min)
REQUEST_DELAY = 2.5 if TOKEN else 6.5


# ---------------------------------------------------------------------------
# Curated repos and keywords
# ---------------------------------------------------------------------------

REPOS = [
    "ggml-org/llama.cpp",
    "pytorch/executorch",
    "microsoft/onnxruntime",
    "openvinotoolkit/openvino",
    "ml-explore/mlx",
    "google-ai-edge/ai-edge-torch",
]

VENDOR_KEYWORDS: dict[str, list[str]] = {
    "Qualcomm":  ["qualcomm", "snapdragon", "hexagon"],
    "NVIDIA":    ["nvidia", "cuda"],
    "Apple":     ["apple", "coreml"],
    "Intel":     ["intel", "openvino"],
    "AMD":       ["amd", "rocm"],
}


def search_repo(repo: str, keyword: str, max_pages: int = 2) -> tuple[int, list[dict]]:
    """Search issues + discussions in a repo for a keyword.

    Returns (total_count, list_of_items).
    """
    items: list[dict] = []
    total_count = 0

    for page in range(1, max_pages + 1):
        url = "https://api.github.com/search/issues"
        params = {
            "q": f"{keyword} repo:{repo}",
            "sort": "created",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }

        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)

            if r.status_code == 403:
                reset = r.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(int(reset) - int(time.time()), 1)
                    print(f"    Rate limited, sleeping {wait}s")
                    time.sleep(wait + 1)
                    continue
                break

            if r.status_code != 200:
                print(f"    HTTP {r.status_code} for {repo} '{keyword}'")
                break

            data = r.json()
            total_count = data.get("total_count", 0)

            for item in data.get("items", []):
                items.append({
                    "repo": repo,
                    "keyword": keyword,
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "is_pr": "pull_request" in item,
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "comments": item.get("comments", 0),
                    "url": item.get("html_url"),
                })

            # Stop if we got all results
            if len(items) >= total_count:
                break

        except Exception as e:
            print(f"    ERROR {repo} '{keyword}' page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    return total_count, items


def main() -> int:
    print(f"\n=== GitHub mentions collection @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Auth: {'token' if TOKEN else 'unauthenticated (10 req/min)'}")

    all_rows: list[dict] = []
    count_rows: list[dict] = []

    targets = [
        (repo, kw, vendor)
        for repo in REPOS
        for vendor, kws in VENDOR_KEYWORDS.items()
        for kw in kws
    ]
    print(f"Targets: {len(targets)} (repo × keyword)\n")

    for i, (repo, kw, vendor) in enumerate(targets, 1):
        total, items = search_repo(repo, kw)

        # Store the total count (for the bar chart — doesn't need all items)
        count_rows.append({
            "repo": repo,
            "keyword": kw,
            "vendor": vendor,
            "total_count": total,
            "fetched_at": datetime.now(timezone.utc),
        })

        for item in items:
            item["vendor"] = vendor

        all_rows.extend(items)

        print(
            f"  [{i:>3}/{len(targets)}] {repo:<35} \"{kw:<12}\" "
            f"-> {total:>5} total  (fetched {len(items)})"
        )
        time.sleep(REQUEST_DELAY)

    if not count_rows:
        print("\nNo data collected.")
        return 1

    # Save detailed items
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.drop_duplicates(subset=["repo", "number", "keyword"])
    df.to_parquet(OUT_PATH, index=False)

    # Also save the counts summary for fast loading in the dashboard
    counts_df = pd.DataFrame(count_rows)
    counts_df.to_parquet(DATA_DIR / "github_mention_counts.parquet", index=False)

    print(f"\n-> saved {len(df)} detail rows to {OUT_PATH}")
    print(f"-> saved {len(counts_df)} count rows to data/github_mention_counts.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
