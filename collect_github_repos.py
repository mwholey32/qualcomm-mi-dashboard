"""
collect_github_repos.py — Track on-device AI toolchain repos across vendors.

Usage:
    python collect_github_repos.py

Output:
    data/github_repo_stats.parquet    one row per repo — stars, forks, issues
    data/github_issue_topics.parquet  one row per issue — with keyword tags
    data/github_stargazers.parquet    one row per star event — starred_at timestamp
    data/github_forks.parquet         one row per fork event — forked_at timestamp

Why this matters:
    HF Hub model counts tell you about *supply*. But the real developer-intent
    signal is toolchain engagement — developers starring, forking, and filing
    issues on the SDKs they use to compile, quantize, and deploy models.

    Comparing repo health across vendors (Qualcomm vs Apple vs Google vs Intel
    vs NVIDIA vs Meta) reveals who has the most active developer ecosystem,
    and keyword analysis on issues shows *what stage* developers are at:
    quantization/compilation issues = production-grade optimization work.
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

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "qualcomm-mi-dashboard",
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

REQUEST_DELAY = 2.0 if TOKEN else 6.5

MAX_STARGAZER_PAGES = 50   # 5,000 stars cap per repo
MAX_FORK_PAGES = 30        # 3,000 forks cap per repo

# ---------------------------------------------------------------------------
# Repos to track — vendor-tagged for cross-vendor comparison
# ---------------------------------------------------------------------------

REPOS: list[dict[str, str]] = [
    # Qualcomm — on-device compilation & quantization
    {"repo": "quic/aimet",                    "vendor": "Qualcomm", "role": "quantization toolkit",      "affinity": "native"},
    {"repo": "quic/ai-hub-models",            "vendor": "Qualcomm", "role": "model zoo + compilation",   "affinity": "native"},
    {"repo": "quic/efficient-transformers",    "vendor": "Qualcomm", "role": "optimized inference",       "affinity": "native"},
    # Apple — on-device ML
    {"repo": "apple/coremltools",             "vendor": "Apple",    "role": "model converter / compiler", "affinity": "native"},
    {"repo": "ml-explore/mlx",                "vendor": "Apple",    "role": "ML framework (Apple Silicon)", "affinity": "native"},
    # Google — edge AI
    {"repo": "google-ai-edge/ai-edge-torch",  "vendor": "Google",   "role": "PyTorch-to-TFLite compiler", "affinity": "abstracted"},
    {"repo": "google-ai-edge/mediapipe",      "vendor": "Google",   "role": "on-device ML pipelines",    "affinity": "abstracted"},
    {"repo": "google-ai-edge/LiteRT",         "vendor": "Google",   "role": "TFLite / LiteRT runtime",   "affinity": "abstracted"},
    {"repo": "jax-ml/jax",                    "vendor": "Google",   "role": "array + autodiff / XLA",    "affinity": "abstracted"},
    # Intel — edge / hybrid
    {"repo": "openvinotoolkit/openvino",      "vendor": "Intel",    "role": "inference engine (CPU/GPU/NPU)", "affinity": "native"},
    {"repo": "intel/neural-compressor",       "vendor": "Intel",    "role": "quantization toolkit",      "affinity": "native"},
    {"repo": "huggingface/optimum-intel",     "vendor": "Intel",    "role": "HF Optimum OpenVINO/IPEX",  "affinity": "native"},
    {"repo": "intel/intel-extension-for-pytorch", "vendor": "Intel", "role": "IPEX CPU/GPU extension",   "affinity": "native"},
    # Meta — cross-platform on-device
    {"repo": "pytorch/executorch",            "vendor": "Meta",     "role": "on-device PyTorch runtime", "affinity": "abstracted"},
    # Microsoft — ONNX / DirectML ecosystem
    {"repo": "microsoft/onnxruntime",         "vendor": "Microsoft", "role": "cross-platform inference runtime", "affinity": "abstracted"},
    {"repo": "microsoft/olive",               "vendor": "Microsoft", "role": "model optimization pipeline",     "affinity": "abstracted"},
    {"repo": "microsoft/DirectML",            "vendor": "Microsoft", "role": "DirectML hardware abstraction",   "affinity": "abstracted"},
    # NVIDIA — datacenter (reference point)
    {"repo": "NVIDIA/TensorRT",              "vendor": "NVIDIA",   "role": "inference optimizer",        "affinity": "native"},
    {"repo": "NVIDIA/TensorRT-LLM",          "vendor": "NVIDIA",   "role": "LLM inference engine",      "affinity": "native"},
    {"repo": "triton-inference-server/server", "vendor": "NVIDIA",  "role": "inference serving platform", "affinity": "native"},
    {"repo": "NVIDIA/cuda-python",            "vendor": "NVIDIA",   "role": "CUDA Python bindings",      "affinity": "native"},
    # Qualcomm (Edge Impulse acquisition) — edge ML tooling
    {"repo": "edgeimpulse/inferencing-sdk-cpp",              "vendor": "Qualcomm", "role": "edge inference SDK",        "affinity": "mixed"},
    {"repo": "edgeimpulse/linux-sdk-python",                 "vendor": "Qualcomm", "role": "Linux ML runner",           "affinity": "mixed"},
    {"repo": "edgeimpulse/courseware-embedded-machine-learning", "vendor": "Qualcomm", "role": "ML education / courseware", "affinity": "mixed"},
    # Arduino (separate vendor — Qualcomm subsidiary)
    {"repo": "arduino/Arduino",       "vendor": "Arduino", "role": "Arduino IDE 1.x (legacy)", "affinity": "mixed"},
    {"repo": "arduino/arduino-cli",   "vendor": "Arduino", "role": "CLI toolchain",            "affinity": "mixed"},
    {"repo": "arduino/arduino-ide",   "vendor": "Arduino", "role": "Arduino IDE 2.x",          "affinity": "mixed"},
]

# Keywords that indicate production-grade deployment work (not just demos)
PRODUCTION_KEYWORDS: dict[str, list[str]] = {
    "quantization": ["quantiz", "int4", "int8", "fp16", "mixed precision", "ptq", "qat", "adaround"],
    "compilation": ["compile", "qnn", "htp", "hexagon", "onnx export", "tflite", "qairt"],
    "on-device": ["on-device", "on device", "mobile", "snapdragon", "edge", "latency", "npu"],
    "layer_fusion": ["layer fusion", "fuse", "graph optim"],
    "model_bring_up": ["custom model", "byom", "bring your own", "my model", "own model", "fine-tun"],
    "production": ["production", "deploy", "inference speed", "throughput", "benchmark", "profil"],
}


def fetch_repo_stats(repo: str, vendor: str = "", role: str = "", affinity: str = "native") -> dict | None:
    """Fetch basic repo metadata from the GitHub API."""
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {repo}")
            return None
        d = r.json()
        return {
            "repo": repo,
            "vendor": vendor,
            "role": role,
            "affinity": affinity,
            "stars": d.get("stargazers_count", 0),
            "forks": d.get("forks_count", 0),
            "open_issues": d.get("open_issues_count", 0),
            "watchers": d.get("subscribers_count", 0),
            "language": d.get("language"),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
            "pushed_at": d.get("pushed_at"),
            "description": d.get("description"),
            "fetched_at": datetime.now(timezone.utc),
        }
    except Exception as e:
        print(f"  ERROR {repo}: {e}")
        return None


def fetch_issues(repo: str, vendor: str = "", affinity: str = "native", max_pages: int = 5) -> list[dict]:
    """Fetch recent issues (including PRs) and tag them with topic keywords."""
    items: list[dict] = []

    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/issues",
                params={
                    "state": "all",
                    "sort": "created",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
                headers=HEADERS,
                timeout=20,
            )

            if r.status_code == 403:
                reset = r.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(int(reset) - int(time.time()), 1)
                    print(f"    Rate limited, sleeping {wait}s")
                    time.sleep(wait + 1)
                    continue
                break

            if r.status_code != 200:
                print(f"    HTTP {r.status_code} for {repo} issues page {page}")
                break

            data = r.json()
            if not data:
                break

            for item in data:
                title = (item.get("title") or "").lower()
                body = (item.get("body") or "").lower()
                text = f"{title} {body}"
                labels = [l["name"] for l in item.get("labels", [])]
                is_pr = "pull_request" in item

                # Tag with production keywords
                matched_topics = []
                for topic, keywords in PRODUCTION_KEYWORDS.items():
                    if any(kw in text for kw in keywords):
                        matched_topics.append(topic)

                items.append({
                    "repo": repo,
                    "vendor": vendor,
                    "affinity": affinity,
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "is_pr": is_pr,
                    "labels": "; ".join(labels),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "comments": item.get("comments", 0),
                    "topics": "; ".join(matched_topics) if matched_topics else "",
                    "topic_count": len(matched_topics),
                    "url": item.get("html_url"),
                })

        except Exception as e:
            print(f"    ERROR {repo} issues page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    return items


def fetch_stargazers(repo: str, vendor: str = "", affinity: str = "native") -> list[dict]:
    """Fetch stargazer events with timestamps (starred_at)."""
    items: list[dict] = []
    star_headers = {**HEADERS, "Accept": "application/vnd.github.star+json"}

    for page in range(1, MAX_STARGAZER_PAGES + 1):
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/stargazers",
                params={"per_page": 100, "page": page},
                headers=star_headers,
                timeout=20,
            )

            if r.status_code == 403:
                reset = r.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(int(reset) - int(time.time()), 1)
                    print(f"    Rate limited, sleeping {wait}s")
                    time.sleep(wait + 1)
                    continue
                break

            if r.status_code != 200:
                print(f"    HTTP {r.status_code} for {repo} stargazers page {page}")
                break

            data = r.json()
            if not data:
                break

            for item in data:
                items.append({
                    "repo": repo,
                    "vendor": vendor,
                    "affinity": affinity,
                    "starred_at": item.get("starred_at"),
                })

        except Exception as e:
            print(f"    ERROR {repo} stargazers page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    return items


def fetch_forks(repo: str, vendor: str = "", affinity: str = "native") -> list[dict]:
    """Fetch fork creation dates (oldest first)."""
    items: list[dict] = []

    for page in range(1, MAX_FORK_PAGES + 1):
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/forks",
                params={"sort": "oldest", "per_page": 100, "page": page},
                headers=HEADERS,
                timeout=20,
            )

            if r.status_code == 403:
                reset = r.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(int(reset) - int(time.time()), 1)
                    print(f"    Rate limited, sleeping {wait}s")
                    time.sleep(wait + 1)
                    continue
                break

            if r.status_code != 200:
                print(f"    HTTP {r.status_code} for {repo} forks page {page}")
                break

            data = r.json()
            if not data:
                break

            for item in data:
                owner = item.get("owner") or {}
                items.append({
                    "repo": repo,
                    "vendor": vendor,
                    "affinity": affinity,
                    "forked_at": item.get("created_at"),
                    "owner_type": owner.get("type", ""),       # "User" or "Organization"
                    "owner_login": owner.get("login", ""),
                })

        except Exception as e:
            print(f"    ERROR {repo} forks page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    return items


def main() -> int:
    print(f"\n=== GitHub repo telemetry @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Auth: {'token' if TOKEN else 'unauthenticated'}")
    print(f"Repos: {len(REPOS)}\n")

    # ── Repo stats ────────────────────────────────────────────────────────
    stat_rows = []
    for entry in REPOS:
        repo, vendor, role = entry["repo"], entry["vendor"], entry["role"]
        affinity = entry.get("affinity", "native")
        stats = fetch_repo_stats(repo, vendor=vendor, role=role, affinity=affinity)
        if stats:
            stat_rows.append(stats)
            print(f"  [{vendor:<10}] {repo:<40} stars={stats['stars']:<6}  forks={stats['forks']:<5}  issues={stats['open_issues']}")
        else:
            print(f"  [{vendor:<10}] {repo:<40} FAILED")
        time.sleep(REQUEST_DELAY)

    if stat_rows:
        stats_df = pd.DataFrame(stat_rows)
        out_stats = DATA_DIR / "github_repo_stats.parquet"
        # Append for time-series tracking
        if out_stats.exists():
            existing = pd.read_parquet(out_stats)
            stats_df = pd.concat([existing, stats_df], ignore_index=True)
        stats_df.to_parquet(out_stats, index=False)
        print(f"\n-> saved repo stats to {out_stats} ({len(stats_df)} total rows)")

    # ── Issues with topic tagging ─────────────────────────────────────────
    print("\nFetching issues...")
    all_issues: list[dict] = []
    for entry in REPOS:
        repo, vendor = entry["repo"], entry["vendor"]
        affinity = entry.get("affinity", "native")
        issues = fetch_issues(repo, vendor=vendor, affinity=affinity)
        all_issues.extend(issues)
        tagged = sum(1 for i in issues if i["topic_count"] > 0)
        print(f"  [{vendor:<10}] {repo:<40} {len(issues)} issues, {tagged} with production keywords")
        time.sleep(REQUEST_DELAY)

    if all_issues:
        issues_df = pd.DataFrame(all_issues)
        issues_df["created_at"] = pd.to_datetime(issues_df["created_at"], utc=True)
        issues_df = issues_df.drop_duplicates(subset=["repo", "number"])
        out_issues = DATA_DIR / "github_issue_topics.parquet"
        issues_df.to_parquet(out_issues, index=False)
        print(f"-> saved {len(issues_df)} issues to {out_issues}")

        # Per-vendor summary
        print("\nProduction-keyword issues by vendor:")
        for vendor in sorted(issues_df["vendor"].unique()):
            v_df = issues_df[issues_df["vendor"] == vendor]
            tagged = int((v_df["topic_count"] > 0).sum())
            print(f"  {vendor}: {len(v_df)} total, {tagged} with production keywords")
    else:
        print("\nNo issues collected.")

    # ── Stargazers & forks history ────────────────────────────────────────
    print("\nFetching stargazer & fork history...")
    all_stargazers: list[dict] = []
    all_forks: list[dict] = []
    for entry in REPOS:
        repo, vendor = entry["repo"], entry["vendor"]
        affinity = entry.get("affinity", "native")
        stars = fetch_stargazers(repo, vendor=vendor, affinity=affinity)
        forks = fetch_forks(repo, vendor=vendor, affinity=affinity)
        all_stargazers.extend(stars)
        all_forks.extend(forks)
        print(f"  [{vendor:<10}] {repo:<40} {len(stars)} stargazers, {len(forks)} forks")

    if all_stargazers:
        sg_df = pd.DataFrame(all_stargazers)
        sg_df["starred_at"] = pd.to_datetime(sg_df["starred_at"], utc=True)
        out_sg = DATA_DIR / "github_stargazers.parquet"
        sg_df.to_parquet(out_sg, index=False)
        print(f"-> saved {len(sg_df)} stargazer events to {out_sg}")
    else:
        print("\nNo stargazer data collected.")

    if all_forks:
        fk_df = pd.DataFrame(all_forks)
        fk_df["forked_at"] = pd.to_datetime(fk_df["forked_at"], utc=True)
        out_fk = DATA_DIR / "github_forks.parquet"
        fk_df.to_parquet(out_fk, index=False)
        print(f"-> saved {len(fk_df)} fork events to {out_fk}")
    else:
        print("\nNo fork data collected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
