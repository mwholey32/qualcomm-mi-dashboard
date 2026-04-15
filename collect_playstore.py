"""
collect_playstore.py — Scrape Google Play Store metadata for a curated list
of Android apps that use on-device AI inference.

Usage:
    python collect_playstore.py

Output:
    data/playstore_apps.parquet

Notes on what this signal can and cannot do:

WHAT IT CAN:
  - Install range (e.g. "1,000,000,000+") — Google's own reported bucket.
  - Raw install count (realInstalls) — sometimes None, region-dependent.
  - Average star rating and rating count.
  - Last updated timestamp — useful as a proxy for active development.
  - App description — can be scanned for AI-feature language.

WHAT IT CANNOT:
  - iOS installs (Apple doesn't publish).
  - Tell you *which* inference backend actually runs inside the app
    (NNAPI vs TFLite vs QNN vs custom). The `backend_attribution` column
    below is my best-effort note from public docs / press coverage and
    should be treated as a starting point, not ground truth.
  - Tell you how many users actually exercise the AI features vs just
    install the app.
  - Measure how often the on-device model is invoked per user.

For a production MI view you would cross-reference this with Sensor Tower /
data.ai paid feeds, engineering blogs, and teardown reports.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from google_play_scraper import app as gp_app
    from google_play_scraper.exceptions import NotFoundError
except ImportError:
    print("Missing dependency: pip install google-play-scraper")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "playstore_apps.parquet"


@dataclass(frozen=True)
class TargetApp:
    package: str
    display_name: str
    vendor: str
    category: str
    backend_attribution: str   # my attribution; not returned by Play Store
    ai_features: str           # my attribution; not returned by Play Store


# Curated target list. Expand as needed — each row is one Play Store fetch.
# Package IDs verified against public Play Store URLs as of this writing,
# but Google occasionally retires or renames apps so some may 404.
TARGETS: list[TargetApp] = [
    # ── Google apps — heaviest known on-device AI users on Android ─────────
    TargetApp(
        package="com.google.android.apps.photos",
        display_name="Google Photos",
        vendor="Google",
        category="mass_market",
        backend_attribution="TFLite + NNAPI / QNN HTP delegate on Snapdragon",
        ai_features="Magic Editor, Best Take, Audio Magic Eraser, on-device face grouping",
    ),
    TargetApp(
        package="com.google.android.apps.translate",
        display_name="Google Translate",
        vendor="Google",
        category="mass_market",
        backend_attribution="TFLite + NNAPI on-device language models",
        ai_features="Offline translation packs, camera translate, conversation mode",
    ),
    TargetApp(
        package="com.google.android.googlequicksearchbox",
        display_name="Google app (incl. Lens)",
        vendor="Google",
        category="mass_market",
        backend_attribution="TFLite + NNAPI",
        ai_features="Google Lens visual search, on-device OCR",
    ),
    TargetApp(
        package="com.google.android.apps.bard",
        display_name="Google Gemini",
        vendor="Google",
        category="ai_assistant",
        backend_attribution="Gemini Nano on-device (Pixel 8+) + cloud hybrid",
        ai_features="On-device LLM on Pixel 8/9, cloud fallback elsewhere",
    ),

    # ── Social / creator apps — heavy AR + on-device vision ───────────────
    TargetApp(
        package="com.snapchat.android",
        display_name="Snapchat",
        vendor="Snap",
        category="creator",
        backend_attribution="Custom SNPE-lineage runtime; NNAPI / QNN on Android",
        ai_features="AR Lenses, My AI, real-time face tracking",
    ),
    TargetApp(
        package="com.zhiliaoapp.musically",
        display_name="TikTok",
        vendor="ByteDance",
        category="creator",
        backend_attribution="NNAPI + custom mobile runtime",
        ai_features="On-device effects, AI avatars, beautify",
    ),
    TargetApp(
        package="com.instagram.android",
        display_name="Instagram",
        vendor="Meta",
        category="creator",
        backend_attribution="PyTorch Mobile / ExecuTorch across Meta apps",
        ai_features="Reels effects, AR filters, background segmentation",
    ),
    TargetApp(
        package="com.facebook.katana",
        display_name="Facebook",
        vendor="Meta",
        category="creator",
        backend_attribution="PyTorch Mobile / ExecuTorch",
        ai_features="Feed ranking, on-device image classification",
    ),

    # ── AI assistants — mostly cloud, included for reference ──────────────
    TargetApp(
        package="com.openai.chatgpt",
        display_name="ChatGPT",
        vendor="OpenAI",
        category="ai_assistant",
        backend_attribution="Primarily cloud; voice mode uses on-device VAD",
        ai_features="Cloud LLM, on-device wake/voice handling",
    ),
    TargetApp(
        package="com.microsoft.copilot",
        display_name="Microsoft Copilot",
        vendor="Microsoft",
        category="ai_assistant",
        backend_attribution="Cloud LLM; Copilot+ PC features are Windows-side",
        ai_features="Cloud chat + image generation on mobile",
    ),

    # ── Samsung / OEM system apps ─────────────────────────────────────────
    TargetApp(
        package="com.samsung.android.bixby.agent",
        display_name="Samsung Bixby",
        vendor="Samsung",
        category="oem_system",
        backend_attribution="Samsung ENN / NNAPI — Hexagon on Snapdragon for Galaxy",
        ai_features="On-device voice wake + command recognition",
    ),

    # ── Pure on-device / offline LLM apps (small installs = pure intent) ──
    TargetApp(
        package="ai.mlc.mlcchat",
        display_name="MLC Chat",
        vendor="MLC AI",
        category="offline_llm",
        backend_attribution="MLC TVM runtime — direct Adreno / Hexagon targeting",
        ai_features="Pure offline LLM (Llama, Phi, Gemma) on phone",
    ),
    TargetApp(
        package="com.pocketpalai",
        display_name="PocketPal AI",
        vendor="community",
        category="offline_llm",
        backend_attribution="llama.cpp on Android",
        ai_features="Offline Llama / Mistral / Gemma",
    ),
]


def fetch(target: TargetApp) -> dict | None:
    """Fetch one app's Play Store metadata. None on 404 / error."""
    try:
        result = gp_app(target.package, lang="en", country="us")
    except NotFoundError:
        return None
    except Exception as e:
        print(f"    ERROR {target.package}: {e}")
        return None

    updated_ts = result.get("updated")
    updated_dt = (
        datetime.fromtimestamp(updated_ts, tz=timezone.utc) if updated_ts else None
    )
    return {
        "package": target.package,
        "display_name": target.display_name,
        "vendor": target.vendor,
        "category": target.category,
        "backend_attribution": target.backend_attribution,
        "ai_features": target.ai_features,
        # Fields returned by the Play Store
        "play_title": result.get("title"),
        "play_developer": result.get("developer"),
        "play_genre": result.get("genre"),
        "play_score": result.get("score"),
        "play_ratings": result.get("ratings"),
        "play_installs_str": result.get("installs"),
        "play_min_installs": result.get("minInstalls"),
        "play_real_installs": result.get("realInstalls"),
        "play_last_updated": updated_dt,
        "play_description_snippet": (result.get("description") or "")[:600],
        "play_url": result.get("url"),
        "fetched_at": datetime.now(timezone.utc),
    }


def main() -> int:
    print(f"\n=== Play Store collection @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Targets: {len(TARGETS)}")

    rows: list[dict] = []
    ok = fail = 0
    for i, t in enumerate(TARGETS, 1):
        row = fetch(t)
        if row:
            ok += 1
            print(
                f"  [{i:>2}/{len(TARGETS)}] {t.package:<45} "
                f"{row.get('play_installs_str') or '?':<18} "
                f"score={row.get('play_score') or '?'}"
            )
            rows.append(row)
        else:
            fail += 1
            print(f"  [{i:>2}/{len(TARGETS)}] {t.package:<45} NOT FOUND / ERROR")
        time.sleep(0.4)

    if not rows:
        print("\nNo data collected.")
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(df)} rows to {OUT_PATH}  (ok={ok}, fail={fail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
