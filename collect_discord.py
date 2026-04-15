"""
collect_discord.py — Snapshot Discord server member counts for AI/ML communities.

Usage:
    python collect_discord.py

Output:
    data/discord_members.parquet   one row per server per snapshot

Why Discord:
    Joining a Discord server is an act of developer intent — you join because
    you're actively building with (or evaluating) a toolchain. Member counts
    are publicly available via the invite API with no authentication, and the
    online/total ratio gives a crude engagement signal.

    Tracking these weekly gives a growth-rate comparison across vendor
    ecosystems that slots into Stage 3 (Developer Intent) alongside PyPI.

Note:
    - Qualcomm has no public Discord server as of this collection date.
      That's itself a competitive gap signal.
    - The invite API returns `approximate_member_count` — Discord rounds
      these, so treat as directional, not exact.
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
OUT_PATH = DATA_DIR / "discord_members.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}


@dataclass(frozen=True)
class DiscordServer:
    invite_code: str
    display_name: str
    vendor: str
    category: str          # on_device | datacenter | cross_platform | community
    note: str = ""


# ---------------------------------------------------------------------------
# Curated server list. Expand as needed — each row is one Discord API call.
#
# Invite codes verified against public invite pages. Discord occasionally
# expires invites, so some may 404 — the collector skips gracefully.
# ---------------------------------------------------------------------------
SERVERS: list[DiscordServer] = [
    # ── On-device / edge AI ecosystems ────────────────────────────────────
    DiscordServer(
        invite_code="hugging-face-879548962464493619",
        display_name="Hugging Face",
        vendor="HuggingFace",
        category="cross_platform",
        note="Central ML model hub community; hardware-agnostic but heavy on-device discussion",
    ),
    DiscordServer(
        invite_code="comfyorg",
        display_name="ComfyUI",
        vendor="community",
        category="on_device",
        note="Image-gen workflow tool; users care deeply about GPU/NPU performance",
    ),
    DiscordServer(
        invite_code="ZKz3xgEKhv",
        display_name="MLX Community",
        vendor="Apple",
        category="on_device",
        note="Apple's on-device ML framework community — direct Qualcomm competitor signal",
    ),
    DiscordServer(
        invite_code="ollama",
        display_name="Ollama",
        vendor="community",
        category="on_device",
        note="Local LLM runner — the consumer face of on-device inference",
    ),
    DiscordServer(
        invite_code="lmstudio",
        display_name="LM Studio",
        vendor="community",
        category="on_device",
        note="Desktop local LLM app — strong on-device intent signal",
    ),
    DiscordServer(
        invite_code="intel",
        display_name="Intel Insiders (incl. OpenVINO)",
        vendor="Intel",
        category="on_device",
        note="Intel's developer community; includes OpenVINO edge inference channels",
    ),

    # ── Datacenter / GPU ecosystems ───────────────────────────────────────
    DiscordServer(
        invite_code="stablediffusion",
        display_name="Stable Diffusion",
        vendor="Stability AI",
        category="cross_platform",
        note="Image gen community; spans cloud + local inference",
    ),
    DiscordServer(
        invite_code="nvidiadeveloper",
        display_name="NVIDIA Developer",
        vendor="NVIDIA",
        category="datacenter",
        note="NVIDIA's official developer community",
    ),

    # Note: Qualcomm has no public Discord server.
    # AMD, PyTorch, vLLM, LocalLLaMA — no working invite codes found.
]


def fetch_server(server: DiscordServer) -> dict | None:
    """Fetch member counts from the Discord invite API. No auth needed."""
    url = f"https://discord.com/api/invites/{server.invite_code}?with_counts=true"
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        d = r.json()
        guild = d.get("guild", {})
        return {
            "invite_code": server.invite_code,
            "guild_id": guild.get("id"),
            "guild_name": guild.get("name"),
            "display_name": server.display_name,
            "vendor": server.vendor,
            "category": server.category,
            "note": server.note,
            "approximate_member_count": d.get("approximate_member_count"),
            "approximate_presence_count": d.get("approximate_presence_count"),
            "fetched_at": datetime.now(timezone.utc),
        }
    except Exception as e:
        print(f"    ERROR {server.display_name}: {e}")
        return None


def main() -> int:
    print(f"\n=== Discord collection @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Servers: {len(SERVERS)}")

    rows: list[dict] = []
    ok = fail = 0
    for i, s in enumerate(SERVERS, 1):
        row = fetch_server(s)
        if row:
            ok += 1
            members = row["approximate_member_count"] or 0
            online = row["approximate_presence_count"] or 0
            pct = f"{online / members * 100:.0f}%" if members else "?"
            print(
                f"  [{i:>2}/{len(SERVERS)}] {s.display_name:<35} "
                f"members={members:>9,}  online={online:>7,}  ({pct})"
            )
            rows.append(row)
        else:
            fail += 1
            print(f"  [{i:>2}/{len(SERVERS)}] {s.display_name:<35} NOT FOUND / ERROR")
        time.sleep(0.5)

    if not rows:
        print("\nNo data collected.")
        return 1

    new_df = pd.DataFrame(rows)

    # Append to existing file so we build a time series across runs.
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(new_df)} new rows to {OUT_PATH} ({len(combined)} total rows)")
    print(f"   (ok={ok}, fail={fail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
