"""
products.py — Silicon-product taxonomy and classifier.

Goal: take a DataFrame of HF models (vendor_models.parquet) plus an optional
DataFrame of model cards (model_cards.parquet) and produce a long-form table
mapping each model to one or more silicon products.

Output schema:
    repo_id | vendor | family | product | match_source

`match_source` describes how the assignment was made:
    'name'     — matched by repo_id pattern (fast, no README needed)
    'readme'   — matched by README regex or extracted chipset key
    'default'  — vendor-level default (e.g. NVIDIA has no chip hint → "Data Center")

The taxonomy is intentionally pragmatic: one well-defined `product` per
generation-ish, grouped into `family` for the UI hierarchy. It is easy to edit
below without touching the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Rule records
# ---------------------------------------------------------------------------

@dataclass
class ProductRule:
    vendor: str
    family: str
    product: str
    # Regex patterns (case-insensitive) matched against repo_id (e.g. "amd/foo-ryzenai-hybrid-bar")
    name_patterns: list[str] = field(default_factory=list)
    # Regex patterns matched against README text
    readme_patterns: list[str] = field(default_factory=list)
    # Regex patterns matched against the semicolon-joined tag list
    tag_patterns: list[str] = field(default_factory=list)
    # Exact library_name match (e.g. "coreml", "mlx")
    library_names: list[str] = field(default_factory=list)
    # Exact Qualcomm chipset keys (from the README chipset table)
    chipset_keys: list[str] = field(default_factory=list)
    # Optional Qualcomm chipset-key prefix match (e.g. "qcs9")
    chipset_prefixes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Qualcomm — the hero vendor for this project
# ---------------------------------------------------------------------------
# Chipset keys come directly from the README chipset tables on AI Hub models.
# Prefixes cover unknown future parts without us having to fabricate codenames.

QUALCOMM_RULES: list[ProductRule] = [
    # Mobile premium — Snapdragon 8 series
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Premium)",
        product="Snapdragon 8 Elite Gen 5",
        chipset_keys=["snapdragon_8_elite_gen5"],
        readme_patterns=[r"snapdragon[^\w]*8\s*elite\s*gen\s*5"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Premium)",
        product="Snapdragon 8 Elite (for Galaxy)",
        chipset_keys=["snapdragon_8_elite_for_galaxy"],
        readme_patterns=[r"snapdragon[^\w]*8\s*elite\s*for\s*galaxy"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Premium)",
        product="Snapdragon 8 Elite",
        chipset_keys=["snapdragon_8_elite", "snapdragon_8elite"],
        # 'Snapdragon 8 Elite' but NOT 'Snapdragon 8 Elite Gen 5' / 'For Galaxy'
        readme_patterns=[r"snapdragon[^\w]*8\s*elite(?!\s*(?:gen\s*5|for\s*galaxy))"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Premium)",
        product="Snapdragon 8 Gen 3",
        chipset_keys=["snapdragon_8gen3", "snapdragon_8_gen_3"],
        readme_patterns=[r"snapdragon[^\w]*8\s*gen\s*3"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Premium)",
        product="Snapdragon 8 Gen 2",
        chipset_keys=["snapdragon_8gen2", "snapdragon_8_gen_2"],
        readme_patterns=[r"snapdragon[^\w]*8\s*gen\s*2"],
    ),
    # Mobile mid — 7 series
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Mid)",
        product="Snapdragon 7 Gen 4",
        chipset_keys=["snapdragon_7gen4", "snapdragon_7_gen_4"],
        readme_patterns=[r"snapdragon[^\w]*7\s*gen\s*4"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Mobile (Mid)",
        product="Snapdragon 7 Gen 3",
        chipset_keys=["snapdragon_7gen3", "snapdragon_7_gen_3"],
        readme_patterns=[r"snapdragon[^\w]*7\s*gen\s*3"],
    ),
    # PC / Compute — X series
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Compute (PC)",
        product="Snapdragon X2 Elite",
        chipset_keys=["snapdragon_x2_elite"],
        readme_patterns=[r"snapdragon[^\w]*x2\s*elite"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Compute (PC)",
        product="Snapdragon X Elite",
        chipset_keys=["snapdragon_x_elite", "snapdragon_xelite"],
        # X Elite but NOT X2 Elite
        readme_patterns=[r"snapdragon[^\w]*x\s+elite(?!.*x2)"],
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Compute (PC)",
        product="Snapdragon X Plus",
        chipset_keys=["snapdragon_x_plus", "snapdragon_xplus"],
        readme_patterns=[r"snapdragon[^\w]*x\s+plus"],
    ),
    # Automotive — Ride / Flex / Cockpit (SA-series codenames)
    ProductRule(
        vendor="Qualcomm",
        family="Snapdragon Automotive",
        product="Automotive (SA-series)",
        chipset_prefixes=["sa"],   # sa8775p, sa8650p, sa8295p, sa7255p, etc.
    ),
    # IoT / Robotics — QCS / QCM series
    ProductRule(
        vendor="Qualcomm",
        family="Qualcomm IoT / Robotics",
        product="QCS IoT (Premium)",
        chipset_prefixes=["qcs"],  # qcs8550, qcs9075, qcs6490, ...
    ),
    ProductRule(
        vendor="Qualcomm",
        family="Qualcomm IoT / Robotics",
        product="QCM IoT",
        chipset_prefixes=["qcm"],  # qcm6690, qcm6490, ...
    ),
    # Cloud
    ProductRule(
        vendor="Qualcomm",
        family="Qualcomm Cloud",
        product="Cloud AI 100",
        name_patterns=[r"cloud[-_]?ai[-_]?100"],
        readme_patterns=[r"cloud\s*ai\s*100"],
    ),
    # Edge Impulse (Qualcomm acquisition)
    ProductRule(
        vendor="Qualcomm",
        family="Qualcomm Edge Impulse",
        product="Edge Impulse Studio",
        name_patterns=[r"edgeimpulse", r"edge.impulse"],
    ),
]


# ---------------------------------------------------------------------------
# Non-Qualcomm vendors — name-pattern based (no README required)
# ---------------------------------------------------------------------------

OTHER_RULES: list[ProductRule] = [
    # -------- AMD --------
    ProductRule(
        vendor="AMD",
        family="Ryzen AI (Client NPU)",
        product="Ryzen AI Hybrid (NPU+iGPU)",
        name_patterns=[r"(?:^|[-_/])(?:ryzenai|rai)[-_]", r"hybrid[-_]npu|npu[-_]hybrid"],
        tag_patterns=[r"ryzenai[-_]hybrid"],
        readme_patterns=[r"ryzen\s*ai\s*hybrid"],
    ),
    ProductRule(
        vendor="AMD",
        family="Ryzen AI (Client NPU)",
        product="Ryzen AI NPU-only",
        tag_patterns=[r"ryzenai[-_]npu\b"],
        readme_patterns=[r"xdna\s*npu"],
    ),
    ProductRule(
        vendor="AMD",
        family="Ryzen AI (Client NPU)",
        product="Ryzen AI (generic)",
        tag_patterns=[r"(?:^|;)(?:ryzenai|RyzenAI)(?:;|$)"],
    ),
    ProductRule(
        vendor="AMD",
        family="Instinct (Data Center)",
        product="Instinct MI300 / MI325 / MI350",
        name_patterns=[r"mi300|mi325|mi350|mi355"],
        readme_patterns=[r"\bmi300\b|\bmi325\b|\bmi350\b"],
    ),
    ProductRule(
        vendor="AMD",
        family="Instinct (Data Center)",
        product="ROCm (generic)",
        name_patterns=[r"rocm"],
        readme_patterns=[r"\brocm\b"],
    ),

    # -------- Apple --------
    ProductRule(
        vendor="Apple",
        family="Apple Silicon",
        product="MLX (Apple Silicon native)",
        name_patterns=[r"(?:^|[-_/])mlx(?:[-_]|$)"],
        tag_patterns=[r"(?:^|;)mlx(?:;|$)"],
        library_names=["mlx"],
    ),
    ProductRule(
        vendor="Apple",
        family="Apple Silicon",
        product="Core ML (Neural Engine)",
        name_patterns=[r"coreml|core[-_]ml"],
        library_names=["coreml"],
        readme_patterns=[r"core\s*ml|neural\s*engine"],
    ),
    ProductRule(
        vendor="Apple",
        family="Apple Vision",
        product="MobileCLIP / FastVLM / CLIP",
        name_patterns=[r"mobileclip|fastvlm|aimv2"],
        library_names=["mobileclip", "ml-fastvlm", "open_clip", "tic-clip"],
    ),

    # -------- NVIDIA --------
    ProductRule(
        vendor="NVIDIA",
        family="NVIDIA Data Center",
        product="Hopper / Blackwell (H100 / H200 / B100 / B200)",
        name_patterns=[r"\b(?:h100|h200|b100|b200|gb200|hopper|blackwell)\b"],
        readme_patterns=[r"\b(?:h100|h200|b100|b200|gb200|hopper|blackwell)\b"],
    ),
    ProductRule(
        vendor="NVIDIA",
        family="NVIDIA Data Center",
        product="Ampere (A100 / A10)",
        name_patterns=[r"\ba100\b|\ba10\b|\bampere\b"],
        readme_patterns=[r"\ba100\b|\ba10\b|\bampere\b"],
    ),
    ProductRule(
        vendor="NVIDIA",
        family="NVIDIA Edge / Robotics",
        product="Jetson (Orin / Thor)",
        name_patterns=[r"jetson|orin|thor|isaac"],
        readme_patterns=[r"\bjetson\b|\borin\b|\bthor\b"],
    ),
    ProductRule(
        vendor="NVIDIA",
        family="NVIDIA Research / Foundation",
        product="Nemotron",
        name_patterns=[r"nemotron"],
    ),
    ProductRule(
        vendor="NVIDIA",
        family="NVIDIA Research / Foundation",
        product="Cosmos (World models)",
        name_patterns=[r"cosmos"],
    ),

    # -------- Intel --------
    ProductRule(
        vendor="Intel",
        family="Intel Client NPU",
        product="Core Ultra (Meteor/Lunar/Arrow Lake)",
        name_patterns=[r"core[-_]?ultra|meteor[-_]?lake|lunar[-_]?lake|arrow[-_]?lake"],
        readme_patterns=[r"core\s*ultra|meteor\s*lake|lunar\s*lake|arrow\s*lake"],
    ),
    ProductRule(
        vendor="Intel",
        family="Intel Data Center AI",
        product="Gaudi (2 / 3)",
        name_patterns=[r"gaudi"],
        readme_patterns=[r"\bgaudi\b"],
    ),
    ProductRule(
        vendor="Intel",
        family="Intel Data Center AI",
        product="Xeon (CPU inference)",
        name_patterns=[r"xeon"],
        readme_patterns=[r"\bxeon\b"],
    ),
    ProductRule(
        vendor="Intel",
        family="Intel GPU",
        product="Arc / Max (GPU)",
        name_patterns=[r"\barc\b|\bxmx\b"],
        readme_patterns=[r"intel\s*arc\b|ponte\s*vecchio"],
    ),
    # Catch-all: Intel Neural Compressor (autoround) quantized variants target Intel silicon generically
    ProductRule(
        vendor="Intel",
        family="Intel (Quantized / OpenVINO)",
        product="OpenVINO / Neural Compressor",
        name_patterns=[r"autoround|openvino|(?:^|[-_/])inc(?:[-_]|$)|int4|int8"],
        tag_patterns=[r"openvino"],
    ),

    # -------- Google --------
    ProductRule(
        vendor="Google",
        family="Google TPU",
        product="TPU (v4 / v5 / Trillium)",
        name_patterns=[r"\btpu\b|trillium|v5e|v5p"],
        readme_patterns=[r"\btpu\b|trillium"],
    ),
    ProductRule(
        vendor="Google",
        family="Google Edge / Mobile",
        product="Gemma (Open weights)",
        name_patterns=[r"gemma"],
    ),
    ProductRule(
        vendor="Google",
        family="Google Edge / Mobile",
        product="PaliGemma (VLM)",
        name_patterns=[r"paligemma"],
    ),

    # -------- Microsoft --------
    ProductRule(
        vendor="Microsoft",
        family="Microsoft Client / Copilot+ PC",
        product="Phi (SLM)",
        name_patterns=[r"\bphi\b"],
    ),
    ProductRule(
        vendor="Microsoft",
        family="Microsoft Client / Copilot+ PC",
        product="ONNX Runtime / DirectML",
        name_patterns=[r"\bonnx\b|directml"],
        readme_patterns=[r"directml|onnx\s*runtime"],
    ),

    # -------- Arm --------
    ProductRule(
        vendor="Arm",
        family="Arm (IP)",
        product="Ethos-U / Cortex / Kleidi",
        name_patterns=[r"ethos|cortex[-_]?[am]|kleidi|executorch"],
        readme_patterns=[r"\bethos[-_]u\b|kleidi"],
    ),
]


ARDUINO_RULES: list[ProductRule] = [
    ProductRule(
        vendor="Arduino",
        family="Arduino Boards",
        product="Arduino UNO Q",
        name_patterns=[r"uno.q", r"arduino.*qualcomm"],
        readme_patterns=[r"uno\s*q", r"dragonwing"],
    ),
    ProductRule(
        vendor="Arduino",
        family="Arduino Boards",
        product="Arduino VENTUNO Q",
        name_patterns=[r"ventuno"],
        readme_patterns=[r"ventuno"],
    ),
]

ALL_RULES: list[ProductRule] = QUALCOMM_RULES + OTHER_RULES + ARDUINO_RULES


# ---------------------------------------------------------------------------
# Qualcomm chipset-key extraction (from README chipset table)
# ---------------------------------------------------------------------------

# Matches `qualcomm_snapdragon_x_elite` style keys in download URLs.
_QUALCOMM_UNDERSCORE_RE = re.compile(r"qualcomm_([a-z0-9_]+?)(?=[\s\"|,)<>/])", re.IGNORECASE)

# Matches bare QCS / QCM / SA / SM codenames in narrative tables.
# e.g. "Qualcomm® QCS8550", "SA8775P", "QCM6690"
_QUALCOMM_BARE_CODE_RE = re.compile(r"\b(qcs\d+\w*|qcm\d+\w*|sa\d+p?\b|sm\d+[a-z]*\b)", re.IGNORECASE)


def extract_qualcomm_chipset_keys(readme_text: str) -> set[str]:
    """Return the set of unique chipset codenames referenced in a Qualcomm README.

    Handles both forms used on AI Hub model cards:
    (1) Download-URL underscored keys:  `qualcomm_snapdragon_x_elite`
    (2) Narrative bare codenames:       `QCS8550`, `SA8775P`, `QCM6690`
    """
    if not readme_text:
        return set()
    cleaned: set[str] = set()

    for k in _QUALCOMM_UNDERSCORE_RE.findall(readme_text):
        k = k.lower().strip("_")
        k = re.sub(r"_proxy$", "", k)
        cleaned.add(k)

    for k in _QUALCOMM_BARE_CODE_RE.findall(readme_text):
        cleaned.add(k.lower())

    return cleaned


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

def _vendor_from_source(source: str) -> str:
    """Convert 'vendor:qualcomm' into display name 'Qualcomm'."""
    raw = source.replace("vendor:", "") if isinstance(source, str) else ""
    return {
        "qualcomm": "Qualcomm",
        "nvidia": "NVIDIA",
        "apple": "Apple",
        "Intel": "Intel",
        "amd": "AMD",
        "google": "Google",
        "microsoft": "Microsoft",
        "Arm": "Arm",
        "arduino": "Arduino",
        "edgeimpulse": "Qualcomm",
    }.get(raw, raw.title())


def _match_name(rule: ProductRule, repo_id: str) -> bool:
    if not rule.name_patterns:
        return False
    return any(re.search(p, repo_id, re.IGNORECASE) for p in rule.name_patterns)


def _match_readme(rule: ProductRule, readme: str) -> bool:
    if not rule.readme_patterns or not readme:
        return False
    return any(re.search(p, readme, re.IGNORECASE) for p in rule.readme_patterns)


def _match_tags(rule: ProductRule, tags: str) -> bool:
    if not rule.tag_patterns or not tags:
        return False
    return any(re.search(p, tags, re.IGNORECASE) for p in rule.tag_patterns)


def _match_library(rule: ProductRule, library_name) -> bool:
    if not rule.library_names:
        return False
    if library_name is None or (isinstance(library_name, float) and pd.isna(library_name)):
        return False
    return str(library_name).lower() in [x.lower() for x in rule.library_names]


def _match_qualcomm_chipsets(rule: ProductRule, keys: set[str]) -> bool:
    if rule.chipset_keys and any(k in keys for k in rule.chipset_keys):
        return True
    if rule.chipset_prefixes:
        return any(any(k.startswith(pref) for k in keys) for pref in rule.chipset_prefixes)
    return False


def assign_products(
    vendor_df: pd.DataFrame,
    cards_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Produce a long-form table of (repo_id, vendor, family, product, match_source).
    A single model can map to multiple rows if it matches several products.
    Unmatched models receive a single row with family='(Unclassified)' so they
    remain visible in the UI.
    """
    if vendor_df.empty:
        return pd.DataFrame(
            columns=["repo_id", "vendor", "family", "product", "match_source"]
        )

    cards_lookup: dict[str, str] = {}
    if cards_df is not None and not cards_df.empty:
        cards_lookup = dict(zip(cards_df["repo_id"], cards_df["readme_text"]))

    rows: list[dict] = []

    for _, m in vendor_df.iterrows():
        repo_id = m["repo_id"]
        vendor = _vendor_from_source(m.get("source", ""))
        readme = cards_lookup.get(repo_id, "")
        tags = m.get("tags") or ""
        library_name = m.get("library_name")
        chipset_keys = (
            extract_qualcomm_chipset_keys(readme) if vendor == "Qualcomm" else set()
        )

        matched_any = False
        for rule in ALL_RULES:
            if rule.vendor != vendor:
                continue

            hit = False
            source_label = ""
            if _match_name(rule, repo_id):
                hit = True
                source_label = "name"
            elif _match_library(rule, library_name):
                hit = True
                source_label = "library"
            elif _match_tags(rule, tags):
                hit = True
                source_label = "tag"
            elif vendor == "Qualcomm" and _match_qualcomm_chipsets(rule, chipset_keys):
                hit = True
                source_label = "readme"
            elif _match_readme(rule, readme):
                hit = True
                source_label = "readme"

            if hit:
                rows.append({
                    "repo_id": repo_id,
                    "vendor": vendor,
                    "family": rule.family,
                    "product": rule.product,
                    "match_source": source_label,
                })
                matched_any = True

        if not matched_any:
            rows.append({
                "repo_id": repo_id,
                "vendor": vendor,
                "family": "(Unclassified)",
                "product": "(Unclassified)",
                "match_source": "none",
            })

    return pd.DataFrame(rows)


def product_rollup(long_df: pd.DataFrame, vendor_df: pd.DataFrame) -> pd.DataFrame:
    """Join long-form assignments back to model metrics and roll up per product."""
    if long_df.empty:
        return pd.DataFrame()
    keep = vendor_df[["repo_id", "downloads", "likes", "created_at", "last_modified"]]
    merged = long_df.merge(keep, on="repo_id", how="left")
    now = pd.Timestamp.now(tz="UTC")
    merged["days_since_created"] = (
        now - pd.to_datetime(merged["created_at"], utc=True, errors="coerce")
    ).dt.days

    out = (
        merged.groupby(["vendor", "family", "product"], dropna=False)
        .agg(
            models=("repo_id", "nunique"),
            downloads=("downloads", "sum"),
            likes=("likes", "sum"),
            median_age_days=("days_since_created", "median"),
        )
        .reset_index()
        .sort_values(["vendor", "downloads"], ascending=[True, False])
    )
    return out
