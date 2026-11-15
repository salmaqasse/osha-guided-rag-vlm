"""
OSHA Construction Safety Screening Pipeline — v9.7 (cue-preservation and PPE trigger repair)
(Image-only batch runner; user-facing reports + developer audit trail)

Architecture
------------
Pass 1: Image-only sparse visual evidence card, including worker-linked PPE.
Pass 2: Two evidence-bounded OSHA gates per supported hazard category, reconciled
        into one locked category gate.
Pass 3: Text-only severity, probability, control, and deterministic risk-matrix
        report for each locked hazard gate.

This version intentionally has NO Pass 4. It keeps the tested Pass 1–3 model
logic, loads the model only once, and separates concise user-facing reports from
full developer/audit JSON outputs. It also records retrieval queries and FAISS
similarity scores for rule-grounding analysis.

Scope: Fall, Struck-by / Equipment, Caught-in/Between, Electrical /
Electrocution, and PPE Compliance Concern. Housekeeping is excluded.

This is a research screening tool, not an OSHA inspection, citation, or legal
finding.
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: qwen-vl-utils.\n"
        "Install it in your vlm_safety environment with:\n"
        "  pip install qwen-vl-utils"
    ) from exc


# ── Command line ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="OSHA construction-safety screening — Qwen2.5-VL-7B, 4-bit NF4",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  One image:
    python testF_vlm_v9_3.py --image image.jpg

  Folder of images:
    python testF_vlm_v9_3.py --image-dir ./Surya-Photos --dataset "Dr. Ram dataset"

  A text file with one image path per line:
    python testF_vlm_v9_3.py --image-list images.txt --output-dir results_run1

Notes:
  - The model is loaded once and reused for every image in a batch.
  - Model loading time is reported separately from per-image pipeline runtime.
  - Use --allow-download only when the model is not already cached locally.
""",
)

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--image", help="Path to one image.")
input_group.add_argument("--image-dir", help="Directory containing image files.")
input_group.add_argument("--image-list", help="Text file with one image path per line.")

parser.add_argument("--dataset", default="Not specified", help="Dataset name for metadata.")
parser.add_argument("--ground-truth", default=None, help="Optional common ground-truth label for this run.")
parser.add_argument("--split", default="Not specified", help="Dataset split for metadata.")
parser.add_argument(
    "--model-id",
    default="Qwen/Qwen2.5-VL-7B-Instruct",
    help="Hugging Face model identifier.",
)
parser.add_argument(
    "--allow-download",
    action="store_true",
    help="Allow Hugging Face downloads if the model or processor is not cached locally.",
)
parser.add_argument(
    "--run-label",
    default="qwen2_5_vl_7b_4bit_nf4_multihazard_v9_7_cuepreserve",
    help="Safe label appended to output filenames.",
)
parser.add_argument(
    "--rules-file",
    default="osha_rules.json",
    help="Path to the validated OSHA rules JSON file.",
)
parser.add_argument(
    "--output-dir",
    default="output",
    help="Directory for per-image outputs and the batch summary.",
)
parser.add_argument(
    "--output-format",
    choices=("json", "html", "md", "all"),
    default="all",
    help="Developer/audit JSON is always written. HTML and Markdown are concise user-facing reports.",
)
parser.add_argument(
    "--ppe-focus",
    action="store_true",
    help=(
        "Record this run as PPE-focused. PPE is always assessed per worker from "
        "the full Pass 1 image; this flag does not suppress other hazards."
    ),
)
parser.add_argument(
    "--pass1-max-tokens",
    type=int,
    default=1200,
    help="Maximum new tokens for the Pass 1 evidence card (default: 1200).",
)
parser.add_argument(
    "--rules-per-category",
    type=int,
    choices=(1, 2, 3),
    default=2,
    help="Retrieved OSHA rules supplied to each category gate (default: 2).",
)
parser.add_argument(
    "--no-cache-clear",
    action="store_true",
    help="Do not clear unused CUDA cache between batch images.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print raw Pass 1/Pass 2 evidence and the full final result for debugging.",
)
args = parser.parse_args()


# ── Input discovery ───────────────────────────────────────────────────────────
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def resolve_input_images() -> list[str]:
    if args.image:
        candidates = [Path(args.image)]
    elif args.image_dir:
        folder = Path(args.image_dir)
        if not folder.is_dir():
            parser.error(f"Image directory not found: {folder}")
        candidates = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    else:
        list_path = Path(args.image_list)
        if not list_path.is_file():
            parser.error(f"Image-list file not found: {list_path}")
        candidates = []
        for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
            item = raw_line.strip()
            if not item or item.startswith("#"):
                continue
            image_path = Path(item)
            if not image_path.is_absolute():
                image_path = (list_path.parent / image_path).resolve()
            candidates.append(image_path)

    if not candidates:
        parser.error("No supported images were found.")

    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:10])
        more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        parser.error(f"These image paths do not exist:\n{preview}{more}")

    unsupported = [str(path) for path in candidates if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES]
    if unsupported:
        preview = "\n".join(f"  - {item}" for item in unsupported[:10])
        parser.error(
            "Unsupported image extension(s). Supported: "
            f"{', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}\n{preview}"
        )

    # Preserve input order for --image-list, but remove exact duplicate paths.
    unique: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


INPUT_IMAGE_PATHS = resolve_input_images()


# ── Experiment configuration ─────────────────────────────────────────────────
IMAGE_PATH = INPUT_IMAGE_PATHS[0]  # Updated inside run_one_image().
DATASET = args.dataset
GROUND_TRUTH = args.ground_truth
SPLIT = args.split
MODEL_ID = args.model_id
RUN_LABEL = re.sub(r"[^A-Za-z0-9._-]+", "_", args.run_label).strip("._-") or "qwen_7b_4bit_nf4"
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FORMAT = args.output_format
CLEAR_CACHE_BETWEEN_IMAGES = not args.no_cache_clear

_dataset_text = str(DATASET).lower()
PPE_FOCUS = bool(
    args.ppe_focus
    or ("constructionsite10k" in _dataset_text and "ppe" in _dataset_text)
)
PASS1_MAX_TOKENS = max(900, int(args.pass1_max_tokens))
RULES_PER_CATEGORY = int(args.rules_per_category)

# Structured audit data is written into each per-image JSON output. It lets the
# batch analysis count preserved, contextual, and dropped Struck-by cues instead
# of relying only on terminal messages.
GUARD_DIAGNOSTICS: list[dict[str, Any]] = []

RULES_PATH = str(Path(args.rules_file))
INDEX_PATH = "osha_rules.faiss"
KEYS_PATH = "osha_rules_keys.json"
HASH_PATH = "osha_rules.hash"
TOP_K = 3
QUANTIZATION = "4-bit NF4 (bitsandbytes, double quantization, FP16 compute)"

VALID_CATEGORIES = [
    "Fall Hazards",
    "Struck-by / Equipment Hazards",
    "Caught-in/Between Hazards",
    "Electrical / Electrocution Hazards",
    "PPE Compliance Concern",
]
VALID_CATEGORY_SET = set(VALID_CATEGORIES)
CATEGORY_TEXT = "; ".join(VALID_CATEGORIES)

FALL_CUE_TERMS = (
    "scaffold", "scaffolding", "ladder", "platform", "roof", "edge",
    "opening", "elevated", "height", "drop", "guardrail", "fall arrest",
    "harness", "lanyard",
    # Explicit edge/drop wording for excavations. "Trench" or "excavation" alone
    # is intentionally not included: a worker inside a trench remains a
    # Caught-in/Between concern unless a separate fall-edge exposure is described.
    "open excavation", "excavation edge", "edge of excavation", "excavation rim",
    "trench edge", "edge of trench", "trench rim", "open trench edge",
    "unprotected edge", "unprotected opening", "open pit", "pit edge",
)
FALL_CONTROL_TERMS = (
    "guardrail", "edge protection", "fall protection", "fall arrest",
    "harness", "lanyard", "safety net",
)

ELECTRICAL_CONTEXT_TERMS = (
    "electrical panel", "electric panel", "panelboard", "circuit breaker",
    "electrical box", "switchboard", "wiring", "wire", "cable", "conduit",
    "power line", "power lines", "overhead line", "overhead lines",
    "utility pole", "utility lines", "transformer", "substation",
    "electrical equipment", "electrical component", "electrical components",
)

# ── Struck-by / Caught-in mechanism matching ───────────────────────────
#
# A generic equipment noun is not an exposure mechanism.  Struck-by is retained
# only when the text contains either (a) a direct worker-exposure phrase, or
# (b) both a dynamic/load mechanism and a worker-exposure relation.

# Direct phrases already bind a worker to an impact, load, vehicle, or swing risk.
STRUCK_BY_DIRECT_EXPOSURE_TERMS = (
    "worker under a suspended load",
    "worker under suspended load",
    "worker under a raised load",
    "worker under raised load",
    "worker under a raised bucket",
    "worker below a raised bucket",
    "worker beneath a raised bucket",
    "worker below a raised load",
    "worker beneath a raised load",
    "bucket above worker",
    "bucket over worker",
    "arm above worker",
    "boom above worker",
    "load above worker",
    "load over worker",
    "suspended load near worker",
    "swinging load near worker",
    "raised forks near worker",
    "forklift load near worker",
    "vehicle moving toward worker",
    "vehicle moving towards worker",
    "vehicle reversing toward worker",
    "truck reversing toward worker",
    "vehicle approaching worker",
    "equipment approaching worker",
    "worker in the vehicle path",
    "worker in the travel path",
    "worker in the swing zone",
    "worker in the swing path",
    "worker within the swing zone",
    "worker within the swing radius",
    "worker in the discharge zone",
)

# These must be paired with a worker-exposure relation by struck_by_mechanism_signal.
STRUCK_BY_DYNAMIC_MECHANISM_TERMS = (
    # Raised / suspended / falling loads.
    "raised bucket",
    "raised excavator arm",
    "raised arm",
    "raised boom",
    "raised load",
    "suspended load",
    "swinging load",
    "hoisted load",
    "crane load",
    "load being lifted",
    "load being lowered",
    "lifting operation",
    "hoisting operation",
    "falling object",
    "falling material",
    "falling debris",
    "objects falling",
    "material falling",
    "debris falling",
    # Vehicle / equipment movement.
    "backing vehicle",
    "vehicle reversing",
    "reversing vehicle",
    "truck reversing",
    "truck backing",
    "vehicle backing",
    "moving vehicle",
    "moving equipment",
    "vehicle approaching",
    "equipment approaching",
    # Swing / rotation / discharge.
    "excavator swing",
    "swing path",
    "swing zone",
    "swing radius",
    "rotation zone",
    "rotating arm",
    "rotating boom",
    "machine swing",
    "loading operation",
    "unloading operation",
    "discharge zone",
    # Forklift exposure only when paired with worker relation.
    "raised forks",
    "forks raised",
    "forklift load",
)

STRUCK_BY_WORKER_EXPOSURE_TERMS = (
    "worker near",
    "workers near",
    "worker next to",
    "workers next to",
    "worker beside",
    "workers beside",
    "worker under",
    "workers under",
    "worker below",
    "workers below",
    "worker beneath",
    "workers beneath",
    "worker in the path",
    "workers in the path",
    "worker in a path",
    "worker in the travel path",
    "workers in the travel path",
    "worker in the swing",
    "workers in the swing",
    "worker within the swing",
    "workers within the swing",
    "near the worker",
    "near a worker",
    "beside the worker",
    "beside a worker",
    "next to the worker",
    "next to a worker",
    "above the worker",
    "above a worker",
    "over the worker",
    "over a worker",
    "toward the worker",
    "towards the worker",
    "toward a worker",
    "towards a worker",
    "approaching the worker",
    "approaching a worker",
    "in the worker's path",
)

# Pass 1 can refer to a person in many natural ways.  Keep this list in one
# place so worker-linked mechanisms do not vanish because the VLM used
# "operator", "employee", or "laborer" instead of the word "worker".
PERSON_NOUN_PATTERN = (
    r"(?:worker|workers|person|people|employee|employees|operator|operators|"
    r"laborer|laborers|labourer|labourers|crew\s+member(?:s)?|individual|"
    r"individuals|man|men|woman|women)"
)

# Regex patterns allow natural VLM phrasing such as "operator standing near a
# raised excavator arm" without accepting equipment presence by itself.
STRUCK_BY_WORKER_EXPOSURE_PATTERNS = (
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,6}}\s+(?:near|next\s+to|beside|under|below|beneath)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,6}}\s+(?:in|within)\s+(?:the|a)?\s*(?:vehicle\s+path|travel\s+path|swing\s+(?:zone|path|radius)|discharge\s+zone)\b",
    rf"\b(?:near|next\s+to|beside|under|below|beneath|above|over|toward|towards|approaching)\s+(?:a|an|the)?\s*{PERSON_NOUN_PATTERN}\b",
    rf"\b(?:vehicle|truck|equipment)\b(?:\s+\w+){{0,5}}\s+(?:toward|towards|approaching)\s+(?:a|an|the)?\s*{PERSON_NOUN_PATTERN}\b",
)

# Caught-in/Between is restricted to crush/trap/entanglement mechanisms or an
# explicit worker-inside-excavation signal. Background trench wording alone is
# deliberately excluded.
CAUGHT_CRUSH_ENTRAPMENT_TERMS = (
    "pinch point",
    "crush point",
    "crushing",
    "crushed",
    "trapped",
    "trapped between",
    "pinned",
    "pinned between",
    "between equipment and",
    "between a vehicle and",
    "between two objects",
    "between moving parts",
    "entangled",
    "entanglement",
    "conveyor",
    "roller",
    "under equipment being repaired",
    "maintenance under raised equipment",
    "between raised equipment",
    "blocking or cribbing",
    "cave-in",
    "cave in",
)

WORKER_INSIDE_EXCAVATION_PATTERNS = (
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,8}}\s+(?:inside|within)\s+(?:a|an|the)?\s*(?:trench|trenches|excavation|excavations)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,8}}\s+in\s+(?:a|an|the)?\s*(?:trench|trenches|excavation|excavations)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,10}}\s+(?:standing|working|located|positioned|operating|visible|seen)\b(?:\s+\w+){{0,7}}\s+(?:in|inside|within|at\s+the\s+bottom\s+of)\s+(?:a|an|the)?\s*(?:trench|trenches|excavation|excavations)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,8}}\s+below\s+(?:ground(?:\s+level)?|grade)\b",
    rf"\b(?:trench|trenches|excavation|excavations)\b(?:\s+\w+){{0,10}}\s+(?:contains|with|holding)\s+(?:a|an|the)?\s*{PERSON_NOUN_PATTERN}\b",
    rf"\b(?:trench|trenches|excavation|excavations)[-\s]+{PERSON_NOUN_PATTERN}\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,6}}\s+(?:at\s+the\s+bottom\s+of|down\s+in)\s+(?:a|an|the)?\s*(?:trench|trenches|excavation|excavations)\b",
)

ELECTRICAL_UNCERTAINTY = (
    "The image does not establish energized status, voltage, clearance, isolation, "
    "grounding, lockout/tagout status, or electrical controls outside the frame."
)

UNSUPPORTED_ELECTRICAL_PATTERNS = (
    r"\blive\s+(?:wire|wires|line|lines|power line|power lines)\b",
    r"\benergized\b",
    r"\bexposed\s+conductors?\b",
    r"\bmissing\s+(?:grounding|ground|safety switch|disconnect|lockout|tagout)\b",
    r"\bno\s+(?:grounding|ground|safety switch|disconnect|lockout|tagout)\b",
)

ELECTRICAL_IRRELEVANT_CONTROL_TERMS = (
    "safety switch", "grounding system", "grounding", "guardrail", "guardrails",
    "safety harness", "harness", "lanyard", "hard hat", "helmet",
    "safety glasses", "glasses", "eye protection",
)

REQUIRED_RULE_FIELDS = {
    "category",
    "official_source",
    "official_source_url",
    "official_text",
    "interpretation",
    "visual_trigger",
    "confirmation_threshold",
    "uncertainty_policy",
    "consistency_rule",
    "source_type",
    "last_verified",
}


# ── Startup checks ────────────────────────────────────────────────────────────
if not os.path.isfile(RULES_PATH):
    raise FileNotFoundError(
        f"Required OSHA rules file not found: {RULES_PATH}\n"
        "Use --rules-file to point to your validated osha_rules.json file."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("The Qwen 7B 4-bit pipeline requires a CUDA-capable GPU.")

print(f"[device] {DEVICE}")
print(f"[model] {MODEL_ID}")
print(f"[quantization] {QUANTIZATION}")
print(f"[run label] {RUN_LABEL}")
print("[pipeline version] v9.8 — gate-integrity repair, concise user report + per-image runtime")
print(f"[inputs] {len(INPUT_IMAGE_PATHS)} image(s) | output directory: {OUTPUT_DIR}")
print(f"[PPE mode] focus={PPE_FOCUS} (per-worker PPE assessment from the full image)")


# ── Small helpers ─────────────────────────────────────────────────────────────
def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def unique_text(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def clean_md(text: str) -> str:
    for pattern in (r"```(?:json)?", r"```", r"\*+", r"#+\s*", r"_{2,}"):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_json(text: str) -> str | None:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else None


def labeled(text: str, label: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^" + re.escape(label), stripped, flags=re.IGNORECASE):
            value = stripped[len(label):].strip(" :.—-\t")
            if value:
                return value
    return ""


def norm_category(value: Any) -> str | None:
    raw = " ".join(str(value or "").strip().lower().split())

    if raw in {"", "none", "not applicable", "no hazard", "no concern"}:
        return None

    # Prefer explicit category names first.
    if "fall" in raw or "scaffold" in raw or "edge" in raw:
        return "Fall Hazards"

    if "electrical" in raw or "electrocution" in raw or "conductor" in raw:
        return "Electrical / Electrocution Hazards"

    if any(token in raw for token in (
        "caught", "pinch", "crush", "entrapment", "trapped"
    )):
        return "Caught-in/Between Hazards"

    if any(token in raw for token in (
        "struck", "suspended load", "falling object",
        "backing vehicle", "vehicle path"
    )):
        return "Struck-by / Equipment Hazards"

    if any(token in raw for token in (
        "ppe", "hard hat", "helmet", "eye", "glove", "respiratory"
    )):
        return "PPE Compliance Concern"

    # Last-resort mappings only.
    if "trench" in raw or "excavation" in raw:
        return "Caught-in/Between Hazards"

    if "vehicle" in raw:
        return "Struck-by / Equipment Hazards"

    return None


def norm_cue_status(value: Any) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    return "Visible" if raw == "visible" or raw.startswith("visible ") else "Unclear"


def norm_control_status(value: Any) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    if raw in {"visible", "present", "in place"}:
        return "Visible"
    if raw in {"not observed", "not visible", "absent", "missing"}:
        return "Not observed"
    if raw in {"not applicable", "n/a", "na"}:
        return "Not applicable"
    return "Unclear"


def text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)

def worker_inside_excavation_signal(text: str) -> bool:
    """Return True only when the text places a worker inside an excavation."""
    lowered = " ".join(str(text or "").lower().split())
    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE) is not None
        for pattern in WORKER_INSIDE_EXCAVATION_PATTERNS
    )


# An excavation itself is not automatically a fall cue. These patterns require
# the generated Pass 1 description to place a worker at an edge, rim, opening,
# or visible drop. The fallback below remains Unclear and therefore routes the
# image to Pass 2 for verification instead of fabricating a confirmed violation.
WORKER_FALL_EDGE_PATTERNS = (
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,10}}\s+"
    r"(?:near|at|on|beside|along|adjacent\s+to|next\s+to)\s+"
    r"(?:the\s+)?(?:open\s+|unprotected\s+)?(?:edge|rim|lip)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,10}}\s+"
    r"(?:near|at|on|beside|along|adjacent\s+to|next\s+to)\s+"
    r"(?:an?\s+)?(?:open\s+|unprotected\s+)?(?:opening|drop)\b",
    rf"\b{PERSON_NOUN_PATTERN}\b(?:\s+\w+){{0,10}}\s+"
    r"(?:near|at|on|beside|along|adjacent\s+to|next\s+to)\s+"
    r"(?:the\s+)?(?:open\s+|unprotected\s+)?(?:excavation|trench)\s+"
    r"(?:edge|rim|lip)\b",
    rf"\b(?:open|unprotected)\s+(?:edge|rim|opening|drop)\b"
    rf"(?:\s+\w+){{0,12}}\s+\b{PERSON_NOUN_PATTERN}\b",
    rf"\b(?:edge|rim|lip)\s+of\s+(?:an?\s+)?(?:excavation|trench|opening|pit)\b"
    rf"(?:\s+\w+){{0,12}}\s+\b{PERSON_NOUN_PATTERN}\b",
)


def worker_fall_edge_signal(text: str) -> bool:
    """Return True only for a worker-linked edge/opening/drop description."""
    lowered = " ".join(str(text or "").lower().split())
    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE) is not None
        for pattern in WORKER_FALL_EDGE_PATTERNS
    )


def _sentence_fragments(text: str) -> list[str]:
    """Split generated descriptions into short chunks for conservative relation checks."""
    return [
        fragment.strip()
        for fragment in re.split(r"[.!?;\n]+", str(text or "").lower())
        if fragment.strip()
    ]


def struck_by_mechanism_level(text: str) -> str:
    """Return direct, contextual, or none for a worker-linked Struck-by mechanism.

    Direct means the old explicit phrase logic succeeded. Contextual is deliberately
    weaker: a worker and dynamic mechanism occur in the same generated sentence with
    a spatial/directional relation or an overhead statement. Contextual cues are
    retained as Unclear for Pass 2 image verification, never promoted by this guard
    to a confirmed finding.
    """
    lowered = " ".join(str(text or "").lower().split())
    direct_exposure = text_has_any(lowered, STRUCK_BY_DIRECT_EXPOSURE_TERMS)
    dynamic_mechanism = text_has_any(lowered, STRUCK_BY_DYNAMIC_MECHANISM_TERMS)
    worker_exposure = (
        text_has_any(lowered, STRUCK_BY_WORKER_EXPOSURE_TERMS)
        or any(
            re.search(pattern, lowered, flags=re.IGNORECASE) is not None
            for pattern in STRUCK_BY_WORKER_EXPOSURE_PATTERNS
        )
    )
    if direct_exposure or (dynamic_mechanism and worker_exposure):
        return "direct"

    worker_pattern = r"\b(?:worker|workers|person|people)\b"
    relation_pattern = (
        r"\b(?:near|nearby|close\s+to|adjacent\s+to|next\s+to|beside|under|below|"
        r"beneath|above|over|overhead|within|in\s+(?:the\s+)?(?:travel\s+path|"
        r"vehicle\s+path|swing\s+(?:zone|path|radius)|discharge\s+zone))\b"
    )
    distance_pattern = r"\b\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?|ft|feet)\s+from\b"
    for fragment in _sentence_fragments(lowered):
        has_worker = re.search(worker_pattern, fragment, flags=re.IGNORECASE) is not None
        has_dynamic = text_has_any(fragment, STRUCK_BY_DYNAMIC_MECHANISM_TERMS)
        if not (has_worker and has_dynamic):
            continue
        has_relation = re.search(relation_pattern, fragment, flags=re.IGNORECASE) is not None
        has_distance = re.search(distance_pattern, fragment, flags=re.IGNORECASE) is not None
        has_overhead = "directly overhead" in fragment or "overhead" in fragment
        if has_relation or has_distance or has_overhead:
            return "contextual"
    return "none"


def struck_by_mechanism_signal(text: str) -> bool:
    """Compatibility helper: any direct or contextual worker-linked mechanism."""
    return struck_by_mechanism_level(text) != "none"


def caught_in_between_mechanism_signal(text: str) -> bool:
    """Require crush/trap evidence or a worker visibly inside an excavation."""
    lowered = " ".join(str(text or "").lower().split())
    return (
        text_has_any(lowered, CAUGHT_CRUSH_ENTRAPMENT_TERMS)
        or worker_inside_excavation_signal(lowered)
    )


def normalise_worker_count(value: Any) -> int | str:
    if isinstance(value, bool):
        return "Unclear"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    raw = str(value or "").strip()
    if re.fullmatch(r"\d+", raw):
        return int(raw)
    return "Unclear"


def reconcile_cue_category(
    category: str,
    scenario: str,
    evidence: str,
    controls: list[dict[str, str]],
    scene_context: str = "",
) -> tuple[str | None, str | None, bool]:
    """Preserve a submitted Pass 1 category and use lexical checks only for certainty.

    Pass 1 already classified the image.  These deterministic checks may identify a
    strong mechanism, but they must not silently erase or reclassify that visual
    observation merely because the VLM used different natural language.  A weak
    lexical match is therefore retained as an Unclear cue for Pass 2 image
    verification.  Scene-wide context is deliberately excluded here to prevent an
    unrelated pole, wire, or object from changing an individual cue's category.
    """
    del controls, scene_context

    cue_text = f"{scenario} {evidence}"
    fall_signal = text_has_any(cue_text, FALL_CUE_TERMS)
    struck_level = struck_by_mechanism_level(cue_text)
    caught_signal = caught_in_between_mechanism_signal(cue_text)

    if category in {"Struck-by / Equipment Hazards", "Caught-in/Between Hazards"}:
        print(
            "[inventory guard signals] "
            f"category={category} | struck_level={struck_level} | "
            f"caught={caught_signal} | fall={fall_signal}"
        )

    if category == "Fall Hazards" and not fall_signal:
        return (
            category,
            "Preserved the submitted Fall cue as Unclear because its wording did not "
            "match the deterministic Fall-term list. Pass 2 must verify the cue "
            "from the image; this guard does not override Pass 1.",
            True,
        )

    if category == "Struck-by / Equipment Hazards":
        if struck_level == "direct":
            return category, None, False
        if struck_level == "contextual":
            return (
                category,
                "Retained a contextual Struck-by cue as Unclear: a person, dynamic "
                "equipment/load, and a spatial or overhead relation were described. "
                "Pass 2 must verify it from the image.",
                True,
            )
        return (
            category,
            "Preserved the submitted Struck-by cue as Unclear because the "
            "deterministic worker-mechanism matcher did not find an exact relation. "
            "Pass 2 must verify the visual cue from the image.",
            True,
        )

    if category == "Caught-in/Between Hazards":
        if caught_signal:
            return category, None, False
        return (
            category,
            "Preserved the submitted Caught-in/Between cue as Unclear because the "
            "deterministic crush/trap matcher did not find an exact phrase. Pass 2 "
            "must verify the visual cue from the image.",
            True,
        )

    return category, None, False

def sanitise_electrical_text(value: Any) -> str:
    """Remove unsupported energisation/control assertions from a text field."""
    cleaned = str(value or "").strip()
    for pattern in UNSUPPORTED_ELECTRICAL_PATTERNS:
        cleaned = re.sub(pattern, "electrical", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bworking on live power lines\b", "working near overhead utility lines", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bworking on electrical lines\b", "working near overhead utility lines", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bexposed electrical components?\b", "electrical components", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bunsafe electrical(?: condition)?\b", "electrical condition requiring verification", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;,")
    return cleaned


def electrical_safe_description(scene: dict[str, Any], fallback: str) -> str:
    """Create conservative text using Pass 1's recorded activity and objects only."""
    activity = str(scene.get("work_activity", "")).strip()
    objects = " ".join(scene.get("key_objects", [])).lower()
    context = f"{activity} {objects}".lower()
    if any(term in context for term in ("power line", "overhead", "utility pole", "bucket truck", "aerial bucket")):
        return "A worker in an aerial bucket is near overhead utility lines and pole-mounted equipment."
    if any(term in context for term in ("panel", "breaker", "wiring", "wire", "electrical box")):
        return "Workers are examining electrical equipment with visible wiring or panel components."
    return sanitise_electrical_text(fallback) or "Electrical equipment or wiring is visible near the worker."


def apply_electrical_inventory_guard(
    scene: dict[str, Any],
    category: str,
    scenario: str,
    evidence: str,
    controls: list[dict[str, str]],
    limitations: list[str],
) -> tuple[str, str, list[dict[str, str]], list[str]]:
    """Make the locked electrical evidence card conservative and internally consistent."""
    if category != "Electrical / Electrocution Hazards":
        return scenario, evidence, controls, limitations

    safe_description = electrical_safe_description(scene, evidence or scenario)
    scenario = safe_description
    evidence = safe_description

    filtered_controls: list[dict[str, str]] = []
    for control in controls:
        name = str(control.get("control", "")).strip()
        if not name:
            continue
        if text_has_any(name, ELECTRICAL_IRRELEVANT_CONTROL_TERMS):
            continue
        filtered_controls.append({"control": name, "status": "Unclear"})
    controls = filtered_controls

    cleaned_limitations = [sanitise_electrical_text(item) for item in limitations]
    irrelevant_limit_terms = (
        "struck", "falling object", "fall protection", "guardrail", "harness",
        "helmet", "hard hat", "worker's head", "worker head",
    )
    cleaned_limitations = [
        item for item in cleaned_limitations
        if item and not text_has_any(item, irrelevant_limit_terms)
    ]
    if ELECTRICAL_UNCERTAINTY not in cleaned_limitations:
        cleaned_limitations.append(ELECTRICAL_UNCERTAINTY)
    return scenario, evidence, controls, unique_text(cleaned_limitations)


def apply_fall_control_guard(
    category: str, scenario: str, evidence: str, controls: list[dict[str, str]], limitations: list[str]
) -> tuple[list[dict[str, str]], list[str]]:
    """Prevent partial scaffold views from becoming false 'missing control' claims."""
    if category != "Fall Hazards":
        return controls, limitations

    combined = f"{scenario} {evidence}".lower()
    scaffold_like = any(term in combined for term in ("scaffold", "scaffolding", "platform"))
    full_edge_established = any(
        phrase in combined
        for phrase in ("full relevant edge", "entire relevant edge", "full edge is visible", "entire edge is visible")
    )
    if not scaffold_like or full_edge_established:
        return controls, limitations

    adjusted: list[dict[str, str]] = []
    changed = False
    for control in controls:
        item = dict(control)
        is_fall_control = text_has_any(item.get("control", ""), FALL_CONTROL_TERMS)
        if is_fall_control and item.get("status") == "Not observed":
            item["status"] = "Unclear"
            changed = True
        adjusted.append(item)

    if changed:
        note = (
            "Scaffold/platform components are visible, but the full relevant edge and "
            "fall-protection coverage cannot be confirmed from this view."
        )
        if note not in limitations:
            limitations = [*limitations, note]
        print("[inventory guard] Scaffold fall-control status changed from Not observed to Unclear.")
    return adjusted, limitations


def norm_screening_status(value: Any) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    if "confirmed" in raw:
        return "Confirmed concern"
    if "possible" in raw or "unclear" in raw:
        return "Possible concern"
    return "No concern visible"


def norm_compliance_status(value: Any, screening_status: str) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    if screening_status == "No concern visible":
        return "No violation visible"
    if screening_status == "Possible concern":
        return "Not assessable"
    if "confirmed" in raw:
        return "Confirmed violation"
    return "Not assessable"


def confirmed_violation_from_compliance(compliance_status: str) -> str:
    return {
        "Confirmed violation": "Yes",
        "Not assessable": "Unclear",
        "No violation visible": "No",
    }.get(compliance_status, "Unclear")


# ── OSHA knowledge base and FAISS ─────────────────────────────────────────────
with open(RULES_PATH, encoding="utf-8") as rules_file:
    OSHA_RULES = json.load(rules_file)

validation_errors: list[str] = []
for rule_name, rule in OSHA_RULES.items():
    missing = REQUIRED_RULE_FIELDS - set(rule.keys())
    if missing:
        validation_errors.append(f"Rule '{rule_name}' missing fields: {sorted(missing)}")
    if rule.get("category") not in VALID_CATEGORY_SET:
        validation_errors.append(
            f"Rule '{rule_name}' has invalid category: {rule.get('category')!r}"
        )
if validation_errors:
    raise ValueError("osha_rules.json validation failed:\n" + "\n".join(validation_errors))
print(f"[rules] Validated {len(OSHA_RULES)} rules across {len(VALID_CATEGORIES)} categories.")


def rules_hash() -> str:
    with open(RULES_PATH, "rb") as source:
        return hashlib.md5(source.read()).hexdigest()


def retrieval_text(rule_name: str, rule: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Rule: {rule_name}",
            f"Category: {rule['category']}",
            f"Citation: {rule['official_source']}",
            f"Visual trigger: {rule['visual_trigger']}",
            f"Interpretation: {rule['interpretation']}",
            f"Confirmation threshold: {rule['confirmation_threshold']}",
            f"Uncertainty policy: {rule['uncertainty_policy']}",
        ]
    )


def format_rule_for_prompt(rule_name: str, rule: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"RULE: {rule_name}",
            f"CATEGORY: {rule['category']}",
            f"CITATION: {rule['official_source']}",
            f"URL: {rule.get('official_source_url', 'See eCFR Title 29')}",
            f"OFFICIAL TEXT: {rule['official_text']}",
            f"INTERPRETATION: {rule['interpretation']}",
            f"VISUAL TRIGGER: {rule['visual_trigger']}",
            f"CONFIRMATION THRESHOLD: {rule['confirmation_threshold']}",
            f"UNCERTAINTY POLICY: {rule['uncertainty_policy']}",
            f"CONSISTENCY RULE: {rule['consistency_rule']}",
        ]
    )


embedder = SentenceTransformer("all-MiniLM-L6-v2")


def build_index() -> tuple[faiss.Index, list[str]]:
    rule_keys = list(OSHA_RULES.keys())
    vectors = embedder.encode(
        [retrieval_text(key, OSHA_RULES[key]) for key in rule_keys],
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    with open(KEYS_PATH, "w", encoding="utf-8") as key_file:
        json.dump(rule_keys, key_file)
    with open(HASH_PATH, "w", encoding="utf-8") as hash_file:
        hash_file.write(rules_hash())
    print("[index] Built and saved.")
    return index, rule_keys


current_rules_hash = rules_hash()
stored_rules_hash = ""
if os.path.exists(HASH_PATH):
    with open(HASH_PATH, encoding="utf-8") as hash_file:
        stored_rules_hash = hash_file.read().strip()

if (
    os.path.exists(INDEX_PATH)
    and os.path.exists(KEYS_PATH)
    and stored_rules_hash == current_rules_hash
):
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(KEYS_PATH, encoding="utf-8") as key_file:
        rule_keys = json.load(key_file)
    print("[index] Loaded from disk (rules unchanged).")
else:
    print("[index] Rules changed or index missing — rebuilding...")
    faiss_index, rule_keys = build_index()


def retrieve_rules(
    query: str,
    allowed_categories: list[str],
    k: int = TOP_K,
) -> tuple[str, list[str], dict[str, float]]:
    """Retrieve category-allowed OSHA rules and retain FAISS similarity scores.

    Scores are saved for developer analysis only. They help distinguish a weak
    retrieval from a later VLM reasoning error during evaluation.
    """
    if not allowed_categories:
        return "No OSHA rule retrieval: Pass 1 recorded no target hazard cues.", [], {}

    vector = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vector)
    similarities, indices = faiss_index.search(vector, len(rule_keys))

    score_by_key: dict[str, float] = {}
    for rank, index_position in enumerate(indices[0]):
        if int(index_position) < 0:
            continue
        rule_key = rule_keys[int(index_position)]
        score_by_key[rule_key] = round(float(similarities[0][rank]), 4)

    selected: list[str] = []
    selected_categories: set[str] = set()
    allowed_set = set(allowed_categories)

    # First preserve coverage: retrieve at least one rule for each detected category.
    for index_position in indices[0]:
        if int(index_position) < 0:
            continue
        rule_key = rule_keys[int(index_position)]
        category = OSHA_RULES[rule_key]["category"]
        if category not in allowed_set or category in selected_categories:
            continue
        selected.append(rule_key)
        selected_categories.add(category)
        if len(selected_categories) == len(allowed_set):
            break

    # Then add the strongest remaining rules within the selected category scope.
    for index_position in indices[0]:
        if len(selected) >= max(k, len(allowed_categories)):
            break
        if int(index_position) < 0:
            continue
        rule_key = rule_keys[int(index_position)]
        category = OSHA_RULES[rule_key]["category"]
        if category not in allowed_set or rule_key in selected:
            continue
        selected.append(rule_key)

    separator = "\n\n" + "=" * 60 + "\n\n"
    rules_text = separator.join(format_rule_for_prompt(key, OSHA_RULES[key]) for key in selected)
    selected_scores = {key: score_by_key.get(key, 0.0) for key in selected}
    return rules_text, selected, selected_scores


# ── Load Qwen 7B in 4-bit NF4 ─────────────────────────────────────────────────
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\nLoading Qwen2.5-VL-7B-Instruct in 4-bit NF4 (bitsandbytes)...")
cuda_sync()
model_load_start = time.perf_counter()
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    local_files_only=not args.allow_download,
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=512 * 28 * 28,
    max_pixels=1024 * 28 * 28,
    local_files_only=not args.allow_download,
)
model.eval()
cuda_sync()
model_load_seconds = time.perf_counter() - model_load_start
model_gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
model_gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
model_load_peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
try:
    model_memory_footprint_gb = model.get_memory_footprint() / (1024 ** 3)
except (AttributeError, TypeError):
    model_memory_footprint_gb = None

print(f"Model loaded in {model_load_seconds:.2f}s")
print(f"[gpu] Model allocated: {model_gpu_allocated_gb:.2f} GB")
print(f"[gpu] Model reserved:  {model_gpu_reserved_gb:.2f} GB")
print(f"[gpu] Load peak:       {model_load_peak_gpu_gb:.2f} GB")
if model_memory_footprint_gb is not None:
    print(f"[model] Reported footprint: {model_memory_footprint_gb:.2f} GB")

torch.cuda.reset_peak_memory_stats()
pipeline_start = time.perf_counter()


# ── VLM helpers ───────────────────────────────────────────────────────────────
def generate(inputs: Any, max_new_tokens: int) -> str:
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    trimmed_ids = [
        generated[len(input_ids):]
        for input_ids, generated in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def infer_text(prompt: str, max_tokens: int = 420) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[template], padding=True, return_tensors="pt").to(DEVICE)
    return generate(inputs, max_tokens)


def infer_image(image_path: str, prompt: str, max_tokens: int = 650) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[template],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    return generate(inputs, max_tokens)


# ── Pass 1 schema normalisation ───────────────────────────────────────────────
def normalise_controls(value: Any) -> list[dict[str, str]]:
    controls: list[dict[str, str]] = []
    for control in value if isinstance(value, list) else []:
        if not isinstance(control, dict):
            continue
        name = str(control.get("control", "")).strip()
        if not name:
            continue
        controls.append(
            {
                "control": name,
                "status": norm_control_status(control.get("status")),
            }
        )
    return controls


def _compact_status_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def norm_ppe_item_status(value: Any) -> str:
    """Normalise a generic PPE status without changing an Unclear observation into absence."""
    raw = _compact_status_text(value)

    # Negative wording must be checked before positive wording: e.g. "not visible"
    # contains the token "visible".
    if raw in {"not applicable", "n/a", "na"}:
        return "Not applicable"
    if (
        raw in {"not observed", "absent", "missing", "not wearing", "not visible", "no"}
        or raw.startswith("not observed")
        or raw.startswith("not visible")
        or raw.startswith("without ")
        or raw.startswith("no ")
        or " absent" in raw
        or " missing" in raw
    ):
        return "Not observed"
    if (
        raw in {"visible", "present", "wearing", "worn", "yes"}
        or raw.startswith("visible")
        or raw.startswith("present")
        or raw.startswith("wearing")
        or raw.startswith("worn")
    ):
        return "Visible"
    return "Unclear"


def norm_hard_hat_status(value: Any, evidence: Any = "") -> str:
    """Normalise hard-hat status, including clear non-hard-hat headwear wording.

    This is deliberately worker-linked: it only affects the hard-hat field and
    cannot turn a cap mention into a missing vest or eye-protection finding.
    """
    raw = _compact_status_text(value)
    evidence_text = _compact_status_text(evidence)
    combined = f"{raw} {evidence_text}".strip()

    explicit_absence = (
        "no hard hat",
        "no helmet",
        "without a hard hat",
        "without hard hat",
        "without a helmet",
        "not wearing a hard hat",
        "not wearing hard hat",
        "not wearing a helmet",
        "bare head",
        "bareheaded",
        "bare headed",
        "bare scalp",
        "visible hair",
        "hair visible",
        "baseball cap",
        "cloth cap",
        "knit cap",
        "beanie",
        "cap instead of",
        "hat instead of",
    )
    if any(term in combined for term in explicit_absence):
        return "Not observed"

    # A visible hard hat/helmet wins only after explicit absence has been ruled out.
    if (
        "hard hat" in combined
        or "safety helmet" in combined
        or "construction helmet" in combined
    ):
        return "Visible"

    return norm_ppe_item_status(value)


def norm_region_visibility(value: Any) -> str:
    raw = _compact_status_text(value)
    if raw in {"not visible", "hidden", "outside crop", "not assessable"} or (
        "not visible" in raw or "outside" in raw or "hidden" in raw
    ):
        return "Not visible"
    if raw in {"partial", "partly visible", "partially visible", "occluded", "blurred", "tiny"} or any(
        token in raw for token in ("partial", "partly", "occluded", "blurred", "tiny", "turned away")
    ):
        return "Partial"
    if raw in {"clear", "visible", "fully visible", "unobstructed"} or any(
        token in raw for token in ("clear", "fully visible", "unobstructed")
    ):
        return "Clear"
    return "Unclear"


def norm_binary_visible(value: Any) -> str:
    raw = _compact_status_text(value)
    if raw in {"not visible", "no", "absent", "not applicable", "n/a", "na"} or raw.startswith("not "):
        return "Not visible"
    if raw in {"visible", "yes", "present"} or raw.startswith("visible") or raw.startswith("present"):
        return "Visible"
    return "Unclear"


def norm_active_construction_context(
    value: Any,
    worker_evidence: Any = "",
    scene_context: Any = "",
) -> str:
    """Normalise construction context without letting a weak negative enum erase scene evidence.

    A worker-level field such as ``Not visible`` is not reliable enough to override
    explicit construction context recorded elsewhere in the same Pass 1 card. For
    example, a worker described beside rebar in an excavation remains in construction
    context even when the dedicated enum was returned as ``Not visible``. This helper
    still preserves ``Not visible`` when neither the worker evidence nor the scene
    describes construction work.
    """
    raw = _compact_status_text(value)
    worker_text = _compact_status_text(worker_evidence)
    scene_text = _compact_status_text(scene_context)
    evidence_context = f"{worker_text} {scene_text}".strip()

    construction_terms = (
        "active construction", "construction site", "construction work",
        "construction zone", "building site", "job site", "jobsite", "work site",
        "worksite", "work zone", "excavation", "trench", "rebar", "reinforcement",
        "formwork", "concrete", "masonry", "brickwork", "scaffold", "scaffolding",
        "roof", "rooftop", "unfinished structure", "under construction",
        "steel frame", "structural steel", "construction plant", "earthmoving",
        "excavator", "backhoe", "loader", "bulldozer", "crane", "dump truck",
    )
    evidence_supports_construction = any(term in evidence_context for term in construction_terms)

    if raw in {"visible", "yes", "present"} or raw.startswith("visible") or raw.startswith("present"):
        return "Visible"
    if evidence_supports_construction:
        return "Visible"
    if raw in {"not visible", "no", "not applicable", "n/a", "na"} or raw.startswith("not "):
        return "Not visible"
    return "Unclear"

def norm_vehicle_equipment_exposure(
    value: Any,
    worker_evidence: Any = "",
) -> str:
    """Normalise a worker-linked vehicle/equipment exposure trigger."""
    raw = _compact_status_text(value)
    evidence = _compact_status_text(worker_evidence)
    combined = f"{raw} {evidence}".strip()

    if raw in {"not visible", "no", "not applicable", "n/a", "na"} or raw.startswith("not "):
        return "Not visible"
    if raw in {"visible", "yes", "present"} or raw.startswith("visible") or raw.startswith("present"):
        return "Visible"

    equipment_terms = (
        "vehicle", "truck", "dump truck", "forklift", "excavator", "loader",
        "backhoe", "bulldozer", "crane", "heavy equipment", "plant equipment",
        "moving equipment", "construction plant",
    )
    relation_terms = (
        "near", "nearby", "next to", "beside", "adjacent", "under", "below",
        "behind", "in front of", "travel path", "vehicle path", "swing zone",
        "swing path", "working around", "alongside",
    )
    if any(term in combined for term in equipment_terms) and any(
        term in combined for term in relation_terms
    ):
        return "Visible"
    return "Unclear"


def norm_eye_hazard_task(
    value: Any,
    worker_evidence: Any = "",
) -> str:
    """Normalise eye-hazard task status from the same worker's evidence."""
    raw = _compact_status_text(value)
    evidence = _compact_status_text(worker_evidence)
    combined = f"{raw} {evidence}".strip()

    if raw in {"not visible", "no", "not applicable", "n/a", "na"} or raw.startswith("not "):
        return "Not visible"
    if raw in {"visible", "yes", "present"} or raw.startswith("visible") or raw.startswith("present"):
        return "Visible"

    task_terms = (
        "drill", "drilling", "grind", "grinding", "cutting", "cut", "saw",
        "sawing", "weld", "welding", "chipping", "hammering", "hammer",
        "torch", "abrasive", "metalwork",
    )
    if any(term in combined for term in task_terms):
        return "Visible"
    return "Unclear"

def normalise_ppe_assessments(value: Any, scene_context: str = "") -> list[dict[str, str]]:
    assessments: list[dict[str, str]] = []
    for index, raw in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(raw, dict):
            continue
        evidence = str(raw.get("evidence", "")).strip()
        worker_id = str(raw.get("worker_id", f"worker_{index}")).strip() or f"worker_{index}"
        location = str(raw.get("location", "unspecified location")).strip() or "unspecified location"
        source = str(raw.get("source", "full image")).strip() or "full image"
        assessment = {
            "worker_id": worker_id,
            "location": location,
            "source": source,
            "active_construction_context": norm_active_construction_context(
                raw.get("active_construction_context"),
                evidence,
                scene_context,
            ),
            "head_visibility": norm_region_visibility(raw.get("head_visibility")),
            "hard_hat": norm_hard_hat_status(raw.get("hard_hat"), evidence),
            "torso_visibility": norm_region_visibility(raw.get("torso_visibility")),
            "vehicle_equipment_exposure": norm_vehicle_equipment_exposure(
                raw.get("vehicle_equipment_exposure"),
                evidence,
            ),
            "high_visibility_vest": norm_ppe_item_status(raw.get("high_visibility_vest")),
            "eye_hazard_task": norm_eye_hazard_task(
                raw.get("eye_hazard_task"),
                evidence,
            ),
            "eye_region_visibility": norm_region_visibility(raw.get("eye_region_visibility")),
            "eye_protection": norm_ppe_item_status(raw.get("eye_protection")),
            "evidence": evidence,
        }
        # Ignore empty hallucinated records that contain no useful observation.
        if evidence or any(
            assessment[key] not in {"Unclear", "Not visible", "Not applicable"}
            for key in (
                "active_construction_context", "hard_hat", "high_visibility_vest",
                "eye_hazard_task", "eye_protection",
            )
        ):
            assessments.append(assessment)
    return assessments

def build_ppe_cues_from_assessments(assessments: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Create worker-linked PPE cues without silently dropping clear missing items.

    A clear missing item in confirmed work/exposure context remains confirmed.  The
    same clear item with an Unclear trigger is retained as a possible concern so
    Pass 2 can inspect the image; it is not silently discarded.
    """
    confirmed: list[dict[str, Any]] = []
    possible: list[dict[str, Any]] = []

    for worker in assessments:
        worker_ref = f"{worker['worker_id']} ({worker['location']}; source: {worker['source']})"
        evidence = worker["evidence"] or f"PPE assessment for {worker_ref}."

        def add_confirmed(control: str, statement: str) -> None:
            confirmed.append(
                {
                    "category": "PPE Compliance Concern",
                    "status": "Visible",
                    "scenario": statement,
                    "evidence": f"{statement} {evidence}".strip(),
                    "controls": [{"control": control, "status": "Not observed"}],
                    "limitations": [],
                }
            )

        def add_possible(control: str, statement: str, limitation: str) -> None:
            possible.append(
                {
                    "category": "PPE Compliance Concern",
                    "status": "Unclear",
                    "scenario": statement,
                    "evidence": f"{statement} {evidence}".strip(),
                    "controls": [{"control": control, "status": "Unclear"}],
                    "limitations": [limitation],
                }
            )

        # Hard hat — clear bare/non-hard-hat headwear is never erased merely
        # because a separate construction-context field was returned as Unclear.
        clear_missing_hard_hat = (
            worker["head_visibility"] == "Clear"
            and worker["hard_hat"] == "Not observed"
        )
        if worker["active_construction_context"] == "Visible":
            if clear_missing_hard_hat:
                add_confirmed(
                    "Head protection (hard hat)",
                    f"{worker_ref} is visibly in active construction work and has a clear visible head without a hard hat.",
                )
            elif (
                worker["head_visibility"] in {"Partial", "Not visible", "Unclear"}
                or worker["hard_hat"] == "Unclear"
            ):
                add_possible(
                    "Head protection (hard hat)",
                    f"{worker_ref} is in active construction work, but hard-hat status cannot be reliably determined.",
                    "The worker's head is partly obscured, too small, turned away, or otherwise not clear enough for a missing-hard-hat finding.",
                )
        elif clear_missing_hard_hat and worker["active_construction_context"] == "Unclear":
            add_possible(
                "Head protection (hard hat)",
                f"{worker_ref} has a clear visible head without a hard hat, but active construction context was not conclusively established by Pass 1.",
                "The missing hard hat is visible, but the construction-context trigger is Unclear; Pass 2 must verify the worker's task and work area from the image.",
            )

        # High-visibility clothing — preserve a clear missing vest as a review cue
        # when the equipment-exposure trigger is uncertain.
        clear_missing_vest = (
            worker["torso_visibility"] == "Clear"
            and worker["high_visibility_vest"] == "Not observed"
        )
        if worker["vehicle_equipment_exposure"] == "Visible":
            if clear_missing_vest:
                add_confirmed(
                    "High-visibility clothing",
                    f"{worker_ref} is visibly exposed to nearby vehicle or heavy-equipment activity and has a clear torso without high-visibility clothing.",
                )
            elif (
                worker["torso_visibility"] in {"Partial", "Not visible", "Unclear"}
                or worker["high_visibility_vest"] == "Unclear"
            ):
                add_possible(
                    "High-visibility clothing",
                    f"{worker_ref} may be exposed to vehicle or equipment activity, but high-visibility clothing cannot be reliably determined.",
                    "The worker's torso or the relevant vehicle/equipment exposure is not clear enough for a missing-vest finding.",
                )
        elif clear_missing_vest and worker["vehicle_equipment_exposure"] == "Unclear":
            add_possible(
                "High-visibility clothing",
                f"{worker_ref} has a clear torso without high-visibility clothing, but vehicle/equipment exposure was not conclusively established by Pass 1.",
                "The missing high-visibility clothing is visible, but the worker-linked equipment-exposure trigger is Unclear; Pass 2 must verify the scene from the image.",
            )

        # Eye/face protection — the task normaliser now recognises natural evidence
        # wording. Preserve a clear missing item as reviewable if the task trigger
        # remains Unclear.
        clear_missing_eye_protection = (
            worker["eye_region_visibility"] == "Clear"
            and worker["eye_protection"] == "Not observed"
        )
        if worker["eye_hazard_task"] == "Visible":
            if clear_missing_eye_protection:
                add_confirmed(
                    "Eye/face protection",
                    f"{worker_ref} is visibly performing an eye-hazard task and has a clear visible eye area without eye or face protection.",
                )
            elif (
                worker["eye_region_visibility"] in {"Partial", "Not visible", "Unclear"}
                or worker["eye_protection"] == "Unclear"
            ):
                add_possible(
                    "Eye/face protection",
                    f"{worker_ref} is visibly performing an eye-hazard task, but eye-protection status cannot be reliably determined.",
                    "The worker's eye area is partly obscured, too small, turned away, or otherwise not clear enough for a missing-eye-protection finding.",
                )
        elif clear_missing_eye_protection and worker["eye_hazard_task"] == "Unclear":
            add_possible(
                "Eye/face protection",
                f"{worker_ref} has a clear eye area without eye/face protection, but an eye-hazard task was not conclusively established by Pass 1.",
                "The missing eye protection is visible, but the worker-linked task trigger is Unclear; Pass 2 must verify whether an eye-hazard task is occurring.",
            )

    combined = [*confirmed, *possible]
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for cue in combined:
        control_name = str((cue.get("controls") or [{}])[0].get("control", "PPE"))
        key = (
            cue.get("status", "Unclear"),
            control_name,
            str(cue.get("evidence", "")).lower(),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(cue)
    return deduplicated

def normalise_inventory(raw_inventory: Any) -> dict[str, Any]:
    """Keep the sparse Pass 1 schema and apply deterministic evidence guards."""
    if not isinstance(raw_inventory, dict):
        raw_inventory = {}

    raw_scene = raw_inventory.get("scene", {})
    if not isinstance(raw_scene, dict):
        raw_scene = {}
    worker_count = normalise_worker_count(raw_scene.get("worker_count", "Unclear"))

    scene = {
        "worker_count": worker_count,
        "work_activity": str(raw_scene.get("work_activity", "Unclear")).strip() or "Unclear",
        "key_objects": unique_text(raw_scene.get("key_objects", [])),
    }
    scene_context = f"{scene['work_activity']} {' '.join(scene['key_objects'])}"

    cues: list[dict[str, Any]] = []
    raw_cue_texts: list[str] = []
    seen_cue_keys: set[tuple[str, str, str]] = set()
    for raw_cue in raw_inventory.get("observed_hazard_cues", []):
        if not isinstance(raw_cue, dict):
            continue
        category = norm_category(raw_cue.get("category"))
        if category not in VALID_CATEGORY_SET:
            continue

        scenario = str(raw_cue.get("scenario", "")).strip()
        evidence = str(raw_cue.get("evidence", "")).strip()
        if not scenario and not evidence:
            continue
        raw_cue_texts.append(f"{scenario} {evidence}".strip())

        controls = normalise_controls(raw_cue.get("controls"))
        submitted_category = category
        category, guard_message, force_unclear = reconcile_cue_category(
            category, scenario, evidence, controls, scene_context
        )
        guard_decision = "kept" if category is not None else "dropped"
        GUARD_DIAGNOSTICS.append(
            {
                "submitted_category": submitted_category,
                "final_category": category,
                "decision": guard_decision,
                "struck_by_signal_level": struck_by_mechanism_level(f"{scenario} {evidence}"),
                "caught_in_between_signal": caught_in_between_mechanism_signal(f"{scenario} {evidence}"),
                "forced_unclear": force_unclear,
                "guard_message": guard_message,
                "scenario": scenario,
                "evidence": evidence,
            }
        )
        if guard_message:
            print(f"[inventory guard] {guard_message}")
        if category is None:
            continue

        limitations = unique_text(raw_cue.get("limitations", []))
        scenario, evidence, controls, limitations = apply_electrical_inventory_guard(
            scene, category, scenario, evidence, controls, limitations
        )
        controls, limitations = apply_fall_control_guard(
            category, scenario, evidence, controls, limitations
        )
        cue = {
            "category": category,
            "status": "Unclear" if force_unclear else norm_cue_status(raw_cue.get("status")),
            "scenario": scenario or evidence,
            "evidence": evidence or scenario,
            "controls": controls,
            "limitations": limitations,
        }
        key = (cue["category"], cue["status"], cue["evidence"].lower())
        if key not in seen_cue_keys:
            seen_cue_keys.add(key)
            cues.append(cue)

    # Narrow Fall fallback: only add an Unclear review cue when Pass 1 text
    # explicitly places a worker at an edge/rim/opening/drop, but the model failed
    # to emit the Fall category. This does not treat every trench as a fall hazard.
    raw_observation_text = f"{scene_context} {' '.join(raw_cue_texts)}"
    has_fall_cue = any(cue.get("category") == "Fall Hazards" for cue in cues)
    if worker_fall_edge_signal(raw_observation_text) and not has_fall_cue:
        cues.append(
            {
                "category": "Fall Hazards",
                "status": "Unclear",
                "scenario": "A worker is described at or near an open edge, rim, opening, or visible drop.",
                "evidence": (
                    "Pass 1 text contains a worker-linked edge/opening/drop description. "
                    "The image does not establish exact height or full edge-protection coverage."
                ),
                "controls": [
                    {
                        "control": "Edge protection / fall protection",
                        "status": "Unclear",
                    }
                ],
                "limitations": [
                    "Exact height, full edge coverage, and fall-protection adequacy cannot be established from a single image."
                ],
            }
        )
        print("[Fall guard] Added an Unclear Fall cue from a worker-linked edge/opening/drop description.")

    has_electrical_context = text_has_any(scene_context, ELECTRICAL_CONTEXT_TERMS)
    has_electrical_cue = any(
        cue.get("category") == "Electrical / Electrocution Hazards" for cue in cues
    )
    if has_electrical_context and not has_electrical_cue:
        safe_description = electrical_safe_description(scene, scene_context)
        cues.append(
            {
                "category": "Electrical / Electrocution Hazards",
                "status": "Unclear",
                "scenario": safe_description,
                "evidence": safe_description,
                "controls": [],
                "limitations": [ELECTRICAL_UNCERTAINTY],
            }
        )
        print("[inventory guard] Added an Unclear Electrical cue from visible electrical scene context.")

    # Trench/excavation fallback: add a cue only when Pass 1 scene text places
    # a worker inside the trench/excavation. Digging activity or a visible trench
    # alone is not enough for a Caught-in/Between concern.
    has_worker_inside_excavation = worker_inside_excavation_signal(scene_context)
    has_generic_worker_trench_context = (
        re.search(rf"\b{PERSON_NOUN_PATTERN}\b", scene_context, flags=re.IGNORECASE) is not None
        and re.search(r"\b(?:trench|trenches|excavation|excavations)\b", scene_context, flags=re.IGNORECASE) is not None
    )
    has_caught_cue = any(
        cue.get("category") == "Caught-in/Between Hazards" for cue in cues
    )
    if has_generic_worker_trench_context and not has_worker_inside_excavation:
        GUARD_DIAGNOSTICS.append(
            {
                "submitted_category": "Caught-in/Between Hazards",
                "final_category": None,
                "decision": "not_auto_added",
                "reason": "Worker and trench/excavation co-occurred in scene context without a worker-inside position relation.",
                "scene_context": scene_context,
            }
        )
        print("[excavation guard] Worker and trench/excavation co-occur, but no worker-inside relation was recorded; no automatic cave-in cue added.")
    if has_worker_inside_excavation and not has_caught_cue:
        activity = scene.get("work_activity", "").strip()
        trench_description = (
            f"{activity}; a worker is described as inside a trench or excavation."
            if activity and activity != "Unclear"
            else "A worker is described as inside a trench or excavation."
        )
        cues.append(
            {
                "category": "Caught-in/Between Hazards",
                "status": "Unclear",
                "scenario": trench_description,
                "evidence": (
                    "Pass 1 scene context places a worker inside a trench or excavation. "
                    "Cave-in protection (sloping, shoring, or shielding) adequacy "
                    "cannot be confirmed from the image."
                ),
                "controls": [
                    {
                        "control": "Cave-in protection (sloping/shoring/shielding)",
                        "status": "Unclear",
                    }
                ],
                "limitations": [
                    "Trench depth, soil type, and protective-system adequacy cannot be "
                    "established from a single image."
                ],
            }
        )
        print(
            "[inventory guard] Added an Unclear Caught-in/Between cue because "
            "scene text places a worker inside a trench/excavation."
        )


    raw_ppe = raw_inventory.get("ppe_observations", {})
    if not isinstance(raw_ppe, dict):
        raw_ppe = {}
    ppe_observations = {"visible_items": unique_text(raw_ppe.get("visible_items", []))}

    # PPE is evaluated per worker rather than through one scene-level list.
    ppe_assessments = normalise_ppe_assessments(raw_inventory.get("ppe_assessments", []), scene_context)
    derived_ppe_cues = build_ppe_cues_from_assessments(ppe_assessments)
    if derived_ppe_cues:
        # Do not silently discard a direct generic PPE cue merely because a separate
        # worker-level array was also returned.  The worker-linked records remain the
        # stronger evidence, but an unlinked Pass 1 PPE observation is preserved as
        # Unclear for Pass 2 rather than being erased.  It cannot by itself force a
        # confirmed violation because its control statuses are downgraded to Unclear.
        existing_ppe_cues = [
            cue for cue in cues if cue.get("category") == "PPE Compliance Concern"
        ]
        non_ppe_cues = [
            cue for cue in cues if cue.get("category") != "PPE Compliance Concern"
        ]
        preserved_generic_ppe: list[dict[str, Any]] = []
        for generic_cue in existing_ppe_cues:
            preserved = dict(generic_cue)
            preserved["status"] = "Unclear"
            preserved["controls"] = [
                {
                    **dict(control),
                    "status": (
                        "Unclear"
                        if control.get("status") == "Not observed"
                        else control.get("status", "Unclear")
                    ),
                }
                for control in generic_cue.get("controls", [])
                if isinstance(control, dict)
            ]
            limitation = (
                "The original Pass 1 PPE cue was not worker-linked; it is retained "
                "for image verification but cannot independently establish a missing control."
            )
            preserved["limitations"] = unique_text([
                *generic_cue.get("limitations", []), limitation
            ])
            preserved_generic_ppe.append(preserved)

        cues = [*non_ppe_cues, *preserved_generic_ppe, *derived_ppe_cues]
        confirmed_count = sum(cue.get("status") == "Visible" for cue in derived_ppe_cues)
        print(
            "[PPE guard] Added "
            f"{len(derived_ppe_cues)} worker-linked PPE cue(s) "
            f"({confirmed_count} with a directly visible missing control); "
            f"preserved {len(preserved_generic_ppe)} generic PPE cue(s) for Pass 2 review."
        )
    else:
        # Fallback only when the model omitted the worker-level array entirely.
        # This preserves the older eye-protection behaviour as an Unclear screening
        # cue, but it never becomes a confirmed violation on its own.
        task_text = f"{scene['work_activity']} {' '.join(scene['key_objects'])}".lower()
        eye_hazard_task_terms = (
            "drill", "drilling", "grind", "grinding", "cutting", "saw", "sawing",
            "weld", "welding", "chipping", "hammering",
        )
        eye_protection_terms = (
            "safety glasses", "goggles", "eye protection", "face shield", "protective eyewear",
        )
        has_visible_eye_hazard_task = any(term in task_text for term in eye_hazard_task_terms)
        eye_protection_visible = any(
            term in " ".join(ppe_observations["visible_items"]).lower()
            for term in eye_protection_terms
        )
        ppe_cue_already_recorded = any(
            cue.get("category") == "PPE Compliance Concern" for cue in cues
        )
        if has_visible_eye_hazard_task and not eye_protection_visible and not ppe_cue_already_recorded:
            cues.append(
                {
                    "category": "PPE Compliance Concern",
                    "status": "Unclear",
                    "scenario": "An eye-hazard task is visible, but the task worker's eye protection cannot be linked reliably from the full image.",
                    "evidence": "The full-scene inventory recorded an eye-hazard task but did not contain a worker-linked eye-protection assessment.",
                    "controls": [{"control": "Eye/face protection", "status": "Unclear"}],
                    "limitations": ["No worker-linked face/eye assessment was returned in Pass 1."],
                }
            )
            print("[PPE guard] Added only an Unclear fallback because worker-level PPE assessment was absent.")

    return {
        "scene": scene,
        "observed_hazard_cues": cues,
        "ppe_observations": ppe_observations,
        "ppe_assessments": ppe_assessments,
        "general_limitations": unique_text(raw_inventory.get("general_limitations", [])),
    }


def cue_categories(inventory: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for cue in inventory.get("observed_hazard_cues", []):
        category = cue.get("category")
        if category in VALID_CATEGORY_SET and category not in categories:
            categories.append(category)
    return categories


def best_cue_for_category(inventory: dict[str, Any], category: str | None) -> dict[str, Any] | None:
    if category is None:
        return None
    cues = [
        cue
        for cue in inventory.get("observed_hazard_cues", [])
        if cue.get("category") == category
    ]
    if not cues:
        return None
    visible = [cue for cue in cues if cue.get("status") == "Visible"]
    if category != "PPE Compliance Concern":
        return visible[0] if visible else cues[0]

    # One category-level gate is still used for PPE, so combine all worker-linked
    # records rather than silently selecting only the first confirmed worker.
    selected = [*visible, *[cue for cue in cues if cue.get("status") != "Visible"]]
    return {
        "category": category,
        "status": "Visible" if visible else "Unclear",
        "scenario": " | ".join(unique_text([cue.get("scenario", "") for cue in selected])),
        "evidence": " | ".join(unique_text([cue.get("evidence", "") for cue in selected])),
        "controls": [control for cue in selected for control in cue.get("controls", [])],
        "limitations": unique_text([item for cue in selected for item in cue.get("limitations", [])]),
    }


# ── Secondary hazard record builder ──────────────────────────────────────────
def build_secondary_hazard_records(
    inventory: dict[str, Any],
    primary_category: str | None,
    retrieved_rule_keys: list[str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for category in cue_categories(inventory):
        if category == primary_category:
            continue

        cue = best_cue_for_category(inventory, category)
        if cue is None:
            continue

        records.append(
            {
                "hazard_category": category,
                "status": "Possible concern / human review required",
                "osha_reference": canonical_retrieved_citation(
                    category,
                    retrieved_rule_keys,
                ),
                "decision_evidence": cue.get("evidence") or cue.get("scenario"),
                "uncertainty": (
                    "; ".join(cue.get("limitations", []))
                    or "Control adequacy requires review."
                ),
            }
        )

    return records


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — image only; sparse unified evidence card
# ══════════════════════════════════════════════════════════════════════════════
pass1_prompt = f"""You are a construction-scene observer. Produce a SPARSE visual evidence card.

TARGET CATEGORIES ONLY:
{CATEGORY_TEXT}

CORE RULES:
- Use only directly visible information from the image. Do not make OSHA, legal,
  compliance, exact-height, depth, voltage, or task-intent conclusions.
- Do NOT force every category into the output.
- Add an item to observed_hazard_cues ONLY when that category has:
  (a) a directly visible cue, or
  (b) a genuinely relevant uncertainty tied to a partly visible cue.
  Do not add a category merely because it is not visible or not assessable.
- Include every target category that meets the rule above. Do not select a primary hazard.

CATEGORY BOUNDARIES — apply these strictly:
- Fall Hazards: scaffolds, ladders, platforms, roofs, open edges/openings, or a visible drop.
  A worker at or near an open excavation/trench rim, an unprotected edge, a floor opening,
  or another visible drop is also a Fall cue. Record the cue when the worker or occupied
  equipment is visibly exposed at the edge; do NOT create a Fall cue merely because an
  excavation exists in the scene. A worker inside a trench/excavation is primarily a
  Caught-in/Between cue unless a separate edge/drop exposure is visible.
  A worker on scaffold steps, a platform, or an edge is a Fall cue, NOT a Caught-in/Between cue.

- Struck-by / Equipment Hazards:
  Use this when a worker is visibly at risk of being hit by moving equipment,
  a vehicle, or a load. Common visually assessable cues include:
    • a raised bucket, arm, or boom directly above or near the worker
    • a suspended, swinging, or hoisted load near the worker
    • an excavator, crane, loader, dump truck, or other heavy equipment with a
      raised or moving arm, bucket, boom, body, or load positioned close to the worker
    • a vehicle that appears to be reversing, loading, or moving toward the worker
    • the worker standing in a visible travel path, swing zone, or discharge zone
    • the worker directly below or beside a raised forklift load or crane hook

  Describe what you see naturally. You do not need to use technical OSHA terms.
  Do NOT infer exact distances, operator visibility, or whether the operator can
  see the worker. Do NOT conclude the hazard is confirmed from proximity alone.

- Caught-in/Between Hazards:
  Use this only when a worker is visibly trapped, crushed, pinned between objects,
  exposed to a machinery pinch point, working under/between raised equipment in a
  maintenance or blocking/cribbing context, OR visibly inside a trench/excavation
  with possible cave-in exposure.

  A worker inside a trench/excavation must be recorded as a Caught-in/Between cue
  when the worker position is visible. Use status Unclear unless trench depth,
  soil conditions, and sloping/shoring/shielding are clearly visible.

  A raised bucket or suspended load directly above a free-standing worker is a
  Struck-by concern, not Caught-in/Between. A worker merely beside equipment or
  beside a trench is not automatically a Caught-in/Between concern.

- Electrical / Electrocution Hazards: exposed/damaged conductors, open electrical equipment,
  overhead lines, or another visible electrical source.
  * Electrical context takes priority over generic equipment proximity.
  * Do not state or imply live/energized status, voltage, exposed conductors, clearance,
    grounding, lockout/tagout, or missing electrical controls.
  * For electrical cues, do not list a safety switch, grounding system, harness, guardrail,
    or PPE item as Not observed unless directly visible and absent.

- PPE Compliance Concern: assess PPE per worker, never as one scene-wide judgment.
  A missing-PPE cue must identify the same worker, the relevant PPE item, and a
  sufficiently clear head, torso, or eye region. Fill the ppe_assessments array
  below directly from the full image; look closely at each worker's head, torso,
  and eye region before deciding. Before finalising JSON, perform one deliberate
  hard-hat scan of priority workers in construction context; do not omit a clearly
  bare-headed or cap-wearing worker merely because the person is walking, inspecting,
  supervising, or not holding a tool.

MULTI-HAZARD RULE:
- One image may contain more than one independent hazard.
- Return one separate cue per category only when each category has its own direct evidence.
- Do not copy generic equipment evidence across categories.

Example:
- Worker inside a trench with steep walls = Caught-in/Between cave-in cue.
- Add Struck-by only if a bucket/load is directly above or near the worker,
  a vehicle is reversing toward the worker, or a visible load/swing/travel
  path puts the worker at direct impact risk.

PPE WORKER-LINKING RULES:
- Fill ppe_assessments for at most SIX priority workers: first workers whose PPE is absent or unclear, then workers performing a triggering task. Treat worker, employee, operator, laborer, crew member, or individual as a person to assess. Do not list extra workers whose PPE is clearly visible.
- Hard hat: assess a worker visibly undertaking OR present in active construction work.
  Active construction context includes a construction site/work zone, excavation or trench work,
  rebar/reinforcement/formwork, concrete work, scaffolding, unfinished structures, or active
  heavy-construction plant work. The worker does NOT need to hold a tool.
  When the worker's head is Clear and you see hair, bare scalp, a baseball/cloth/knit cap,
  a beanie, a hood, or other non-hard-hat headwear, record:
  active_construction_context=Visible, head_visibility=Clear, hard_hat=Not observed.
  Use Unclear only when the head itself is tiny, blurred, occluded, turned away, or outside frame.
- High-visibility clothing: assess only for a worker visibly exposed to nearby vehicles
  or heavy equipment. Use Not observed only when the same worker's torso is Clear.
- Eye/face protection: a visible drill, grinder, saw, cutting tool, welding activity,
  chipping task, or hammering task is a trigger. Use Not observed only when the SAME
  task worker's eye area is Clear and no eye/face protection is visible.
- Use Unclear, not Not observed, when the relevant worker's head, torso, or eye area is
  tiny, blurred, occluded, turned away, or outside the image.
- Do not create a Fall cue without a visible elevated surface, ladder, scaffold,
  roof, platform, edge, opening, or drop. A worker at a visible open excavation/trench
  rim or unprotected edge is an edge/drop cue; an excavation by itself is not.
- Do not create a Caught-in/Between cue merely because a worker is bending,
  near rocks, near blocks, near materials, or near concrete structures.

FALL-CONTROL CAUTION:
- On scaffolds and elevated platforms, unless the relevant edge and missing protection
  are both clearly visible, use control status Unclear rather than Not observed.
- Use only category-relevant controls.

- For each cue status, use EXACTLY: Visible or Unclear.
- For each control status, use EXACTLY: Visible, Not observed, Unclear, or Not applicable.
  * Not observed: the relevant area is visible and the control is absent.
  * Unclear: the relevant area/control cannot be judged reliably.
  * Never treat Unclear as Not observed.

- If no target cue is supported, output an empty observed_hazard_cues array.

Output ONLY valid JSON matching this exact structure:
{{
  "scene": {{
    "worker_count": <integer or "Unclear">,
    "work_activity": "<directly visible activity or Unclear>",
    "key_objects": ["<visible object>"]
  }},
  "observed_hazard_cues": [
    {{
      "category": "<one exact target category>",
      "status": "Visible or Unclear",
      "scenario": "<short direct visual description>",
      "evidence": "<specific image evidence only>",
      "controls": [
        {{"control": "<relevant control only>", "status": "Visible / Not observed / Unclear / Not applicable"}}
      ],
      "limitations": ["<only limitations relevant to this cue>"]
    }}
  ],
  "ppe_observations": {{
    "visible_items": ["<only clearly visible PPE items>"]
  }},
  "ppe_assessments": [
    {{
      "worker_id": "worker_1",
      "location": "left / centre / right",
      "active_construction_context": "Visible / Unclear / Not visible (use Visible for a worker in a visible construction/trench/rebar/formwork/scaffold/unfinished-structure work area)",
      "head_visibility": "Clear / Partial / Not visible",
      "hard_hat": "Visible / Not observed / Unclear / Not applicable (use Not observed for a clear bare head or cap/non-hard-hat headwear)",
      "torso_visibility": "Clear / Partial / Not visible",
      "vehicle_equipment_exposure": "Visible / Unclear / Not applicable",
      "high_visibility_vest": "Visible / Not observed / Unclear / Not applicable",
      "eye_hazard_task": "Visible / Not visible / Unclear",
      "eye_region_visibility": "Clear / Partial / Not visible",
      "eye_protection": "Visible / Not observed / Unclear / Not applicable",
      "evidence": "short worker-specific visual statement"
    }}
  ],
  "general_limitations": ["<image-level uncertainty only>"]
}}"""



# ── User-facing reports + developer/audit JSON ───────────────────────────────
def _report_hazards(result: dict[str, Any]) -> list[dict[str, Any]]:
    hazards = result.get("hazards", [])
    return hazards if isinstance(hazards, list) else []


def _display_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def user_facing_summary(
    result: dict[str, Any], image_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a concise safety-user view while JSON retains the full audit trail."""
    hazards = _report_hazards(result)
    image_info = image_info or {}
    runtime_value = image_info.get(
        "image_processing_runtime_seconds",
        result.get("image_processing_runtime_seconds", result.get("pipeline_runtime_seconds")),
    )
    try:
        runtime_seconds = round(float(runtime_value), 2)
    except (TypeError, ValueError):
        runtime_seconds = None
    main_category = _display_text(
        result.get("main_hazard") or result.get("hazard_category"),
        "No target hazard concern visible",
    )
    screening_status = _display_text(result.get("screening_status"), "Unknown")

    status_label = {
        "Confirmed concern": "Confirmed concern",
        "Possible concern": "Possible concern — human review recommended",
        "No concern visible": "No visible concern detected",
    }.get(screening_status, screening_status)

    if screening_status == "No concern visible":
        evidence = _display_text(
            result.get("decision_evidence"),
            "No supported Focus Four or PPE concern was identified from the visible image.",
        )
    else:
        evidence = _display_text(
            result.get("decision_evidence") or result.get("scene_summary"),
            "A relevant safety cue was identified from the image.",
        )

    requires_review = bool(result.get("requires_human_review", False))
    if requires_review:
        review_text = "Human review is required before relying on this image-based result."
    elif screening_status == "No concern visible":
        review_text = "No immediate image-based review flag was generated."
    else:
        review_text = "No additional human-review flag was generated."

    additional_concerns = []
    for hazard in hazards:
        category = _display_text(hazard.get("hazard_category"), "Unknown category")
        if category == main_category:
            continue
        additional_concerns.append(
            {
                "category": category,
                "status": _display_text(hazard.get("screening_status"), "Unknown"),
                "risk": _display_text(hazard.get("risk_level"), "Not assessable"),
                "evidence": _display_text(
                    hazard.get("decision_evidence") or hazard.get("scene_summary"),
                    "No concise image evidence was recorded.",
                ),
            }
        )

    return {
        "overall_status": status_label,
        "main_hazard": main_category,
        "risk_level": _display_text(result.get("risk_level"), "Not assessable"),
        "runtime_seconds": runtime_seconds,
        "runtime_label": (
            f"{runtime_seconds:.2f} seconds (model loading excluded)"
            if runtime_seconds is not None
            else "Not available"
        ),
        "evidence": evidence,
        "human_review": review_text,
        "osha_reference": _display_text(result.get("osha_reference"), "Not applicable"),
        "additional_concerns": additional_concerns,
    }

def _markdown_safe(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()


def write_additional_reports(final_output: dict[str, Any], json_path: Path) -> None:
    """Write concise end-user reports; the paired JSON remains the developer audit record."""
    if OUTPUT_FORMAT == "json":
        return

    result = final_output.get("vlm_result", {})
    summary = user_facing_summary(result, final_output.get("image_info", {}))

    additional_html = ""
    if summary["additional_concerns"]:
        rows = []
        for item in summary["additional_concerns"]:
            rows.append(
                "<li>"
                f"<strong>{html.escape(item['category'])}</strong> — "
                f"{html.escape(item['status'])} | Risk: {html.escape(item['risk'])}<br>"
                f"<strong>Evidence:</strong> {html.escape(item['evidence'])}"
                "</li>"
            )
        additional_html = "<section><h2>Other concerns detected</h2><ul>" + "".join(rows) + "</ul></section>"

    if OUTPUT_FORMAT in {"html", "all"}:
        html_report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construction Safety Screening Result</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 36px auto; max-width: 860px; line-height: 1.5; color: #222; }}
h1 {{ margin-bottom: 0.2em; }}
h2 {{ margin-top: 1.5em; font-size: 1.12em; }}
.card {{ border: 1px solid #c9c9c9; border-radius: 8px; padding: 18px; margin-top: 18px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.label {{ font-size: 0.9em; color: #555; }}
.value {{ font-weight: 700; }}
.note {{ margin-top: 28px; color: #555; font-size: 0.9em; }}
ul {{ padding-left: 22px; }}
</style>
</head>
<body>
<h1>Construction Safety Screening Result</h1>
<div class="card summary">
  <div><div class="label">Overall status</div><div class="value">{html.escape(summary['overall_status'])}</div></div>
  <div><div class="label">Main hazard</div><div class="value">{html.escape(summary['main_hazard'])}</div></div>
  <div><div class="label">Risk level</div><div class="value">{html.escape(summary['risk_level'])}</div></div>
  <div><div class="label">Runtime</div><div class="value">{html.escape(summary['runtime_label'])}</div></div>
</div>
<section>
<h2>Evidence</h2>
<p>{html.escape(summary['evidence'])}</p>
</section>
<section>
<h2>Human review</h2>
<p>{html.escape(summary['human_review'])}</p>
</section>
<section>
<h2>OSHA reference</h2>
<p>{html.escape(summary['osha_reference'])}</p>
</section>
{additional_html}
<p class="note">Image-based research screening only. This is not an official OSHA inspection, citation, or legal finding. Full technical evidence, retrieved-rule details, reviewer outputs, recommended controls, and detailed timing are retained in the paired JSON audit file. Runtime shown above excludes one-time model loading.</p>
</body>
</html>"""
        html_path = json_path.with_suffix(".html")
        html_path.write_text(html_report, encoding="utf-8")
        print(f"[output] User-facing HTML report saved: {html_path}")

    if OUTPUT_FORMAT in {"md", "all"}:
        md_lines = [
            "# Construction Safety Screening Result",
            "",
            f"**Overall status:** {_markdown_safe(summary['overall_status'])}  ",
            f"**Main hazard:** {_markdown_safe(summary['main_hazard'])}  ",
            f"**Risk level:** {_markdown_safe(summary['risk_level'])}  ",
            f"**Runtime:** {_markdown_safe(summary['runtime_label'])}",
            "",
            "## Evidence",
            _markdown_safe(summary['evidence']),
            "",
            "## Human review",
            _markdown_safe(summary['human_review']),
            "",
            "## OSHA reference",
            _markdown_safe(summary['osha_reference']),
        ]
        if summary["additional_concerns"]:
            md_lines.extend(["", "## Other concerns detected"])
            for item in summary["additional_concerns"]:
                md_lines.append(
                    f"- **{_markdown_safe(item['category'])}:** "
                    f"{_markdown_safe(item['status'])} | Risk: {_markdown_safe(item['risk'])}.  "
                    f"**Evidence:** {_markdown_safe(item['evidence'])}"
                )
        md_lines.extend([
            "",
            "---",
            "Image-based research screening only. This is not an official OSHA inspection, citation, or legal finding.",
            "Full technical evidence, retrieved-rule details, reviewer outputs, recommended controls, and detailed timing are retained in the paired JSON audit file. Runtime shown above excludes one-time model loading.",
        ])
        md_path = json_path.with_suffix(".md")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[output] User-facing Markdown report saved: {md_path}")

def run_one_image(image_path: str, image_number: int, total_images: int) -> tuple[dict[str, Any], Path]:
    """Run the unchanged Pass 1–3 reasoning sequence for one image."""
    global GUARD_DIAGNOSTICS

    IMAGE_PATH = str(image_path)
    GUARD_DIAGNOSTICS = []

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print("\n" + "=" * 88)
    print(f"[image {image_number}/{total_images}] {os.path.basename(IMAGE_PATH)}")
    print("=" * 88)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Per-image timer starts only after the model has already been loaded.
    # It measures Pass 1–3 processing and excludes model-loading and report-writing time.
    cuda_sync()
    pipeline_start = time.perf_counter()

    print("\n[Pass 1] Sparse visual evidence card...")
    cuda_sync()
    pass1_start = time.perf_counter()
    raw_pass1 = infer_image(IMAGE_PATH, pass1_prompt, max_tokens=PASS1_MAX_TOKENS)
    cuda_sync()
    pass1_seconds = time.perf_counter() - pass1_start
    if args.verbose:
        print("\n--- Raw Pass 1 Output ---\n", raw_pass1)
    else:
        print("[Pass 1] Evidence card generated.")

    parsed_pass1: dict[str, Any] | None = None
    pass1_json = extract_json(raw_pass1)
    if pass1_json:
        try:
            parsed_pass1 = json.loads(pass1_json)
            print("\n[Pass 1] JSON parsed successfully.")
        except json.JSONDecodeError as error:
            print(f"[Pass 1] JSON parse error: {error} — attempting one repair.")

    if parsed_pass1 is None:
        repair_prompt = (
            "Return only a corrected valid JSON object for the following malformed output. "
            "Do not add information or infer facts.\n\n" + raw_pass1
        )
        repaired = infer_text(repair_prompt, max_tokens=PASS1_MAX_TOKENS)
        repaired_json = extract_json(repaired)
        if repaired_json:
            try:
                parsed_pass1 = json.loads(repaired_json)
                print("[Pass 1] JSON repaired successfully.")
            except json.JSONDecodeError:
                pass

    if parsed_pass1 is None:
        raise RuntimeError(
            "Pass 1 JSON remained invalid after one repair attempt."
        )

    inventory = normalise_inventory(parsed_pass1)
    full_inventory_str = json.dumps(inventory, indent=2, ensure_ascii=False)

    # Every independently supported category proceeds to its own OSHA gate.
    gate_inventory = inventory
    locked_inventory_str = json.dumps(gate_inventory, indent=2, ensure_ascii=False)
    observed_categories = cue_categories(gate_inventory)

    if args.verbose:
        print("\n--- Locked Sparse Inventory (all supported hazards) ---\n", locked_inventory_str)
    if PPE_FOCUS:
        print("[PPE focus] Extra attention to per-worker PPE linking is requested; all supported hazards are still assessed.")
    print(f"\n[index] Gate categories: {observed_categories or 'None'}")


    # ── Pass 2 parser and generic consistency rules ───────────────────────────────
    def parse_secondary_categories(value: str, primary: str | None) -> list[str]:
        raw = str(value or "").strip()
        if raw.lower() in {"", "none", "not applicable", "n/a"}:
            return []
        values: list[str] = []
        for part in re.split(r"[,;|]", raw):
            category = norm_category(part)
            if category and category != primary and category not in values:
                values.append(category)
        return values


    def parse_gate(gate_text: str) -> dict[str, Any]:
        cleaned = clean_md(gate_text)
        screening = norm_screening_status(labeled(cleaned, "SCREENING STATUS:"))
        primary = norm_category(labeled(cleaned, "PRIMARY HAZARD CATEGORY:"))
        compliance = norm_compliance_status(labeled(cleaned, "COMPLIANCE STATUS:"), screening)
        record = {
            "screening_status": screening,
            "compliance_status": compliance,
            "primary_hazard_category": primary,
            "secondary_hazard_categories": parse_secondary_categories(
                labeled(cleaned, "SECONDARY HAZARD CATEGORIES:"), primary
            ),
            "relevant_osha_rule": labeled(cleaned, "RELEVANT OSHA RULE:") or "Not applicable",
            "visible_trigger": labeled(cleaned, "VISIBLE TRIGGER:") or "No trigger returned.",
            "worker_exposure": labeled(cleaned, "WORKER EXPOSURE:") or "Not assessable from image.",
            "control_status": norm_control_status(labeled(cleaned, "CONTROL STATUS:")),
            "evidence": labeled(cleaned, "EVIDENCE:") or "No evidence returned.",
            "uncertainty": labeled(cleaned, "UNCERTAINTY:") or "Not assessable from image.",
        }
        return record


    def no_cue_gate() -> dict[str, Any]:
        return {
            "screening_status": "No concern visible",
            "compliance_status": "No violation visible",
            "primary_hazard_category": None,
            "secondary_hazard_categories": [],
            "relevant_osha_rule": "Not applicable",
            "visible_trigger": "No target hazard cue was recorded in the locked Pass 1 inventory.",
            "worker_exposure": "No target worker exposure was established from the image.",
            "control_status": "Not applicable",
            "evidence": "Pass 1 recorded no Fall, Struck-by, Caught-in/Between, Electrical, or supported PPE cue.",
            "uncertainty": "No target hazard cue available for OSHA screening.",
        }


    def append_uncertainty(existing: str, addition: str) -> str:
        current = str(existing or "").strip()
        if current.lower() in {"", "none", "not applicable", "not assessable from image."}:
            return addition
        if addition.lower() in current.lower():
            return current
        return f"{current} {addition}".strip()


    def canonical_retrieved_citation(primary: str | None, retrieved_rule_keys: list[str]) -> str:
        """Return citations only from rules actually retrieved for the locked category."""
        citations: list[str] = []
        for rule_key in retrieved_rule_keys:
            rule = OSHA_RULES.get(rule_key, {})
            if rule.get("category") != primary:
                continue
            citation = str(rule.get("official_source", "")).strip()
            if citation and citation not in citations:
                citations.append(citation)
        return "; ".join(citations) if citations else "Not assessable from retrieved evidence"


    def apply_generic_consistency(
        gate: dict[str, Any], inventory: dict[str, Any], retrieved_rule_keys: list[str]
    ) -> dict[str, Any]:
        """Apply evidence locks, citation locks, and conservative electrical handling."""
        categories = cue_categories(inventory)
        if not categories:
            return no_cue_gate()

        primary = gate.get("primary_hazard_category")
        if primary not in categories:
            # A reviewer may legitimately return "No concern visible" with no
            # primary category.  Preserve that conclusion rather than converting
            # every uncertain Pass 1 cue into a positive prediction.  A named but
            # unsupported category is still corrected to the locked evidence.
            if primary is None and gate.get("screening_status") == "No concern visible":
                gate["secondary_hazard_categories"] = []
                gate["uncertainty"] = append_uncertainty(
                    gate.get("uncertainty", ""),
                    "A Pass 1 cue was retained for review, but the image reviewers did not corroborate a target concern.",
                )
                return gate

            primary = categories[0]
            corrected_cue = best_cue_for_category(inventory, primary)
            assert corrected_cue is not None
            gate["primary_hazard_category"] = primary
            gate["screening_status"] = "Possible concern"
            gate["compliance_status"] = "Not assessable"
            gate["visible_trigger"] = corrected_cue.get("evidence") or corrected_cue.get("scenario")
            gate["worker_exposure"] = corrected_cue.get("scenario") or corrected_cue.get("evidence")
            gate["control_status"] = "Unclear"
            gate["evidence"] = corrected_cue.get("evidence") or corrected_cue.get("scenario")
            gate["uncertainty"] = append_uncertainty(
                "; ".join(corrected_cue.get("limitations", [])) or "Not assessable from the image.",
                "The gate selected an unsupported category and was corrected to the locked Pass 1 cue.",
            )

        cue = best_cue_for_category(inventory, primary)
        assert cue is not None
        cue_status = cue.get("status", "Unclear")
        controls = cue.get("controls", [])
        control_statuses = {control.get("status", "Unclear") for control in controls}

        gate["secondary_hazard_categories"] = [category for category in categories if category != primary]
        gate["relevant_osha_rule"] = canonical_retrieved_citation(primary, retrieved_rule_keys)

        if primary == "PPE Compliance Concern":
            missing_controls = [
                control.get("control", "PPE")
                for control in controls
                if control.get("status") == "Not observed"
            ]
            if cue_status == "Visible" and missing_controls:
                gate["screening_status"] = "Confirmed concern"
                gate["compliance_status"] = "Confirmed violation"
                gate["control_status"] = "Not observed"
                gate["visible_trigger"] = cue.get("scenario") or cue.get("evidence")
                gate["worker_exposure"] = cue.get("scenario") or cue.get("evidence")
                gate["evidence"] = cue.get("evidence") or cue.get("scenario")
                gate["uncertainty"] = "; ".join(cue.get("limitations", [])) or "None"
                return gate

        if primary == "Electrical / Electrocution Hazards":
            gate["screening_status"] = "Possible concern"
            gate["compliance_status"] = "Not assessable"
            gate["control_status"] = "Unclear"
            gate["visible_trigger"] = cue.get("evidence") or cue.get("scenario")
            gate["worker_exposure"] = cue.get("scenario") or cue.get("evidence")
            gate["evidence"] = cue.get("evidence") or cue.get("scenario")
            gate["uncertainty"] = append_uncertainty(
                "; ".join(cue.get("limitations", [])),
                ELECTRICAL_UNCERTAINTY,
            )
            return gate

        if cue_status == "Unclear":
            gate["screening_status"] = "Possible concern"
            gate["compliance_status"] = "Not assessable"
            gate["control_status"] = "Unclear"
            gate["uncertainty"] = append_uncertainty(
                gate["uncertainty"],
                "The recorded hazard cue is Unclear, so compliance cannot be confirmed from the image.",
            )
            return gate

        if gate["screening_status"] == "No concern visible":
            gate["screening_status"] = "Possible concern"
            gate["compliance_status"] = "Not assessable"
            gate["uncertainty"] = append_uncertainty(
                gate["uncertainty"],
                "A visible hazard cue exists, but control adequacy requires further assessment.",
            )

        if gate["screening_status"] == "Confirmed concern":
            if not controls or "Unclear" in control_statuses or "Visible" in control_statuses:
                gate["screening_status"] = "Possible concern"
                gate["compliance_status"] = "Not assessable"
                gate["control_status"] = "Unclear"
                gate["uncertainty"] = append_uncertainty(
                    gate["uncertainty"],
                    "A required control was not directly established as absent from the image.",
                )
            elif "Not observed" not in control_statuses:
                gate["screening_status"] = "Possible concern"
                gate["compliance_status"] = "Not assessable"
                gate["control_status"] = "Unclear"
                gate["uncertainty"] = append_uncertainty(
                    gate["uncertainty"],
                    "No required control was explicitly recorded as Not observed.",
                )

        gate["compliance_status"] = norm_compliance_status(
            gate["compliance_status"], gate["screening_status"]
        )
        return gate


    # ══════════════════════════════════════════════════════════════════════════════
    # PASS 2 — image + text; TWO independent reviewer perspectives per hazard,
    # reconciled into one locked gate. This is the multi-perspective analysis step:
    # a "literal evidence" reviewer and a "cautious safety officer" reviewer look at
    # the same image + Pass 1 evidence and answer independently. When they agree,
    # the gate is reconciled with high confidence. When they disagree, the pipeline
    # keeps the more cautious (higher-concern) reading but downgrades certainty and
    # flags the record for human review rather than silently picking one answer.
    # ══════════════════════════════════════════════════════════════════════════════
    def single_category_inventory(inventory: dict[str, Any], category: str) -> dict[str, Any]:
        """Return the target-only evidence card used for deterministic category locking."""
        return {
            **inventory,
            "observed_hazard_cues": [
                cue for cue in inventory.get("observed_hazard_cues", [])
                if cue.get("category") == category
            ],
        }


    def build_retrieval_query(category_inventory: dict[str, Any], category: str) -> str:
        """Create a compact evidence-focused text query for OSHA rule retrieval.

        This avoids embedding JSON punctuation and field boilerplate while retaining
        only the locked visual facts that should drive category-restricted retrieval.
        """
        scene = category_inventory.get("scene", {})
        cue = best_cue_for_category(category_inventory, category) or {}
        controls = cue.get("controls", [])
        control_text = "; ".join(
            f"{item.get('control', 'Control')}: {item.get('status', 'Unclear')}"
            for item in controls
            if isinstance(item, dict)
        ) or "No category-relevant control observation recorded."
        limitations = "; ".join(
            str(item) for item in cue.get("limitations", []) if str(item).strip()
        ) or "None recorded."
        key_objects = ", ".join(
            str(item) for item in scene.get("key_objects", []) if str(item).strip()
        ) or "Unclear"

        return "\n".join(
            [
                "Construction-safety OSHA rule retrieval query.",
                f"Target hazard category: {category}",
                f"Work activity: {scene.get('work_activity', 'Unclear')}",
                f"Key visible objects: {key_objects}",
                f"Observed scenario: {cue.get('scenario', 'Unclear')}",
                f"Visible evidence: {cue.get('evidence', 'Unclear')}",
                f"Observed controls: {control_text}",
                f"Image limitations: {limitations}",
            ]
        )


    def build_category_gate_context(inventory: dict[str, Any], category: str) -> dict[str, Any]:
        """Retain full scene context without letting another category replace the target."""
        target_cues = [
            cue for cue in inventory.get("observed_hazard_cues", [])
            if cue.get("category") == category
        ]
        other_context = [
            {
                "category": cue.get("category"),
                "scenario": cue.get("scenario"),
                "evidence": cue.get("evidence"),
            }
            for cue in inventory.get("observed_hazard_cues", [])
            if cue.get("category") != category
        ]
        return {
            "scene": inventory.get("scene", {}),
            "target_hazard_category": category,
            "target_hazard_cues": target_cues,
            "other_observed_hazard_context": other_context,
            "ppe_observations": inventory.get("ppe_observations", {}),
            "general_limitations": inventory.get("general_limitations", []),
        }


    GATE_REASONING_RULES = """REASONING RULES:
    1. The image is the only source of evidence; the category and evidence boundary are locked.
    2. Never treat Unclear as Not observed.
    3. Do not infer exact heights, depths, voltages, energisation, distances, clearance,
       grounding, lockout/tagout, or unobserved PPE.
    4. Confirmed concern requires a Visible cue, applicable OSHA support, and a relevant
       required control explicitly recorded as Not observed.
    5. Possible concern applies when the cue, control, threshold, or worker exposure is Unclear.
    6. For PPE, a confirmed concern requires the SAME worker, a clear relevant body region,
       and a required PPE item explicitly Not observed.
    7. For Struck-by, stationary equipment proximity alone is insufficient.
    8. For Caught-in/Between, mere proximity to equipment/trenches is insufficient.
    9. For Electrical, do not infer live/energized status or missing controls."""

    GATE_OUTPUT_FORMAT = """Return exactly these ten plain-text lines and no other text:
    SCREENING STATUS: Confirmed concern / Possible concern / No concern visible
    COMPLIANCE STATUS: Confirmed violation / Not assessable / No violation visible
    PRIMARY HAZARD CATEGORY: {category}
    SECONDARY HAZARD CATEGORIES: None
    RELEVANT OSHA RULE: most relevant retrieved citation or Not applicable
    VISIBLE TRIGGER: one concise fact from the image
    WORKER EXPOSURE: one concise fact from the image
    CONTROL STATUS: Visible / Not observed / Unclear / Not applicable
    EVIDENCE: image-based facts only
    UNCERTAINTY: image limitations only, or None
    """


    def build_literal_gate_prompt(category: str, category_inventory_str: str, category_rules_text: str) -> str:
        """Perspective A: a literal, evidence-only reviewer. Reports exactly what is supported."""
        return f"""You are Reviewer A, a construction-safety screening assistant applying retrieved OSHA
    29 CFR Part 1926 rules. This is research screening only, not an official OSHA inspection or
    legal determination. You take a strictly LITERAL evidence stance: report only what is directly
    and unambiguously supported by the image and the locked evidence below.

    Assess ONLY this target hazard category:
    TARGET HAZARD CATEGORY: {category}

    LOCKED PASS 1 CONTEXT FOR {category}:
    {category_inventory_str}

    The target category is locked. Other observed hazards are scene context only and
    must not replace the target category or be reclassified by this gate.

    RETRIEVED OSHA RULES FOR {category}:
    {category_rules_text}

    {GATE_REASONING_RULES}
    10. If in doubt between two ratings, choose the LOWER-concern rating that the visible
        evidence still literally supports.

    {GATE_OUTPUT_FORMAT.format(category=category)}"""


    def build_cautious_gate_prompt(category: str, category_inventory_str: str, category_rules_text: str) -> str:
        """Perspective B: a cautious safety officer. Re-checks whether visible evidence was
        under-weighted, without inventing facts that are not in the locked evidence."""
        return f"""You are Reviewer B, an experienced construction safety officer applying retrieved OSHA
    29 CFR Part 1926 rules. This is research screening only, not an official OSHA inspection or
    legal determination. You take a CAUTIOUS stance: you double-check the image and the locked
    evidence for any worker exposure or missing control that a quick read might understate, but you
    still never invent facts that are not visible in the image or the locked evidence below.

    Assess ONLY this target hazard category:
    TARGET HAZARD CATEGORY: {category}

    LOCKED PASS 1 CONTEXT FOR {category}:
    {category_inventory_str}

    The target category is locked. Other observed hazards are scene context only and
    must not replace the target category or be reclassified by this gate.

    RETRIEVED OSHA RULES FOR {category}:
    {category_rules_text}

    {GATE_REASONING_RULES}
    10. If in doubt between two ratings, choose the HIGHER-concern rating only if the visible
        evidence genuinely supports it — do not escalate on speculation alone.

    {GATE_OUTPUT_FORMAT.format(category=category)}"""


    SCREENING_RANK = {"No concern visible": 0, "Possible concern": 1, "Confirmed concern": 2}


    def reconcile_perspectives(
        gate_a: dict[str, Any], gate_b: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, str]:
        """Merge two independent reviewer gates for the same locked category.

        Returns (reconciled_gate, perspectives_agree, note). When the two perspectives
        disagree on screening status, the more cautious (higher-concern) reading is kept
        but a Confirmed concern is never accepted unless BOTH reviewers confirmed it —
        disagreement always caps the result at Possible concern and is flagged for
        human review.
        """
        rank_a = SCREENING_RANK.get(gate_a["screening_status"], 0)
        rank_b = SCREENING_RANK.get(gate_b["screening_status"], 0)

        if gate_a["screening_status"] == gate_b["screening_status"]:
            merged = gate_a
            merged["uncertainty"] = append_uncertainty(
                merged["uncertainty"], "Two independent reviewer perspectives agreed on this rating."
            )
            return merged, True, "Both reviewer perspectives independently reached the same screening status."

        # Disagreement: never let a lone Confirmed reading stand uncontested.
        more_cautious = gate_a if rank_a >= rank_b else gate_b
        less_cautious = gate_b if more_cautious is gate_a else gate_a
        merged = dict(more_cautious)
        if merged["screening_status"] == "Confirmed concern":
            merged["screening_status"] = "Possible concern"
            merged["compliance_status"] = "Not assessable"
            merged["control_status"] = "Unclear"
        note = (
            "Reviewer A and Reviewer B disagreed on screening status "
            f"({gate_a['screening_status']!r} vs {gate_b['screening_status']!r}). "
            "The more cautious reading was kept, capped below Confirmed concern, and flagged for human review."
        )
        merged["uncertainty"] = append_uncertainty(merged["uncertainty"], note)
        return merged, False, note


    # ══════════════════════════════════════════════════════════════════════════════
    # RISK MATRIX — deterministic severity × probability → risk level + action
    # ══════════════════════════════════════════════════════════════════════════════
    # This is a simple, auditable lookup table (not a model call). It turns the
    # severity/probability pair that Pass 3 already produces into a standard
    # Low/Medium/High/Critical risk rating plus a plain-language recommended action.
    RISK_MATRIX: dict[tuple[str, str], str] = {
        ("High", "High"): "Critical",
        ("High", "Medium"): "High",
        ("High", "Low"): "Medium",
        ("High", "Not assessable from image"): "High",
        ("Medium", "High"): "High",
        ("Medium", "Medium"): "Medium",
        ("Medium", "Low"): "Low",
        ("Medium", "Not assessable from image"): "Medium",
        ("Low", "High"): "Medium",
        ("Low", "Medium"): "Low",
        ("Low", "Low"): "Low",
        ("Low", "Not assessable from image"): "Low",
        ("Minimal", "High"): "Low",
        ("Minimal", "Medium"): "Low",
        ("Minimal", "Low"): "Low",
        ("Minimal", "Not assessable from image"): "Low",
        ("Not assessable", "High"): "Medium",
        ("Not assessable", "Medium"): "Medium",
        ("Not assessable", "Low"): "Low",
        ("Not assessable", "Not assessable from image"): "Medium",
    }

    RISK_ACTIONS: dict[str, str] = {
        "Critical": "Stop the exposed activity and require urgent human review before work continues.",
        "High": "Correct the deficiency or verify the control before work continues; prioritise human review.",
        "Medium": "Schedule corrective action and monitor closely; review when practical.",
        "Low": "Monitor; no immediate corrective action required from this screening alone.",
    }


    def compute_risk(severity: str, probability: str, screening_status: str) -> dict[str, str]:
        """Deterministically map a Pass 3 severity/probability pair to a risk level and action."""
        if screening_status == "No concern visible":
            return {
                "risk_level": "Low",
                "risk_basis": "No target hazard concern was visible in the image.",
                "recommended_action": "No action required from this screening.",
            }
        risk_level = RISK_MATRIX.get((severity, probability), "Medium")
        return {
            "risk_level": risk_level,
            "risk_basis": f"Derived from severity={severity!r} x probability={probability!r} on the standard risk matrix.",
            "recommended_action": RISK_ACTIONS[risk_level],
        }


    def gate_priority(gate: dict[str, Any]) -> tuple[int, int]:
        """Stable legacy primary selection. Full reports are still created for every gate."""
        status_score = {
            "Confirmed concern": 3,
            "Possible concern": 2,
            "No concern visible": 1,
        }.get(gate.get("screening_status"), 0)
        category_score = {
            "Fall Hazards": 5,
            "Electrical / Electrocution Hazards": 4,
            "Struck-by / Equipment Hazards": 3,
            "Caught-in/Between Hazards": 2,
            "PPE Compliance Concern": 1,
        }.get(gate.get("primary_hazard_category"), 0)
        return status_score, category_score


    hazard_gates: list[dict[str, Any]] = []
    hazard_gate_raw_outputs: dict[str, dict[str, str]] = {}
    hazard_retrievals: dict[str, dict[str, Any]] = {}
    perspective_agreement: dict[str, dict[str, Any]] = {}
    retrieved_rule_keys: list[str] = []
    retrieved_rules_text: dict[str, str] = {}
    pass2_seconds = 0.0
    pass2_category_seconds: dict[str, float] = {}

    if not observed_categories:
        print("\n[Pass 2] No target cues in Pass 1 — no hazard gates created.")
        gate_reasoning: dict[str, Any] = {
            "no_hazard": "Pass 2 skipped because Pass 1 recorded no target hazard cue."
        }
        locked_gate = no_cue_gate()
    else:
        print(f"\n[Pass 2] Running two-reviewer OSHA gate per category: {observed_categories}")
        for category in observed_categories:
            category_inventory = single_category_inventory(gate_inventory, category)
            category_gate_context = build_category_gate_context(gate_inventory, category)
            category_inventory_str = json.dumps(category_gate_context, indent=2, ensure_ascii=False)
            retrieval_query = build_retrieval_query(category_inventory, category)
            category_rules_text, category_rule_keys, category_rule_scores = retrieve_rules(
                retrieval_query,
                [category],
                k=RULES_PER_CATEGORY,
            )
            hazard_retrievals[category] = {
                "query": retrieval_query,
                "rule_keys": category_rule_keys,
                "retrieval_scores": category_rule_scores,
                "rules_text": category_rules_text,
            }
            retrieved_rules_text[category] = category_rules_text
            for rule_key in category_rule_keys:
                if rule_key not in retrieved_rule_keys:
                    retrieved_rule_keys.append(rule_key)

            print(f"\n[Pass 2] OSHA gate for: {category} (Reviewer A + Reviewer B)...")
            cuda_sync()
            category_start = time.perf_counter()

            prompt_a = build_literal_gate_prompt(category, category_inventory_str, category_rules_text)
            raw_gate_a = infer_image(IMAGE_PATH, prompt_a, max_tokens=380)
            cuda_sync()

            prompt_b = build_cautious_gate_prompt(category, category_inventory_str, category_rules_text)
            raw_gate_b = infer_image(IMAGE_PATH, prompt_b, max_tokens=380)
            cuda_sync()

            elapsed = time.perf_counter() - category_start
            pass2_seconds += elapsed
            pass2_category_seconds[category] = round(elapsed, 2)
            hazard_gate_raw_outputs[category] = {"reviewer_a": raw_gate_a, "reviewer_b": raw_gate_b}
            if args.verbose:
                print(f"--- Reviewer A ({category}) ---\n{raw_gate_a}")
                print(f"--- Reviewer B ({category}) ---\n{raw_gate_b}")

            gate_a = apply_generic_consistency(parse_gate(raw_gate_a), category_inventory, category_rule_keys)
            gate_b = apply_generic_consistency(parse_gate(raw_gate_b), category_inventory, category_rule_keys)
            gate, agree, reconciliation_note = reconcile_perspectives(gate_a, gate_b)

            # The category-specific inventory is a hard boundary: a model response cannot
            # rename the record into a different hazard after the consistency lock.
            gate["primary_hazard_category"] = category
            gate["secondary_hazard_categories"] = []
            gate["relevant_osha_rule"] = canonical_retrieved_citation(category, category_rule_keys)
            hazard_gates.append(gate)
            perspective_agreement[category] = {
                "reviewer_a_screening_status": gate_a["screening_status"],
                "reviewer_b_screening_status": gate_b["screening_status"],
                "perspectives_agree": agree,
                "note": reconciliation_note,
            }
            print(f"[Pass 2] {category}: reconciled screening_status={gate['screening_status']} | agree={agree}")

        gate_reasoning = hazard_gate_raw_outputs
        locked_gate = max(hazard_gates, key=gate_priority)

    if args.verbose:
        print("\n--- Locked Hazard Gates ---\n", json.dumps(hazard_gates, indent=2, ensure_ascii=False))


    # ══════════════════════════════════════════════════════════════════════════════
    # PASS 3 — text only; one full report per locked hazard gate
    # ══════════════════════════════════════════════════════════════════════════════
    VALID_SEVERITIES = {"High", "Medium", "Low", "Minimal", "Not assessable"}
    VALID_PROBABILITIES = {"High", "Medium", "Low", "Not assessable from image"}


    def normalise_choice(value: Any, allowed: set[str], fallback: str) -> str:
        raw = " ".join(str(value or "").strip().lower().split())
        mapping = {
            "high": "High",
            "medium": "Medium",
            "moderate": "Medium",
            "low": "Low",
            "minimal": "Minimal",
            "not assessable": "Not assessable",
            "not assessable from image": "Not assessable from image",
            "likely": "High",
            "probable": "High",
            "possible": "Medium",
            "unlikely": "Low",
        }
        candidate = mapping.get(raw)
        return candidate if candidate in allowed else fallback


    def build_single_hazard_report_prompt(gate: dict[str, Any], confirmed: str) -> str:
        """Generate severity/probability/control for exactly one already-locked hazard."""
        return f"""You are producing one concise construction-safety screening report.
    This is not an official OSHA inspection, citation, or legal determination.

    Use ONLY the locked gate record below. Do not reinterpret the image, change the
    hazard category, add a second hazard, or add facts that are not in this record.

    LOCKED GATE RECORD:
    SCREENING STATUS: {gate['screening_status']}
    COMPLIANCE STATUS: {gate['compliance_status']}
    HAZARD CATEGORY: {gate['primary_hazard_category']}
    RELEVANT OSHA RULE: {gate['relevant_osha_rule']}
    VISIBLE TRIGGER: {gate['visible_trigger']}
    WORKER EXPOSURE: {gate['worker_exposure']}
    CONTROL STATUS: {gate['control_status']}
    EVIDENCE: {gate['evidence']}
    UNCERTAINTY: {gate['uncertainty']}

    LOCKED VALUES — copy exactly:
    confirmed_violation = {confirmed}
    hazard_category = {gate['primary_hazard_category']}
    osha_reference = {gate['relevant_osha_rule']}
    decision_evidence = {gate['evidence']}

    SEVERITY:
    - High: clear credible potential for death or permanent disability — substantial fall
      exposure, energized electrical exposure near a worker, major crushing/struck-by
      exposure, or a worker inside an unprotected excavation.
    - Medium: potential serious but reversible injury, OR a meaningful hazard exists but
      exposure magnitude is not clear enough to support High.
    - Low: foreseeable minor reversible injury only.
    - Minimal: no meaningful injury mechanism established.
    - Not assessable: insufficient information to identify an injury mechanism.
    Do not automatically classify every elevated work surface as High.

    PROBABILITY:
    - High: exposure visibly active, close, and uncontrolled.
    - Medium: partial control, intermittent access, or uncertain control adequacy.
    - Low: hazard present but visibly well controlled or worker clearly distant.
    - Not assessable from image: worker access/exposure cannot be determined.

    Return ONLY valid JSON with this exact structure:
    {{
      "confirmed_violation": "{confirmed}",
      "hazard_category": "{gate['primary_hazard_category']}",
      "osha_reference": "{gate['relevant_osha_rule']}",
      "screening_status": "{gate['screening_status']}",
      "compliance_status": "{gate['compliance_status']}",
      "scene_summary": "",
      "decision_evidence": "{gate['evidence']}",
      "uncertainty": "{gate['uncertainty']}",
      "potential_harm": "",
      "severity": "",
      "severity_basis": "",
      "probability": "",
      "probability_basis": "",
      "recommended_control": ""
    }}
    """


    def fallback_hazard_report(gate: dict[str, Any], confirmed: str) -> dict[str, Any]:
        """Preserve a valid evidence-locked record if Pass 3 returns malformed JSON."""
        return {
            "confirmed_violation": confirmed,
            "hazard_category": gate["primary_hazard_category"],
            "osha_reference": gate["relevant_osha_rule"],
            "screening_status": gate["screening_status"],
            "compliance_status": gate["compliance_status"],
            "scene_summary": gate["evidence"],
            "decision_evidence": gate["evidence"],
            "uncertainty": gate["uncertainty"],
            "potential_harm": "Potential harm requires human review from the locked image evidence.",
            "severity": "Not assessable",
            "severity_basis": "Pass 3 did not return valid JSON; severity was not inferred.",
            "probability": "Not assessable from image",
            "probability_basis": "Pass 3 did not return valid JSON; probability was not inferred.",
            "recommended_control": "Review the locked image evidence and verify relevant controls before work continues.",
        }


    def finalise_hazard_report(report: dict[str, Any], gate: dict[str, Any], confirmed: str) -> dict[str, Any]:
        """Hard-lock decision fields and retain only valid severity/probability choices."""
        report["confirmed_violation"] = confirmed
        report["hazard_category"] = gate["primary_hazard_category"]
        report["osha_reference"] = gate["relevant_osha_rule"]
        report["screening_status"] = gate["screening_status"]
        report["compliance_status"] = gate["compliance_status"]
        report["decision_evidence"] = gate["evidence"]
        if str(gate["uncertainty"]).strip().lower() not in {"", "none", "not applicable"}:
            report["uncertainty"] = gate["uncertainty"]

        report["severity"] = normalise_choice(
            report.get("severity"), VALID_SEVERITIES, "Not assessable"
        )
        report["probability"] = normalise_choice(
            report.get("probability"), VALID_PROBABILITIES, "Not assessable from image"
        )

        if gate["primary_hazard_category"] == "Electrical / Electrocution Hazards":
            report["confirmed_violation"] = "Unclear"
            report["screening_status"] = "Possible concern"
            report["compliance_status"] = "Not assessable"
            report["scene_summary"] = gate["evidence"]
            report["decision_evidence"] = gate["evidence"]
            report["uncertainty"] = gate["uncertainty"]
            report["potential_harm"] = "Potential electrical contact or arc-related injury if an electrical hazard is present."
            report["severity"] = "Not assessable"
            report["severity_basis"] = "Electrical context is visible, but energized status and exposure conditions cannot be established from one image."
            report["probability"] = "Not assessable from image"
            report["probability_basis"] = "Line/panel status, clearance, isolation, and worker exposure controls cannot be established from the image."
            report["recommended_control"] = (
                "Verify line or panel status, required clearance or isolation procedures, qualified-worker requirements, "
                "and applicable electrical PPE before work continues."
            )

        if not report.get("scene_summary"):
            report["scene_summary"] = gate["evidence"]
        if not report.get("potential_harm"):
            report["potential_harm"] = "Potential injury related to the recorded hazard mechanism."
        if not report.get("severity_basis"):
            report["severity_basis"] = "Based only on the locked visual trigger, worker exposure, and control status."
        if not report.get("probability_basis"):
            report["probability_basis"] = "Based only on the locked visual exposure and control information."
        if not report.get("recommended_control"):
            report["recommended_control"] = "Verify the relevant control and correct any confirmed deficiency before work continues."
        return report


    hazard_reports: list[dict[str, Any]] = []
    raw_pass3: dict[str, str] = {}
    pass3_seconds = 0.0
    pass3_category_seconds: dict[str, float] = {}

    if hazard_gates:
        print(f"\n[Pass 3] Producing full reports for {len(hazard_gates)} supported hazard(s)...")
        for gate in hazard_gates:
            category = gate["primary_hazard_category"]
            confirmed = confirmed_violation_from_compliance(gate["compliance_status"])
            prompt = build_single_hazard_report_prompt(gate, confirmed)
            cuda_sync()
            category_start = time.perf_counter()
            raw_report = infer_text(prompt, max_tokens=520)
            cuda_sync()
            elapsed = time.perf_counter() - category_start
            pass3_seconds += elapsed
            pass3_category_seconds[category] = round(elapsed, 2)
            raw_pass3[category] = raw_report

            parsed: dict[str, Any] | None = None
            report_json = extract_json(raw_report)
            if report_json:
                try:
                    loaded = json.loads(report_json)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                print(f"[Pass 3] Invalid JSON for {category}; using evidence-locked fallback record.")
                parsed = fallback_hazard_report(gate, confirmed)

            report = finalise_hazard_report(parsed, gate, confirmed)

            # Attach the deterministic risk-matrix rating right away, since severity
            # and probability are now locked for this hazard.
            risk = compute_risk(report["severity"], report["probability"], report["screening_status"])
            report["risk_level"] = risk["risk_level"]
            report["risk_basis"] = risk["risk_basis"]
            report["recommended_action"] = risk["recommended_action"]

            hazard_reports.append(report)
            print(
                f"[Pass 3] {category}: severity={report['severity']} | probability={report['probability']} "
                f"| risk={report['risk_level']}"
            )
    else:
        raw_pass3 = {"no_hazard": "Pass 3 skipped because Pass 1 recorded no supported target hazard."}


    # ══════════════════════════════════════════════════════════════════════════════


    # ── Post-Pass-3 human-review routing (no additional VLM pass) ────────────────
    def review_reasons_after_pass3(
        report: dict[str, Any],
        agreement: dict[str, Any] | None,
    ) -> list[str]:
        """Route uncertain, disputed, or high-risk reports without adding Pass 4."""
        reasons: list[str] = []
        if report.get("screening_status") == "Possible concern":
            reasons.append("Screening status is a possible concern and needs image review.")
        if report.get("confirmed_violation") == "Unclear":
            reasons.append("Compliance could not be confirmed from the image alone.")
        if agreement is not None and not agreement.get("perspectives_agree", True):
            reasons.append(f"Pass 2 reviewer perspectives disagreed: {agreement.get('note', 'No note provided.')}")
        if report.get("risk_level") in {"High", "Critical"}:
            reasons.append(f"Risk matrix rated this hazard {report.get('risk_level')}.")
        return reasons


    human_review_queue: list[dict[str, Any]] = []
    for report in hazard_reports:
        category = report["hazard_category"]
        agreement = perspective_agreement.get(category)
        review_reasons = review_reasons_after_pass3(report, agreement)

        report["perspectives_agree"] = agreement["perspectives_agree"] if agreement else True
        report["requires_human_review"] = bool(review_reasons)
        report["human_review_reasons"] = review_reasons

        if review_reasons:
            human_review_queue.append(
                {
                    "hazard_category": category,
                    "screening_status": report["screening_status"],
                    "risk_level": report.get("risk_level"),
                    "reasons": review_reasons,
                    "decision_evidence": report.get("decision_evidence"),
                }
            )

    if human_review_queue:
        print(f"[review routing] {len(human_review_queue)} hazard(s) require human review.")
    else:
        print("[review routing] No hazards required human review.")


    def report_priority(report: dict[str, Any]) -> tuple[int, int, int, int]:
        status_score = {
            "Confirmed concern": 3,
            "Possible concern": 2,
            "No concern visible": 1,
        }.get(report.get("screening_status"), 0)
        severity_score = {
            "High": 4,
            "Medium": 3,
            "Low": 2,
            "Minimal": 1,
            "Not assessable": 0,
        }.get(report.get("severity"), 0)
        probability_score = {
            "High": 3,
            "Medium": 2,
            "Low": 1,
            "Not assessable from image": 0,
        }.get(report.get("probability"), 0)
        category_score = {
            "Fall Hazards": 5,
            "Electrical / Electrocution Hazards": 4,
            "Struck-by / Equipment Hazards": 3,
            "Caught-in/Between Hazards": 2,
            "PPE Compliance Concern": 1,
        }.get(report.get("hazard_category"), 0)
        return status_score, severity_score, probability_score, category_score


    if hazard_reports:
        primary_report = max(hazard_reports, key=report_priority)
        result = dict(primary_report)
        result["hazards"] = hazard_reports
        result["secondary_hazard_categories"] = [
            item["hazard_category"] for item in hazard_reports
            if item is not primary_report
        ]
        result["secondary_hazards"] = [
            item for item in hazard_reports if item is not primary_report
        ]
        result["main_hazard"] = primary_report["hazard_category"]
        result["human_review_queue"] = human_review_queue
        result["requires_human_review"] = bool(human_review_queue)
        result["dataset"] = DATASET
        result["ground_truth_label"] = GROUND_TRUTH
    else:
        no_concern_risk = compute_risk("Minimal", "Low", "No concern visible")
        result = {
            "confirmed_violation": "No",
            "hazard_category": None,
            "osha_reference": "Not applicable",
            "screening_status": "No concern visible",
            "compliance_status": "No violation visible",
            "main_hazard": "No target hazard concern visible",
            "scene_summary": "Pass 1 recorded no target hazard cue.",
            "decision_evidence": "Pass 1 recorded no Fall, Struck-by, Caught-in/Between, Electrical, or supported PPE cue.",
            "uncertainty": "No target hazard cue available for OSHA screening.",
            "potential_harm": "No meaningful target injury mechanism established.",
            "severity": "Minimal",
            "severity_basis": "No supported target hazard cue was recorded.",
            "probability": "Low",
            "probability_basis": "No supported target worker exposure was recorded.",
            "recommended_control": "No target control action determined from this image.",
            "risk_level": no_concern_risk["risk_level"],
            "risk_basis": no_concern_risk["risk_basis"],
            "recommended_action": no_concern_risk["recommended_action"],
            "requires_human_review": False,
            "human_review_queue": [],
            "hazards": [],
            "secondary_hazard_categories": [],
            "secondary_hazards": [],
            "dataset": DATASET,
            "ground_truth_label": GROUND_TRUTH,
        }

    cuda_sync()
    pipeline_seconds = time.perf_counter() - pipeline_start
    result["status"] = "success"
    # Backward-compatible alias retained for existing analysis scripts.
    result["image_processing_runtime_seconds"] = round(pipeline_seconds, 2)
    result["pipeline_runtime_seconds"] = round(pipeline_seconds, 2)
    pipeline_peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    gpu_allocated_end_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "image"
    out_path = OUTPUT_DIR / f"vlm_output_{safe_stem}_{RUN_LABEL}.json"

    final_output = {
        "image_info": {
            "image_path": IMAGE_PATH,
            "image_name": os.path.basename(IMAGE_PATH),
            "input_mode": "image_only",
            "model_id": MODEL_ID,
            "run_label": RUN_LABEL,
            "quantization": QUANTIZATION,
            "device": DEVICE,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "dataset": DATASET,
            "dataset_split": SPLIT,
            "ground_truth_label": GROUND_TRUTH,
            "ppe_focus": PPE_FOCUS,
            "model_load_seconds": round(model_load_seconds, 2),
            "pass1_seconds": round(pass1_seconds, 2),
            "pass1_max_tokens": PASS1_MAX_TOKENS,
            "rules_per_category": RULES_PER_CATEGORY,
            "pass2_seconds": round(pass2_seconds, 2),
            "pass2_category_seconds": pass2_category_seconds,
            "pass3_seconds": round(pass3_seconds, 2),
            "pass3_category_seconds": pass3_category_seconds,
            "hazard_gate_count": len(hazard_gates),
            "full_hazard_report_count": len(hazard_reports),
            "human_review_count": len(human_review_queue),
            "pass4_enabled": False,
            "image_processing_runtime_seconds": round(pipeline_seconds, 2),
            "pipeline_runtime_seconds": round(pipeline_seconds, 2),
            "runtime_excludes_model_load": True,
            "runtime_scope": (
                "Per-image Pass 1–3 processing; excludes one-time model loading "
                "and report-file writing."
            ),
            "model_gpu_allocated_gb": round(model_gpu_allocated_gb, 2),
            "model_gpu_reserved_gb": round(model_gpu_reserved_gb, 2),
            "model_load_peak_gpu_gb": round(model_load_peak_gpu_gb, 2),
            "model_memory_footprint_gb": (
                round(model_memory_footprint_gb, 2)
                if model_memory_footprint_gb is not None
                else None
            ),
            "pipeline_peak_gpu_allocated_gb": round(pipeline_peak_gpu_gb, 2),
            "gpu_allocated_at_end_gb": round(gpu_allocated_end_gb, 2),
        },
        "raw_pass1_output": raw_pass1,
        "raw_pass2_gate_outputs": hazard_gate_raw_outputs,
        "raw_pass3_outputs": raw_pass3,
        "guard_diagnostics": GUARD_DIAGNOSTICS,
        "locked_sparse_inventory": inventory,
        "locked_gate_inventory": gate_inventory,
        "observed_categories": observed_categories,
        "retrieved_rule_keys": retrieved_rule_keys,
        "osha_rules_retrieved": retrieved_rules_text,
        "gate_reasoning": gate_reasoning,
        "perspective_agreement": perspective_agreement,
        "locked_gate_record": locked_gate,
        "locked_hazard_gate_records": hazard_gates,
        "hazard_rule_retrievals": hazard_retrievals,
        "human_review_queue": human_review_queue,
        "vlm_result": result,
    }

    with open(out_path, "w", encoding="utf-8") as output_file:
        json.dump(final_output, output_file, indent=2, ensure_ascii=False)

    write_additional_reports(final_output, out_path)

    user_summary = user_facing_summary(result, final_output["image_info"])
    print("\n===== CONSTRUCTION SAFETY SCREENING RESULT =====")
    print(f"Status: {user_summary['overall_status']}")
    print(f"Main hazard: {user_summary['main_hazard']} | Risk: {user_summary['risk_level']}")
    print(f"Runtime: {user_summary['runtime_label']}")
    print(f"Evidence: {user_summary['evidence']}")
    print(f"Human review: {user_summary['human_review']}")
    print(f"OSHA reference: {user_summary['osha_reference']}")
    if args.verbose:
        print("\n--- Full Developer Result ---\n", json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[output] Developer audit JSON: {out_path}")
    print(
        "[timing detail] "
        f"pass1={pass1_seconds:.2f}s | pass2_all_hazards={pass2_seconds:.2f}s | "
        f"pass3_all_hazards={pass3_seconds:.2f}s | "
        f"per_image_processing={pipeline_seconds:.2f}s (model loading excluded)"
    )
    print(f"[gpu] Pipeline peak allocated: {pipeline_peak_gpu_gb:.2f} GB")

    return result, out_path

def write_batch_summary(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> Path:
    """Write a compact machine-readable summary for a one-image or batch run."""
    successful = [record for record in records if record.get("status") == "success"]
    image_processing_times = [
        float(record.get("image_processing_runtime_seconds", 0))
        for record in successful
        if isinstance(record.get("image_processing_runtime_seconds"), (int, float))
    ]
    summary = {
        "run_label": RUN_LABEL,
        "model_id": MODEL_ID,
        "dataset": DATASET,
        "image_count": len(records) + len(errors),
        "successful_images": len(successful),
        "failed_images": len(errors),
        "mean_image_processing_runtime_seconds": (
            round(sum(image_processing_times) / len(image_processing_times), 2)
            if image_processing_times else None
        ),
        "records": records,
        "errors": errors,
        "note": (
            "Model loading occurs once before the batch and is excluded from "
            "mean_image_processing_runtime_seconds."
        ),
    }
    summary_path = OUTPUT_DIR / f"batch_summary_{RUN_LABEL}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> int:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    batch_start = time.perf_counter()

    print(f"\n[batch] Starting {len(INPUT_IMAGE_PATHS)} image(s). The model is already loaded once.")
    for image_number, image_path in enumerate(INPUT_IMAGE_PATHS, start=1):
        try:
            result, output_path = run_one_image(image_path, image_number, len(INPUT_IMAGE_PATHS))
            image_info = {
                "image_path": image_path,
                "output_path": str(output_path),
                "status": "success",
                "screening_status": result.get("screening_status"),
                "main_hazard": result.get("main_hazard"),
                "risk_level": result.get("risk_level"),
                "requires_human_review": bool(result.get("requires_human_review")),
                "image_processing_runtime_seconds": result.get("image_processing_runtime_seconds"),
            }
            # Runtime lives inside final_output metadata, so read only the just-written
            # JSON if it was not promoted to the top-level result.
            if image_info["image_processing_runtime_seconds"] is None:
                try:
                    saved = json.loads(Path(output_path).read_text(encoding="utf-8"))
                    image_info["image_processing_runtime_seconds"] = (
                        saved.get("image_info", {}).get("image_processing_runtime_seconds")
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            records.append(image_info)
        except KeyboardInterrupt:
            print("\n[batch] Interrupted by user.")
            break
        except Exception as error:
            message = str(error)
            print(f"[ERROR] {os.path.basename(image_path)} failed: {message}")
            errors.append({"image_path": image_path, "error": message})

        if CLEAR_CACHE_BETWEEN_IMAGES and torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch_elapsed = time.perf_counter() - batch_start
    summary_path = write_batch_summary(records, errors)

    print("\n" + "=" * 88)
    print("[batch complete]")
    print(f"Processed: {len(records) + len(errors)} | Successful: {len(records)} | Failed: {len(errors)}")
    print(f"Batch elapsed time: {batch_elapsed:.2f}s")
    print(f"Batch summary: {summary_path}")
    if errors:
        print("[batch] Some images failed. See the batch summary for error details.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
