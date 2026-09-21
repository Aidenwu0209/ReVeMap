"""Frozen English object-name normalization for prediction-side consensus.

This module never reads ground truth, scene identifiers, files, or model names.
Exact synonym rules and explicit subtype projections are separately recorded.
It intentionally does not merge an object part with the whole object, a book
with a bookshelf, a desk with a table, or a container with a door.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


RULESET_VERSION = "english_object_names_v1_20260915"

# Source: general English lexical relations. No dataset label table was loaded
# to construct these rules. Some example categories were already known from
# earlier development results; the resulting experiment is NOT a held-out test.
_UNKNOWN = frozenset({
    "", "unknown", "unknown object", "unidentified", "unidentified object",
    "unrecognizable", "unrecognisable", "unclear", "none", "n/a", "na",
    "not sure", "cannot identify", "unable to identify",
})

_LEXICAL = {
    "black board": "blackboard", "white board": "whiteboard",
    "book shelf": "bookshelf", "book shelves": "bookshelf",
    "bookshelves": "bookshelf", "books": "book", "chairs": "chair",
    "tables": "table", "desks": "desk", "boxes": "box",
    "cabinets": "cabinet", "shelves": "shelf", "doors": "door",
    "windows": "window", "sofas": "sofa", "couches": "couch",
    "pillows": "pillow", "cushions": "cushion", "curtains": "curtain",
    "blinds": "blind", "bottles": "bottle", "cups": "cup",
    "mugs": "mug", "plates": "plate", "bowls": "bowl",
    "lamps": "lamp", "lights": "light", "containers": "container",
    "bins": "bin", "trash cans": "trash can", "garbage cans": "garbage can",
    "recycling bins": "recycling bin", "cardboard boxes": "cardboard box",
    "office chairs": "office chair", "dining chairs": "dining chair",
    "armchairs": "armchair", "recliners": "recliner",
    "computers": "computer", "monitors": "monitor", "keyboards": "keyboard",
    "backpacks": "backpack", "bags": "bag", "plants": "plant",
    "trashcan": "trash can", "trashbin": "trash bin",
}

_SYNONYMS = {
    "chalkboard": "blackboard", "couch": "sofa", "settee": "sofa",
    "garbage can": "trash can", "garbage bin": "trash can",
    "trash bin": "trash can", "waste bin": "trash can",
    "wastebasket": "trash can", "waste basket": "trash can",
    "rubbish bin": "trash can", "dustbin": "trash can",
    "television": "tv", "television set": "tv", "tv set": "tv",
    "fridge": "refrigerator", "washbasin": "sink", "wash basin": "sink",
    "handbag": "purse", "rucksack": "backpack",
    "power outlet": "electrical outlet", "wall socket": "electrical outlet",
    "light switch": "light switch", "computer display": "monitor",
}

# These are semantic projections, not claims that the phrases are synonyms.
# The original/specific name remains in fine_class for later grounding.
_SUBTYPES = {
    "armchair": "chair", "recliner": "chair", "recliner chair": "chair",
    "reclining chair": "chair", "office chair": "chair", "swivel chair": "chair",
    "desk chair": "chair", "folding chair": "chair", "dining chair": "chair",
    "dining room chair": "chair", "rocking chair": "chair", "plastic chair": "chair",
    "upholstered chair": "chair", "leather chair": "chair", "wooden chair": "chair",
    "lounge chair": "chair", "leather armchair": "chair", "rolling chair": "chair",
    "sectional sofa": "sofa", "sectional couch": "sofa", "loveseat": "sofa",
    "conference table": "table", "meeting table": "table", "dining table": "table",
    "coffee table": "table", "side table": "table", "end table": "table",
    "folding table": "table", "round table": "table", "wooden table": "table",
    "plastic table": "table", "dining room table": "table",
    "cardboard box": "box", "cardboard carton": "box", "cardboard storage box": "box",
    "storage box": "box", "plastic box": "box", "wooden box": "box",
    "recycling bin": "trash can", "recycle bin": "trash can",
    "paper recycling bin": "trash can", "recycling container": "trash can",
    "shipping container": "container", "cargo container": "container",
    "freight container": "container", "storage container": "container",
    "computer monitor": "monitor", "lcd monitor": "monitor", "led monitor": "monitor",
    "desk lamp": "lamp", "floor lamp": "lamp", "table lamp": "lamp",
    "potted plant": "plant", "indoor plant": "plant",
    "water bottle": "bottle", "plastic bottle": "bottle", "glass bottle": "bottle",
    "coffee mug": "mug", "ceramic mug": "mug", "tea cup": "cup",
    "filing cabinet": "cabinet", "file cabinet": "cabinet", "storage cabinet": "cabinet",
    "kitchen cabinet": "cabinet", "bathroom cabinet": "cabinet",
    "metal cabinet": "cabinet", "wooden cabinet": "cabinet",
    "sliding door": "door", "glass door": "door", "wooden door": "door",
    "fire door": "door", "garage door": "door",
}

# Parts remain distinct in consensus; this is explicit provenance only.
_PARTS = {
    "chair back": "chair", "chair backrest": "chair", "chair seat": "chair",
    "chair leg": "chair", "chair arm": "chair", "armrest": None,
    "sofa back": "sofa", "sofa cushion": "sofa", "table leg": "table",
    "tabletop": "table", "table top": "table", "book spine": "book",
    "book cover": "book", "cabinet door": "cabinet", "cabinet handle": "cabinet",
    "door handle": "door", "door frame": "door", "window frame": "window",
    "container door": "container", "shipping container door": "container",
}


def _clean(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label)).lower().strip()
    text = text.strip("\"'` ")
    text = re.sub(r"[.!;,]+$", "", text).strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(?:a|an|the)\s+", "", text)


def normalize_name(label: str, *, project_subtypes: bool = True) -> dict[str, Any]:
    """Return serializable name fields; canonical_name is the consensus token.

    ``project_subtypes=False`` isolates lexical/synonym normalization (A1s).
    Unknown and part detection never uses context or a model confidence score.
    A part is NOT converted into its parent, and broad furniture remains broad.
    The caller must retain its normal observation/geometry validation gates.
    """
    raw = str(label)
    cleaned = _clean(raw)
    if cleaned in _UNKNOWN:
        return {
            "ruleset_version": RULESET_VERSION, "raw_name": raw,
            "cleaned_name": cleaned, "canonical_name": "unknown",
            "standard_class": "unknown", "fine_class": None,
            "relation": "unknown", "is_part": False, "part_of": None,
            "transformations": [], "project_subtypes": project_subtypes,
        }
    name = _LEXICAL.get(cleaned, cleaned)
    transforms: list[dict[str, str]] = []
    if name != cleaned:
        transforms.append({"kind": "lexical", "from": cleaned, "to": name})
    fine = name
    synonym = _SYNONYMS.get(name, name)
    if synonym != name:
        transforms.append({"kind": "synonym", "from": name, "to": synonym})
    name = synonym
    if project_subtypes:
        parent = _SUBTYPES.get(name, name)
        if parent != name:
            transforms.append({"kind": "subtype_projection", "from": name, "to": parent})
        name = parent
    is_part = fine in _PARTS
    relation = transforms[-1]["kind"] if transforms else "identity"
    return {
        "ruleset_version": RULESET_VERSION, "raw_name": raw,
        "cleaned_name": cleaned, "canonical_name": name,
        "standard_class": name, "fine_class": fine,
        "relation": "part" if is_part else relation,
        "is_part": is_part, "part_of": _PARTS.get(fine),
        "transformations": transforms, "project_subtypes": project_subtypes,
    }


def canonicalize_name(label: str, *, project_subtypes: bool = True) -> str:
    """Convenience wrapper for prediction-side vote keys."""
    return normalize_name(label, project_subtypes=project_subtypes)["canonical_name"]


def export_rules() -> dict[str, Any]:
    """A stable JSON-serializable representation of the frozen rule tables."""
    return {
        "version": RULESET_VERSION, "unknown": sorted(_UNKNOWN),
        "lexical": dict(sorted(_LEXICAL.items())),
        "synonyms": dict(sorted(_SYNONYMS.items())),
        "subtype_projections": dict(sorted(_SUBTYPES.items())),
        "parts_never_projected": dict(sorted(_PARTS.items())),
    }
