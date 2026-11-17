"""Arctander -> GS+LF vocabulary mapping, fixed by name in plan-milestone-3.md (Phase A1).

Exact names map to themselves. Only the spelling variants listed in the plan are added. No
semantic mappings (e.g. lilac -> floral) are made, and the mapping is never fitted to data.
"""

from __future__ import annotations

from collections.abc import Sequence

ARCTANDER_VARIANTS = {
    "ambre": "amber", "camphoraceous": "camphoreous", "caramelic": "caramellic",
    "leather": "leathery", "minty": "mint", "musky": "musk", "sulfuraceous": "sulfurous",
    "vanillin": "vanilla",
}
ARCTANDER_UNMAPPED = ("acid", "coco", "lilac", "mimosa", "mossy", "narcissus",
                      "orange-blossom", "pepper", "rooty", "tarry")


def arctander_mapping(arctander_labels: Sequence[str], gslf_labels: Sequence[str]) -> dict:
    """Map Arctander label -> GS+LF label. Raises if a name is neither mapped nor listed as
    unmapped, so an unexpected vocabulary cannot pass silently."""
    gslf = set(gslf_labels)
    out = {}
    for a in arctander_labels:
        if a in gslf:
            out[a] = a
        elif a in ARCTANDER_VARIANTS:
            if ARCTANDER_VARIANTS[a] not in gslf:
                raise ValueError(f"variant target {ARCTANDER_VARIANTS[a]!r} not a GS+LF label")
            out[a] = ARCTANDER_VARIANTS[a]
        elif a not in ARCTANDER_UNMAPPED:
            raise ValueError(f"Arctander label {a!r} is not covered by the fixed mapping")
    return out
