"""Filter names and guiding prompts of PG-IA-NILUT (paper Table 1). Torch-free."""
from __future__ import annotations

__all__ = ["FILTER_NAMES", "GUIDING_PROMPTS", "guiding_prompts", "name_key"]

FILTER_NAMES = ("Exposure", "Contrast", "Saturation", "Color temperature", "Tint correction")

# (positive, negative) per filter, in FILTER_NAMES order. Only Contrast differs by dataset.
_COMMON = {
    "Exposure": ("Overexposed photo.", "Underexposed photo."),
    "Saturation": ("Full color photo.", "No color photo."),
    "Color temperature": ("Yellow tinted photo.", "Blue tinted photo."),
    "Tint correction": ("Magenta tinted photo.", "Green tinted photo."),
}
_CONTRAST = {"fivek": ("Clear photo.", "Unclear photo."),
             "ppr10k": ("High contrast photo.", "Low contrast photo.")}

GUIDING_PROMPTS = {
    ds: tuple(_CONTRAST[ds] if n == "Contrast" else _COMMON[n] for n in FILTER_NAMES)
    for ds in _CONTRAST
}


def guiding_prompts(dataset: str = "fivek") -> tuple[tuple[str, str], ...]:
    """The five (positive, negative) prompt pairs for ``'fivek'`` or ``'ppr10k'``."""
    try:
        return GUIDING_PROMPTS[dataset.lower()]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; options: {', '.join(GUIDING_PROMPTS)}") from None


def name_key(name: str) -> str:
    """'Color temperature' -> 'color_temperature' (for keyword arguments)."""
    return name.strip().lower().replace(" ", "_")
