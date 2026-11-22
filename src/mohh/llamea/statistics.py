"""Alias de compatibilidad: la estadistica vive en `mohh.shared.statistics`."""

from mohh.shared.statistics import (
    add_holm,
    finite_pairs,
    holm_adjust,
    minimum_detectable_difference,
    paired_summary,
    wilcoxon_signed_rank,
)

__all__ = [
    "finite_pairs",
    "wilcoxon_signed_rank",
    "paired_summary",
    "holm_adjust",
    "add_holm",
    "minimum_detectable_difference",
]
