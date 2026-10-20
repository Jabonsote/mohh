"""Estadistica pareada de la comparacion LLaMEA vs clasicos."""

import numpy as np
import pytest

from mohh.llamea.statistics import (
    add_holm,
    holm_adjust,
    minimum_detectable_difference,
    paired_summary,
    wilcoxon_signed_rank,
)


def test_wilcoxon_reports_exact_minimum_for_eleven_positive_pairs():
    statistic, pvalue = wilcoxon_signed_rank(list(range(1, 12)))
    assert statistic is not None
    assert pvalue == pytest.approx(2 / 2 ** 11)


def test_wilcoxon_ignores_all_zero_differences():
    assert wilcoxon_signed_rank([0, 0, 0]) == (None, None)


def test_paired_summary_uses_only_valid_pairs():
    summary = paired_summary([1.0, None, 3.0, np.nan], [0.5, 2.0, 2.0, 1.0])
    assert summary["pairs"] == 2
    assert summary["missing"] == 2
    assert summary["wins"] == 2
    assert summary["losses"] == 0
    assert summary["operator_mean"] == pytest.approx(2.0)
    assert summary["baseline_mean"] == pytest.approx(1.25)
    assert summary["mean_difference"] == pytest.approx(0.75)


def test_paired_summary_does_not_invent_a_p_value_with_one_pair():
    summary = paired_summary([1.0], [0.5])
    assert summary["pairs"] == 1
    assert summary["wilcoxon_p"] is None
    assert summary["ci95_low"] is None and summary["ci95_high"] is None
    assert summary["relative"] == pytest.approx(1.0)


def test_holm_adjust_controls_the_family_and_preserves_none():
    adjusted = holm_adjust([0.01, 0.04, 0.03, None])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[3] is None


def test_add_holm_marks_significant_rows():
    rows = [{"wilcoxon_p": 0.01}, {"wilcoxon_p": 0.5}, {"wilcoxon_p": None}]
    add_holm(rows)
    assert rows[0]["wilcoxon_p_holm"] == pytest.approx(0.02)
    assert rows[0]["significant_holm"] is True
    assert rows[1]["significant_holm"] is False
    assert rows[2]["wilcoxon_p_holm"] is None
    assert rows[2]["significant_holm"] is False


def test_minimum_detectable_difference_is_zero_for_identical_pairs():
    assert minimum_detectable_difference([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert minimum_detectable_difference([1.0], [0.0]) is None


def test_paired_summary_reports_a_point_interval_for_constant_difference():
    summary = paired_summary([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert summary["mean_difference"] == pytest.approx(1.0)
    assert summary["ci95_low"] == pytest.approx(1.0)
    assert summary["ci95_high"] == pytest.approx(1.0)
    assert summary["cohens_dz"] is None
