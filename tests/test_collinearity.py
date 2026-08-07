import numpy as np
import pandas as pd

import pytest

from src.modeling.collinearity import (
    compute_pooled_correlation,
    compute_two_way_fe_residuals,
    compute_vif,
    compute_within_region_correlation,
    flag_high_correlation_pairs,
    pivot_scores_to_wide,
)


def test_pivot_scores_to_wide_builds_region_year_by_group_matrix():
    long_scores = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "세부영역": "고용여건", "점수": 10.0},
            {"지역": "서울", "연도": 2021, "세부영역": "주거안정성", "점수": 20.0},
            {"지역": "부산", "연도": 2021, "세부영역": "고용여건", "점수": 30.0},
            {"지역": "부산", "연도": 2021, "세부영역": "주거안정성", "점수": 40.0},
        ]
    )
    wide = pivot_scores_to_wide(long_scores, group_col="세부영역", value_col="점수")

    assert set(wide.columns) == {"지역", "연도", "고용여건", "주거안정성"}
    seoul = wide.loc[wide["지역"].eq("서울")].iloc[0]
    assert seoul["고용여건"] == 10.0
    assert seoul["주거안정성"] == 20.0


def test_pivot_scores_to_wide_rejects_incomplete_grid():
    long_scores = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "세부영역": "고용여건", "점수": 10.0},
            {"지역": "부산", "연도": 2021, "세부영역": "주거안정성", "점수": 40.0},
        ]
    )
    with pytest.raises(ValueError, match="완전격자"):
        pivot_scores_to_wide(long_scores, group_col="세부영역", value_col="점수")


def test_compute_pooled_correlation_reports_perfect_relationship_and_counts():
    wide = pd.DataFrame(
        {
            "지역": ["A", "A", "B", "B"],
            "연도": [2016, 2017, 2016, 2017],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [2.0, 4.0, 6.0, 8.0],
        }
    )
    corr, counts = compute_pooled_correlation(wide, columns=["X", "Y"], method="pearson")
    assert corr.loc["X", "Y"] == pytest.approx(1.0)
    assert counts.loc["X", "Y"] == 4


def test_compute_within_region_correlation_removes_between_region_effect():
    """지역 간 수준 차이가 만드는 가짜 pooled 상관이 within 상관에서는 사라져야 한다.

    두 지역의 지역 내 연도별 변동 패턴을 정확히 반대 부호로 만들면(§3.3 진단
    3번), 지역 수준 차이 때문에 pooled 상관은 강하게 나오지만 within 상관은
    정확히 0이어야 한다.
    """
    wide = pd.DataFrame(
        {
            "지역": ["A", "A", "A", "B", "B", "B"],
            "연도": [2016, 2017, 2018, 2016, 2017, 2018],
            "X": [-1.0, 0.0, 1.0, 9.0, 10.0, 11.0],
            "Y": [-1.0, 0.0, 1.0, 11.0, 10.0, 9.0],
        }
    )
    pooled_corr, _ = compute_pooled_correlation(wide, columns=["X", "Y"], method="pearson")
    within_corr = compute_within_region_correlation(wide, columns=["X", "Y"])

    assert pooled_corr.loc["X", "Y"] > 0.9
    assert within_corr.loc["X", "Y"] == pytest.approx(0.0, abs=1e-9)


def test_compute_two_way_fe_residuals_sum_to_zero_within_region_and_year():
    wide = pd.DataFrame(
        {
            "지역": ["A", "A", "B", "B"],
            "연도": [2016, 2017, 2016, 2017],
            "X": [10.0, 30.0, 110.0, 130.0],
        }
    )
    residuals = compute_two_way_fe_residuals(wide, columns=["X"])

    assert residuals.groupby(wide["지역"])["X"].sum().abs().max() < 1e-9
    assert residuals.groupby(wide["연도"])["X"].sum().abs().max() < 1e-9


def test_compute_vif_flags_near_collinear_pair_but_not_independent_column():
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(size=n)
    wide = pd.DataFrame(
        {
            "지역": [f"r{i % 5}" for i in range(n)],
            "연도": [2016 + i % 9 for i in range(n)],
            "x1": x1,
            "x2": 2 * x1 + rng.normal(scale=0.01, size=n),
            "x3": rng.normal(size=n),
        }
    )
    vif_table, condition_number = compute_vif(wide, columns=["x1", "x2", "x3"])

    x1_vif = vif_table.loc[vif_table["변수"].eq("x1"), "VIF"].iloc[0]
    x2_vif = vif_table.loc[vif_table["변수"].eq("x2"), "VIF"].iloc[0]
    x3_vif = vif_table.loc[vif_table["변수"].eq("x3"), "VIF"].iloc[0]
    assert x1_vif > 10
    assert x2_vif > 10
    assert x3_vif < 5
    assert condition_number > 10


def test_flag_high_correlation_pairs_respects_threshold():
    corr = pd.DataFrame(
        {
            "A": [1.0, 0.9, 0.1],
            "B": [0.9, 1.0, 0.2],
            "C": [0.1, 0.2, 1.0],
        },
        index=["A", "B", "C"],
    )
    flagged = flag_high_correlation_pairs(corr, threshold=0.7)
    assert len(flagged) == 1
    assert set(flagged.iloc[0][["변수1", "변수2"]]) == {"A", "B"}
