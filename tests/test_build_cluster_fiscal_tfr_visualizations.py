import pandas as pd
import pytest

from scripts.build_cluster_fiscal_tfr_visualizations import (
    build_cluster_trends,
    validate_inputs,
)


def test_validate_inputs_requires_17_unique_regions() -> None:
    assignments = pd.DataFrame({"region": ["서울"], "군집_2개": [1], "군집_3개": [1]})
    centers = pd.DataFrame(
        {
            "군집수": [2],
            "군집": [1],
            "가족·생활": [1],
            "경제·고용·주거": [1],
            "보건·안전": [1],
            "사회·문화": [1],
        }
    )
    with pytest.raises(ValueError, match="17개 시도"):
        validate_inputs(assignments, centers)


def test_build_cluster_trends_sums_subareas_before_group_average() -> None:
    sample = pd.DataFrame(
        {
            "지역": ["가", "가", "나", "나"],
            "연도": [2020] * 4,
            "세부영역": ["A", "B", "A", "B"],
            "인구1인당_실질예산_3개년평균": [10.0, 20.0, 30.0, 40.0],
            "합계출산율": [1.0, 1.0, 2.0, 2.0],
        }
    )
    assignments = pd.DataFrame({"region": ["가", "나"], "군집_2개": [1, 1]})
    result = build_cluster_trends(sample, assignments)
    assert result.loc[0, "실질_1인당_3개년평균예산"] == 50.0
    assert result.loc[0, "합계출산율"] == 1.5


def test_build_cluster_trends_does_not_turn_missing_moving_average_into_zero() -> None:
    sample = pd.DataFrame(
        {
            "지역": ["가", "가"],
            "연도": [2017, 2018],
            "세부영역": ["A", "A"],
            "인구1인당_실질예산_3개년평균": [float("nan"), 10.0],
            "합계출산율": [1.1, 1.0],
        }
    )
    assignments = pd.DataFrame({"region": ["가"], "군집_2개": [1]})
    result = build_cluster_trends(sample, assignments)
    assert result["연도"].tolist() == [2018]


def test_build_cluster_trends_excludes_partial_subarea_budget() -> None:
    sample = pd.DataFrame(
        {
            "지역": ["가", "가", "가", "가"],
            "연도": [2020, 2020, 2021, 2021],
            "세부영역": ["A", "B", "A", "B"],
            "인구1인당_실질예산_3개년평균": [10.0, float("nan"), 30.0, 40.0],
            "합계출산율": [1.0, 1.0, 0.9, 0.9],
        }
    )
    assignments = pd.DataFrame({"region": ["가"], "군집_2개": [1]})
    result = build_cluster_trends(sample, assignments)
    assert result["연도"].tolist() == [2021]
    assert result.loc[0, "실질_1인당_3개년평균예산"] == 70.0
