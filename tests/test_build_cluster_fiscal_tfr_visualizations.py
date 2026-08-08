import pandas as pd
import pytest

from scripts.build_cluster_fiscal_tfr_visualizations import (
    build_cluster_trends,
    build_three_cluster_lagged_trends,
    plot_daejeon_sejong_exploratory_coefficients,
    plot_three_cluster_subgroup_coefficients,
    plot_three_cluster_lagged_trends,
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


def test_plot_daejeon_sejong_requires_coefficient_only_rows() -> None:
    rows = []
    for lag in ("t+1", "t+2"):
        for index in range(11):
            rows.append(
                {
                    "군집": 2,
                    "시차": lag,
                    "모형": f"영역 {index}",
                    "계수": index / 100,
                    "p값": float("nan"),
                }
            )
    figure = plot_daejeon_sejong_exploratory_coefficients(pd.DataFrame(rows))
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_yticklabels()) == 11


def test_plot_three_cluster_subgroup_marks_two_region_cluster_without_intervals() -> None:
    rows = []
    for lag in ("t+1", "t+2"):
        for cluster in (1, 2, 3):
            for index in range(11):
                coefficient = (cluster - 2) * 0.01 + index / 100
                rows.append(
                    {
                        "시차": lag,
                        "군집": cluster,
                        "모형": f"영역 {index}",
                        "계수": coefficient,
                        "95%신뢰구간_하한": (float("nan") if cluster == 2 else coefficient - 0.01),
                        "95%신뢰구간_상한": (float("nan") if cluster == 2 else coefficient + 0.01),
                    }
                )
    figure = plot_three_cluster_subgroup_coefficients(pd.DataFrame(rows))
    assert len(figure.axes) == 2
    assert len(figure.axes[0].get_yticklabels()) == 11


def test_build_and_plot_three_cluster_lagged_trends() -> None:
    rows = []
    assignments = []
    for cluster, region in ((1, "가"), (2, "나"), (3, "다")):
        assignments.append({"region": region, "군집_3개": cluster})
        for year in (2020, 2021):
            for subarea, budget in (("A", 10.0), ("B", 20.0)):
                rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부영역": subarea,
                        "인구1인당_실질예산_3개년평균": budget * cluster,
                        "합계출산율_t+1": 1.0 + cluster / 10,
                        "합계출산율_t+2": 0.9 + cluster / 10,
                    }
                )
    trends = build_three_cluster_lagged_trends(pd.DataFrame(rows), pd.DataFrame(assignments))
    assert len(trends) == 6
    assert trends.loc[trends["군집_3개"].eq(2), "실질_1인당_3개년평균예산"].eq(60).all()
    figure = plot_three_cluster_lagged_trends(trends)
    assert len(figure.axes) == 3
