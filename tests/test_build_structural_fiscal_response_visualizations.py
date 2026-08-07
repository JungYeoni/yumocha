import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.build_structural_fiscal_response_visualizations import (
    plot_aligned_structure_budget_trends,
    plot_fixed_effects_response_scatter,
    plot_fiscal_budget_trends,
)


def _fiscal_panel() -> pd.DataFrame:
    rows = []
    for subarea_index, subarea in enumerate(("1-1. 고용여건", "1-2. 주거안정성"), start=1):
        for year in range(2016, 2025):
            for region_index, region in enumerate(("서울", "부산", "대구"), start=1):
                rows.append(
                    {
                        "연도": year,
                        "세부영역": subarea,
                        "지역": region,
                        "인구1인당_실질예산_원": float(
                            subarea_index * region_index * (year - 2015) * 1_000
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _response_sample() -> pd.DataFrame:
    rows = []
    for region_index, region in enumerate(("서울", "부산", "대구"), start=1):
        for year_index, structural_year in enumerate(range(2016, 2023), start=1):
            structure = float(region_index * year_index + (year_index % 2) * 0.3)
            log_budget = 0.2 * structure + region_index * 0.4 + year_index * 0.1
            rows.append(
                {
                    "세부영역": "1-1. 고용여건",
                    "지역": region,
                    "구조환경연도": structural_year,
                    "예산연도": structural_year + 2,
                    "구조환경지수_t": structure,
                    "인구1인당_실질예산_t+2_원": float(1_000 * year_index * region_index),
                    "log1p_인구1인당_실질예산_t+2": log_budget,
                }
            )
    return pd.DataFrame(rows)


def test_plot_fiscal_budget_trends_draws_subarea_median_and_iqr():
    figure = plot_fiscal_budget_trends(_fiscal_panel())

    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible_axes) == 2
    assert visible_axes[0].get_title(loc="left") == "1-1. 고용여건"
    assert len(visible_axes[0].lines[0].get_xdata()) == 9
    assert visible_axes[0].get_ylim()[0] == pytest.approx(0.0)
    plt.close(figure)


def test_plot_fiscal_budget_trends_rejects_negative_budget():
    fiscal = _fiscal_panel()
    fiscal.loc[0, "인구1인당_실질예산_원"] = -1.0

    with pytest.raises(ValueError, match="음수"):
        plot_fiscal_budget_trends(fiscal)


def test_plot_aligned_structure_budget_trends_draws_two_lines_and_budget_band():
    figure = plot_aligned_structure_budget_trends(_response_sample())

    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible_axes) == 1
    assert len(visible_axes[0].lines) == 3  # 기준선 + 구조환경 + 예산
    assert len(visible_axes[0].collections) == 1
    plt.close(figure)


def test_plot_fixed_effects_response_scatter_draws_points_line_and_interval():
    common = pd.DataFrame(
        {
            "모형": ["1-1. 고용여건"],
            "계수": [0.2],
            "95%신뢰구간_하한": [0.1],
            "95%신뢰구간_상한": [0.3],
            "p값": [0.04],
            "FDR_q값": [0.2],
        }
    )

    figure = plot_fixed_effects_response_scatter(_response_sample(), common)

    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible_axes) == 1
    assert len(visible_axes[0].collections) == 2  # 산점도 + 신뢰구간 음영
    assert len(visible_axes[0].lines) == 3  # 계수선 + 가로·세로 기준선
    plt.close(figure)


def test_t_plus_1_visualizations_use_matching_budget_columns():
    sample = _response_sample().rename(
        columns={
            "인구1인당_실질예산_t+2_원": "인구1인당_실질예산_t+1_원",
            "log1p_인구1인당_실질예산_t+2": "log1p_인구1인당_실질예산_t+1",
        }
    )
    sample["예산연도"] = sample["구조환경연도"] + 1
    common = pd.DataFrame(
        {
            "모형": ["1-1. 고용여건"],
            "계수": [0.2],
            "95%신뢰구간_하한": [0.1],
            "95%신뢰구간_상한": [0.3],
            "p값": [0.04],
            "FDR_q값": [0.2],
        }
    )

    trend = plot_aligned_structure_budget_trends(sample, lag_years=1)
    scatter = plot_fixed_effects_response_scatter(sample, common, lag_years=1)

    assert "1년 후" in trend._suptitle.get_text()
    assert "1년 후" in scatter._suptitle.get_text()
    plt.close(trend)
    plt.close(scatter)
