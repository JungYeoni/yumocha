import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.build_structural_fiscal_response_visualizations import (
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
