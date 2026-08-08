import numpy as np
import pandas as pd

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
)
from scripts.build_structural_budget_share_change_sample import (
    BUDGET_SHARE_CHANGE,
    STRUCTURAL_CHANGE,
    build_change_response_sample,
)
from scripts.run_structural_budget_share_change_response import (
    build_region_descriptive_summary,
    run_subarea_change_models,
)
from scripts.build_structural_budget_share_change_visualizations import (
    plot_coefficient_forest,
    plot_region_summary,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    structural_rows = []
    budget_rows = []
    for region_index in range(17):
        region = f"지역{region_index}"
        for year in range(2016, 2025):
            for area_index, (fiscal, structural) in enumerate(
                FISCAL_TO_STRUCTURAL_SUBCATEGORY.items()
            ):
                structural_rows.append(
                    {
                        "region": region,
                        "year": year,
                        "subcategory": structural,
                        "subcategory_score": 30
                        + area_index
                        + region_index
                        + (year - 2016) ** 2
                        + (region_index + 1) * (year - 2016) ** 2 * 0.01,
                    }
                )
                budget_rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부영역": fiscal,
                        "계획예산비중_pct": float(
                            20 - 0.1 * region_index - area_index - (year - 2016) ** 2
                        ),
                        "예산누락주의": False,
                    }
                )
        for year in range(2016, 2025):
            budget_rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "세부영역": "지표체계 외",
                    "계획예산비중_pct": 10.0,
                    "예산누락주의": False,
                }
            )
    return pd.DataFrame(structural_rows), pd.DataFrame(budget_rows)


def test_change_sample_has_exact_time_alignment() -> None:
    structural, budget = _inputs()
    result = build_change_response_sample(structural, budget)
    assert len(result) == 17 * 6 * 11
    assert set(result["기준연도"]) == set(range(2016, 2022))
    assert result["예산_t2연도"].eq(result["기준연도"] + 2).all()
    assert result["예산_t3연도"].eq(result["기준연도"] + 3).all()
    assert result[STRUCTURAL_CHANGE].notna().all()
    assert result[BUDGET_SHARE_CHANGE].notna().all()


def test_change_models_and_region_summary_are_complete() -> None:
    structural, budget = _inputs()
    sample = build_change_response_sample(structural, budget)
    # 완전한 선형관계를 약간 흔들어 군집분산 0 문제를 방지한다.
    sample[BUDGET_SHARE_CHANGE] += np.sin(np.arange(len(sample))) * 0.01
    results = run_subarea_change_models(sample)
    regions = build_region_descriptive_summary(sample)
    assert len(results) == 11
    assert results["관측치"].eq(17 * 6).all()
    assert results["지역수"].eq(17).all()
    assert results["연도수"].eq(6).all()
    assert results["FDR_q값"].between(0, 1).all()
    assert len(regions) == 17
    assert regions["반대방향_변화비율_pct"].between(0, 100).all()
    assert set(regions["대응성_탐색집단"]) == {"상대적으로 높음", "중간", "상대적으로 낮음"}

    forest = plot_coefficient_forest(results)
    region_plot = plot_region_summary(regions)
    assert len(forest.axes) == 1
    assert len(region_plot.axes) == 1
