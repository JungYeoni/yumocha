import pandas as pd
import pytest

from scripts.run_cluster_total_fiscal_tfr_regression import (
    PREDICTOR,
    TOTAL_BUDGET,
    build_total_budget_sample,
    run_total_interaction_models,
    validate_full_panel,
)


def _subarea_sample() -> pd.DataFrame:
    rows = []
    for region in ("가", "나"):
        for year in (2018, 2019):
            for subarea in range(11):
                rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부영역": f"영역{subarea}",
                        "인구1인당_실질예산_3개년평균": float(subarea + year),
                        "합계출산율_t+1": 1.0,
                        "합계출산율_t+2": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def test_build_total_budget_sample_sums_exactly_11_subareas() -> None:
    sample = _subarea_sample()
    result = build_total_budget_sample(sample)
    row = result.loc[result["지역"].eq("가") & result["연도"].eq(2018)].iloc[0]
    assert len(result) == 4
    assert row[TOTAL_BUDGET] == pytest.approx(sum(range(2018, 2029)))
    assert row[PREDICTOR] > 0
    assert row["세부영역수"] == 11


def test_build_total_budget_sample_preserves_missing_moving_average() -> None:
    sample = _subarea_sample()
    sample.loc[
        sample["지역"].eq("가") & sample["연도"].eq(2018) & sample["세부영역"].eq("영역0"),
        "인구1인당_실질예산_3개년평균",
    ] = float("nan")
    result = build_total_budget_sample(sample)
    row = result.loc[result["지역"].eq("가") & result["연도"].eq(2018)].iloc[0]
    assert pd.isna(row[TOTAL_BUDGET])
    assert pd.isna(row[PREDICTOR])


def test_build_total_budget_sample_rejects_incomplete_subarea_set() -> None:
    sample = _subarea_sample()
    sample = sample.drop(sample.index[0])
    with pytest.raises(ValueError, match="구성이 불완전"):
        build_total_budget_sample(sample)


def test_validate_full_panel_rejects_nonofficial_shape() -> None:
    with pytest.raises(ValueError, match="17개 시도"):
        validate_full_panel(build_total_budget_sample(_subarea_sample()))


def test_interaction_results_include_both_lags_and_cluster_pairs() -> None:
    rows = []
    for region_id in range(6):
        cluster = 1 if region_id < 3 else 2
        for year in range(2018, 2024):
            budget = 100 + 10 * year + region_id * year
            rows.append(
                {
                    "지역": f"지역{region_id}",
                    "연도": year,
                    "군집_2개": cluster,
                    PREDICTOR: budget / 1000,
                    "합계출산율_t+1": 0.7 + 0.01 * budget,
                    "합계출산율_t+2": 0.6 + 0.02 * budget,
                }
            )
    coefficients, contrasts = run_total_interaction_models(pd.DataFrame(rows), cluster_count=2)
    assert len(coefficients) == 4
    assert coefficients["시차"].value_counts().to_dict() == {"t+1": 2, "t+2": 2}
    assert len(contrasts) == 2
    assert set(contrasts["시차"]) == {"t+1", "t+2"}
