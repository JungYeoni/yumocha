from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_structural_index import (
    compare_family_friendly_decision,
    compare_scenarios,
    run_family_friendly_weight_transfer_scenarios,
)
from src.evaluation.structural_index import (
    StructuralIndexResult,
    compute_structural_index,
    deflate_structural_cost_indicators,
    load_structural_indicator_manifest,
    load_structural_index_weights,
    standardize_structural_indicators,
    validate_structural_index_input,
    validate_structural_index_weights,
    prepare_structural_index_input,
    prepare_processed_structural_panel,
    run_structural_index_scenarios,
)


def test_deflate_structural_cost_indicators_includes_all_four_monetary_indicators():
    panel = pd.DataFrame(
        {
            "year": [2016, 2016, 2016, 2016],
            "indicator_id": [
                "renter_household_annual_housing_cost_hcc",
                "private_education_cost",
                "postpartum_center_fee",
                "housing_price",
            ],
            "value": [95.718, 95.718, 95.718, 95.718],
        }
    )
    cpi = pd.DataFrame({"연도": [2016], "소비자물가지수": [95.718], "기준연도": ["2020=100"]})

    result = deflate_structural_cost_indicators(panel, cpi)

    assert np.allclose(result["value"], 100.0)
    assert result["price_basis"].eq("real(2020=100)").all()
    assert result.attrs["cpi_base_year"] == "2020=100"
    assert result.attrs["cpi_base_value"] == 100.0


@pytest.mark.parametrize(
    "cpi,base_value,error_match",
    [
        (
            pd.DataFrame({"연도": [2016], "소비자물가지수": [100.0], "기준연도": [None]}),
            100.0,
            "기준연도",
        ),
        (
            pd.DataFrame(
                {
                    "연도": [2016, 2017],
                    "소비자물가지수": [100.0, 101.0],
                    "기준연도": ["2020=100", "2015=100"],
                }
            ),
            100.0,
            "기준연도",
        ),
        (
            pd.DataFrame(
                {"연도": [2016], "소비자물가지수": [float("inf")], "기준연도": ["2020=100"]}
            ),
            100.0,
            "소비자물가지수",
        ),
        (
            pd.DataFrame({"연도": [2016], "소비자물가지수": [100.0], "기준연도": ["2020=100"]}),
            0.0,
            "기준값",
        ),
    ],
)
def test_deflate_structural_cost_indicators_rejects_invalid_cpi_metadata(
    cpi, base_value, error_match
):
    panel = pd.DataFrame({"year": [2016], "indicator_id": ["housing_price"], "value": [100.0]})

    with pytest.raises(ValueError, match=error_match):
        deflate_structural_cost_indicators(panel, cpi, base_value=base_value)


def test_processed_panel_adapter_and_scenarios_allow_nationwide_only_missing():
    rows = []
    for region, values in (("A", (1.0, 10.0)), ("B", (3.0, 30.0)), ("전국", (None, 20.0))):
        for indicator_id, value in zip(("x", "y"), values, strict=True):
            rows.append(
                {
                    "지역": region,
                    "연도": 2022,
                    "지표_id": indicator_id,
                    "processed_value": value,
                    "대분류": "cat",
                    "세부영역": "sub",
                    "방향": "positive",
                    "is_imputed": region == "A",
                }
            )
    panel = prepare_processed_structural_panel(pd.DataFrame(rows))
    weights = pd.DataFrame([{"id": "x", "weight": 0.5}, {"id": "y", "weight": 0.5}])

    results = run_structural_index_scenarios(
        panel,
        weights,
        expected_regions=["A", "B"],
        expected_years=[2022],
    )

    assert set(results) == {"pooled", "yearly"}
    assert all(len(result.indicator_scores) == 4 for result in results.values())
    assert all(len(result.final_index) == 2 for result in results.values())
    assert all("전국" not in set(result.indicator_scores["region"]) for result in results.values())
    assert "is_imputed" in results["pooled"].indicator_scores.columns

    comparison = compare_scenarios(results)
    assert len(comparison) == 2
    assert set(comparison["region"]) == {"A", "B"}
    assert comparison["abs_score_diff"].ge(0).all()
    assert comparison["abs_rank_diff"].ge(0).all()


def test_family_friendly_weight_moves_to_parental_leave_in_2016_and_2017():
    rows = []
    for year in (2016, 2017, 2019):
        for region, values in (("A", (1.0, 4.0)), ("B", (3.0, 2.0)), ("전국", (2.0, 3.0))):
            for indicator_id, value in zip(
                ("parental_leave_usage", "family_friendly_certification_rate"),
                values,
                strict=True,
            ):
                rows.append(
                    {
                        "region": region,
                        "year": year,
                        "indicator_id": indicator_id,
                        "value": value + (year - 2016),
                        "category": "사회·문화",
                        "subcategory": "일·가정 양립 여건",
                        "direction": "positive",
                    }
                )
    panel = pd.DataFrame(rows)
    weights = pd.DataFrame(
        [
            {"id": "parental_leave_usage", "weight": 0.7},
            {"id": "family_friendly_certification_rate", "weight": 0.3},
        ]
    )

    results = run_family_friendly_weight_transfer_scenarios(
        panel,
        weights,
        expected_regions=("A", "B"),
        expected_years=(2016, 2017, 2019),
    )

    scores = results["pooled"].indicator_scores
    transferred = scores["weight_transfer_applied"]
    assert transferred.sum() == 4
    assert scores.loc[transferred, "year"].isin((2016, 2017)).all()
    assert scores.loc[transferred, "effective_score_source"].eq("parental_leave_usage").all()
    for _, group in scores.loc[scores["year"].isin((2016, 2017))].groupby(["region", "year"]):
        assert group["directional_score"].nunique() == 1
        assert np.isclose(
            group["indicator_weighted_contribution"].sum(), group["directional_score"].iloc[0]
        )

    raking_results = run_structural_index_scenarios(
        panel,
        weights,
        expected_regions=("A", "B"),
        expected_years=(2016, 2017, 2019),
    )
    comparison = compare_family_friendly_decision(results, raking_results)
    assert len(comparison) == 6
    assert comparison.loc[comparison["year"].isin((2016, 2017)), "abs_score_diff"].gt(0).any()


@pytest.mark.filterwarnings("ignore:Unknown extension is not supported.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:Conditional Formatting extension is not supported.*:UserWarning"
)
def test_load_structural_index_weights_matches_manifest_and_source_workbook():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_structural_indicator_manifest(repo_root)
    weights = load_structural_index_weights(repo_root)
    source = pd.read_excel(
        repo_root / "data" / "lookup" / "구조환경지표_AHP가중치_3방식.xlsx",
        sheet_name="AHP",
        header=None,
    )
    source_weights = source.iloc[6:34, [4, 12]].copy()
    source_weights.columns = ["code", "source_weight"]
    source_weights["source_weight"] = pd.to_numeric(source_weights["source_weight"], errors="raise")

    validate_structural_index_weights(weights, manifest)
    assert len(weights) == 28
    assert weights["id"].is_unique
    assert manifest["id"].is_unique
    assert set(weights["id"]) == set(manifest["id"])
    assert np.isclose(weights["weight"].sum(), 1.0, atol=1e-9)
    assert weights["code"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    assert weights["name"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    comparison = weights.merge(source_weights, on="code", how="outer", validate="one_to_one")
    assert len(comparison) == 28
    assert comparison[["weight", "source_weight"]].notna().all().all()
    assert np.allclose(comparison["weight"], comparison["source_weight"], atol=1e-15)
    assert np.isclose(source.iloc[1, 12], 0.998, atol=1e-12)
    assert np.isclose(source.iloc[1, 14], 1.0, atol=1e-12)


def test_validate_structural_index_input_basic():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 1.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 2.0,
            },
        ]
    )

    validated = validate_structural_index_input(
        df,
        expected_regions=["A"],
        expected_years=[2022],
        expected_indicator_ids=["x"],
    )
    assert validated.shape[0] == 2


def test_standardize_structural_indicators_pooled():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 1.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 3.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 2.0,
            },
        ]
    )
    result = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2022],
        expected_indicator_ids=["x"],
    )

    assert set(result["region"]) == {"A", "B"}
    assert np.isclose(result.loc[result["region"] == "A", "score_0_100"].iloc[0], 0.0)
    assert np.isclose(result.loc[result["region"] == "B", "score_0_100"].iloc[0], 100.0)


def test_standardize_structural_indicators_without_expected_ids_uses_no_manifest():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 1.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 3.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "value": 2.0,
            },
        ]
    )

    result = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2022],
    )

    assert len(result) == 2


def test_compute_structural_index_scores():
    indicator_scores = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "directional_score": 80.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "y",
                "category": "cat",
                "subcategory": "sub",
                "direction": "positive",
                "directional_score": 20.0,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {"id": "x", "weight": 0.75},
            {"id": "y", "weight": 0.25},
        ]
    )

    result = compute_structural_index(indicator_scores, weights)
    assert isinstance(result, StructuralIndexResult)
    assert result.final_index.loc[0, "final_index"] == 0.75 * 80.0 + 0.25 * 20.0
    assert (
        result.category_scores.loc[0, "category_score"] == result.final_index.loc[0, "final_index"]
    )
    assert (
        result.subcategory_scores.loc[0, "subcategory_score"]
        == result.final_index.loc[0, "final_index"]
    )
    assert result.final_index.loc[0, "missing_indicator_count"] == 0


def test_compute_structural_index_rejects_missing_or_nonfinite_scores():
    import pytest

    weights = pd.DataFrame([{"id": "x", "weight": 1.0}])
    for invalid_score in (np.nan, np.inf, -np.inf):
        indicator_scores = pd.DataFrame(
            [
                {
                    "region": "A",
                    "year": 2022,
                    "indicator_id": "x",
                    "category": "cat",
                    "subcategory": "sub",
                    "directional_score": invalid_score,
                }
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            compute_structural_index(indicator_scores, weights)

        message = str(exc_info.value)
        assert "1" in message
        assert "A" in message
        assert "2022" in message
        assert "x" in message
        assert "부분 지수는 계산하지 않습니다" in message


def test_z_score_does_not_affect_final_index():
    indicator_scores = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "cat",
                "subcategory": "sub",
                "directional_score": 80.0,
                "z_score": -1.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "y",
                "category": "cat",
                "subcategory": "sub",
                "directional_score": 20.0,
                "z_score": 1.0,
            },
        ]
    )
    changed_z_scores = indicator_scores.copy()
    changed_z_scores["z_score"] = [999.0, -999.0]
    weights = pd.DataFrame([{"id": "x", "weight": 0.75}, {"id": "y", "weight": 0.25}])

    original = compute_structural_index(indicator_scores, weights)
    changed = compute_structural_index(changed_z_scores, weights)

    pd.testing.assert_frame_equal(original.final_index, changed.final_index)


def test_pooled_min_max_exact_calculation():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 10.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 30.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 20.0,
            },
        ]
    )
    res = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2022],
        expected_indicator_ids=["i1"],
    )
    a_score = res.loc[res["region"] == "A", "score_0_100"].iloc[0]
    b_score = res.loc[res["region"] == "B", "score_0_100"].iloc[0]
    assert np.isclose(a_score, 0.0)
    assert np.isclose(b_score, 100.0)


def test_pooled_vs_yearly_difference():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2021,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 0.0,
            },
            {
                "region": "B",
                "year": 2021,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 50.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 0.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 100.0,
            },
            {
                "region": "전국",
                "year": 2021,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 25.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 50.0,
            },
        ]
    )
    pooled = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2021, 2022],
        expected_indicator_ids=["i2"],
    )
    yearly = standardize_structural_indicators(
        df,
        method="yearly",
        expected_regions=["A", "B"],
        expected_years=[2021, 2022],
        expected_indicator_ids=["i2"],
    )
    # pooled across years: global min=0, max=100 -> A@2022 vs B@2022 differ under yearly
    pooled_vals = pooled.sort_values(["region", "year"]).reset_index(drop=True)
    yearly_vals = yearly.sort_values(["region", "year"]).reset_index(drop=True)
    assert not np.allclose(pooled_vals["score_0_100"].values, yearly_vals["score_0_100"].values)


def test_negative_direction_flip():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i3",
                "category": "c",
                "subcategory": "s",
                "direction": "negative",
                "value": 10.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "i3",
                "category": "c",
                "subcategory": "s",
                "direction": "negative",
                "value": 30.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "i3",
                "category": "c",
                "subcategory": "s",
                "direction": "negative",
                "value": 20.0,
            },
        ]
    )
    res = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2022],
        expected_indicator_ids=["i3"],
    )
    # raw score: A=0, B=100 ; negative directional -> A=100, B=0
    a_dir = res.loc[res["region"] == "A", "directional_score"].iloc[0]
    b_dir = res.loc[res["region"] == "B", "directional_score"].iloc[0]
    assert np.isclose(a_dir, 100.0)
    assert np.isclose(b_dir, 0.0)


def test_center_or_positive_allowed_for_housework():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "housework_gender_equality",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 0.8,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "housework_gender_equality",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 0.6,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "housework_gender_equality",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 0.7,
            },
        ]
    )
    res = standardize_structural_indicators(
        df,
        method="pooled",
        expected_regions=["A", "B"],
        expected_years=[2022],
        expected_indicator_ids=["housework_gender_equality"],
    )
    assert "directional_score" in res.columns


def test_center_or_positive_rejected_for_non_housework_indicator():
    import pytest

    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "other_indicator",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 1.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "other_indicator",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 2.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "other_indicator",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_positive",
                "value": 1.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="center_or_positive"):
        standardize_structural_indicators(
            df,
            method="pooled",
            expected_regions=["A", "B"],
            expected_years=[2022],
            expected_indicator_ids=["other_indicator"],
        )


def test_center_or_negative_rejected():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "x",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_negative",
                "value": 1.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "x",
                "category": "c",
                "subcategory": "s",
                "direction": "center_or_negative",
                "value": 1.0,
            },
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        standardize_structural_indicators(
            df,
            method="pooled",
            expected_regions=["A"],
            expected_years=[2022],
            expected_indicator_ids=["x"],
        )


def test_pooled_zero_variance_error():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "zv",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 5.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "zv",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 5.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "zv",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 5.0,
            },
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        standardize_structural_indicators(
            df,
            method="pooled",
            expected_regions=["A", "B"],
            expected_years=[2022],
            expected_indicator_ids=["zv"],
        )


def test_yearly_zero_variance_midpoint():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "yzz",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 7.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "yzz",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 7.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "yzz",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 7.0,
            },
        ]
    )
    res = standardize_structural_indicators(
        df,
        method="yearly",
        expected_regions=["A", "B"],
        expected_years=[2022],
        expected_indicator_ids=["yzz"],
    )
    assert np.isclose(res["score_0_100"].iloc[0], 50.0)


def test_duplicate_region_year_indicator_rejected():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "d",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 1.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "d",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 2.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "d",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 1.5,
            },
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        validate_structural_index_input(
            df, expected_regions=["A"], expected_years=[2022], expected_indicator_ids=["d"]
        )


def test_strict_grid_missing_combination():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "g",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 1.0,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "g",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 1.0,
            },
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        validate_structural_index_input(
            df,
            expected_regions=["A", "B"],
            expected_years=[2022],
            expected_indicator_ids=["g"],
            strict_grid=True,
        )


def test_missing_values_rejected_in_standardize():
    df = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "m",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": None,
            },
            {
                "region": "전국",
                "year": 2022,
                "indicator_id": "m",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "value": 1.0,
            },
        ]
    )
    import pytest

    with pytest.raises(ValueError):
        standardize_structural_indicators(
            df,
            method="pooled",
            expected_regions=["A"],
            expected_years=[2022],
            expected_indicator_ids=["m"],
            allow_missing_values=True,
        )


def test_subcategory_and_category_scores_and_contributions():
    indicator_definitions = [
        ("i1", "cat_a", "sub_a", 0.2),
        ("i2", "cat_a", "sub_a", 0.1),
        ("i3", "cat_a", "sub_b", 0.3),
        ("i4", "cat_b", "sub_c", 0.4),
    ]
    regional_scores = {
        "A": [80.0, 20.0, 60.0, 40.0],
        "B": [30.0, 70.0, 10.0, 90.0],
    }
    score_records = []
    for region, scores in regional_scores.items():
        for (indicator_id, category, subcategory, _), score in zip(
            indicator_definitions, scores, strict=True
        ):
            score_records.append(
                {
                    "region": region,
                    "year": 2022,
                    "indicator_id": indicator_id,
                    "category": category,
                    "subcategory": subcategory,
                    "direction": "positive",
                    "directional_score": score,
                }
            )
    indicator_scores = pd.DataFrame(score_records)
    weights = pd.DataFrame(
        [
            {"id": indicator_id, "code": indicator_id, "name": indicator_id, "weight": weight}
            for indicator_id, _, _, weight in indicator_definitions
        ]
    )

    res = compute_structural_index(indicator_scores, weights)

    indicator_to_subcategory = (
        res.indicator_scores.groupby(["region", "year", "category", "subcategory"], sort=True)[
            "indicator_weighted_contribution"
        ]
        .sum()
        .sort_index()
    )
    subcategory_contributions = res.subcategory_scores.set_index(
        ["region", "year", "category", "subcategory"]
    )["subcategory_contribution"].sort_index()
    assert indicator_to_subcategory.index.equals(subcategory_contributions.index)
    np.testing.assert_allclose(
        indicator_to_subcategory.to_numpy(), subcategory_contributions.to_numpy()
    )

    subcategory_to_category = (
        res.subcategory_scores.groupby(["region", "year", "category"], sort=True)[
            "subcategory_contribution"
        ]
        .sum()
        .sort_index()
    )
    category_contributions = res.category_scores.set_index(["region", "year", "category"])[
        "category_contribution"
    ].sort_index()
    assert subcategory_to_category.index.equals(category_contributions.index)
    np.testing.assert_allclose(
        subcategory_to_category.to_numpy(), category_contributions.to_numpy()
    )

    category_to_final = (
        res.category_scores.groupby(["region", "year"], sort=True)["category_contribution"]
        .sum()
        .sort_index()
    )
    final_index = res.final_index.set_index(["region", "year"])["final_index"].sort_index()
    assert category_to_final.index.equals(final_index.index)
    np.testing.assert_allclose(category_to_final.to_numpy(), final_index.to_numpy())

    indicator_to_final = (
        res.indicator_scores.groupby(["region", "year"], sort=True)[
            "indicator_weighted_contribution"
        ]
        .sum()
        .sort_index()
    )
    assert indicator_to_final.index.equals(final_index.index)
    np.testing.assert_allclose(indicator_to_final.to_numpy(), final_index.to_numpy())

    assert res.subcategory_scores["subcategory_score"].between(0, 100).all()
    assert res.category_scores["category_score"].between(0, 100).all()
    assert res.final_index["missing_indicator_count"].eq(0).all()


def test_missing_indicator_count_and_tie_ranking():
    # three regions, two indicators; A and B tie
    scores = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 50.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 50.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 50.0,
            },
            {
                "region": "B",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 50.0,
            },
            {
                "region": "C",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 20.0,
            },
            {
                "region": "C",
                "year": 2022,
                "indicator_id": "i2",
                "category": "c",
                "subcategory": "s",
                "direction": "positive",
                "directional_score": 20.0,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {"id": "i1", "code": "i1", "name": "i1", "weight": 0.5},
            {"id": "i2", "code": "i2", "name": "i2", "weight": 0.5},
        ]
    )
    res = compute_structural_index(scores, weights)
    ranks = res.final_index.set_index("region")["rank"].to_dict()
    assert res.final_index["missing_indicator_count"].eq(0).all()
    assert not res.final_index["has_missing_indicators"].any()
    assert ranks["A"] == 1 and ranks["B"] == 1 and ranks["C"] == 3


def test_prepare_adapter_and_manifest_direction_roundtrip():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_structural_indicator_manifest(repo_root)
    row = manifest[manifest["id"] == "housework_gender_equality"]
    assert not row.empty
    assert row.iloc[0]["direction"] == "center_or_positive"

    df = pd.DataFrame(
        [
            {
                "r": "A",
                "y": 2022,
                "iid": "positive_indicator",
                "val": 1.0,
                "cat": "c",
                "sub": "s",
                "dir": "positive",
            },
            {
                "r": "A",
                "y": 2022,
                "iid": "negative_indicator",
                "val": 2.0,
                "cat": "c",
                "sub": "s",
                "dir": "negative",
            },
            {
                "r": "A",
                "y": 2022,
                "iid": "housework_gender_equality",
                "val": 0.8,
                "cat": "c",
                "sub": "s",
                "dir": "center_or_positive",
            },
        ]
    )
    out = prepare_structural_index_input(
        df,
        region_col="r",
        year_col="y",
        indicator_id_col="iid",
        value_col="val",
        category_col="cat",
        subcategory_col="sub",
        direction_col="dir",
    )
    assert set(
        ["region", "year", "indicator_id", "value", "category", "subcategory", "direction"]
    ).issubset(set(out.columns))
    assert out["direction"].tolist() == ["positive", "negative", "center_or_positive"]


def test_compute_structural_index_rejects_missing_indicator_row():
    scores = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "directional_score": 50.0,
            }
        ]
        + [
            {
                "region": "B",
                "year": 2022,
                "indicator_id": indicator_id,
                "category": "c",
                "subcategory": "s",
                "directional_score": 50.0,
            }
            for indicator_id in ("i1", "i2")
        ]
    )
    weights = pd.DataFrame(
        [
            {"id": "i1", "weight": 0.5},
            {"id": "i2", "weight": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="지역·연도별 지표 구성"):
        compute_structural_index(scores, weights)


def test_compute_structural_index_rejects_duplicate_indicator_row():
    scores = pd.DataFrame(
        [
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "directional_score": 50.0,
            },
            {
                "region": "A",
                "year": 2022,
                "indicator_id": "i1",
                "category": "c",
                "subcategory": "s",
                "directional_score": 60.0,
            },
        ]
    )
    weights = pd.DataFrame([{"id": "i1", "weight": 1.0}])

    with pytest.raises(ValueError, match=r"지역\+연도\+지표 ID 중복"):
        compute_structural_index(scores, weights)


def test_compute_structural_index_rejects_inconsistent_indicator_metadata():
    scores = pd.DataFrame(
        [
            {
                "region": region,
                "year": 2022,
                "indicator_id": "i1",
                "category": category,
                "subcategory": "s",
                "directional_score": 50.0,
            }
            for region, category in (("A", "c1"), ("B", "c2"))
        ]
    )
    weights = pd.DataFrame([{"id": "i1", "weight": 1.0}])

    with pytest.raises(ValueError, match="대분류·세부영역이 일관되지"):
        compute_structural_index(scores, weights)
