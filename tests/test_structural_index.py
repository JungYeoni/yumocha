from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.structural_index import (
    StructuralIndexResult,
    compute_structural_index,
    load_structural_indicator_manifest,
    load_structural_index_weights,
    standardize_structural_indicators,
    validate_structural_index_input,
    validate_structural_index_weights,
    prepare_structural_index_input,
)


def test_load_structural_index_weights_matches_manifest(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_structural_indicator_manifest(repo_root)
    weights = load_structural_index_weights(repo_root)
    yaml_lines = (
        (repo_root / "configs" / "structural_index_weights.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    yaml_indicator_ids = [
        line.strip()[:-1]
        for line in yaml_lines
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":")
    ]

    validate_structural_index_weights(weights, manifest)
    assert len(yaml_indicator_ids) == 28
    assert len(yaml_indicator_ids) == len(set(yaml_indicator_ids))
    assert len(weights) == 28
    assert weights["id"].is_unique
    assert manifest["id"].is_unique
    assert set(weights["id"]) == set(manifest["id"])
    assert np.isclose(weights["weight"].sum(), 1.0, atol=1e-9)
    assert weights["code"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    assert weights["name"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()


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
