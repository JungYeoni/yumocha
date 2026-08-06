import pandas as pd
import pytest

from src.visualization.structural_index import (
    DEFAULT_STRUCTURAL_REGIONS,
    DEFAULT_STRUCTURAL_YEARS,
    build_index_summary,
    plot_region_component_comparison,
    plot_region_contribution_heatmap,
    validate_structural_index_artifacts,
)


def _synthetic_artifacts():
    keys = [
        (region, year) for region in DEFAULT_STRUCTURAL_REGIONS for year in DEFAULT_STRUCTURAL_YEARS
    ]
    final = pd.DataFrame(
        [
            {
                "region": region,
                "year": year,
                "final_index": 50.0,
                "missing_indicator_count": 0,
                "indicator_count": 28,
                "has_missing_indicators": False,
                "rank": 1,
            }
            for region, year in keys
        ]
    )
    category = pd.DataFrame(
        [
            {
                "region": region,
                "year": year,
                "category": category_name,
                "category_contribution": 12.5,
                "category_weight_total": 0.25,
                "category_score": 50.0,
            }
            for region, year in keys
            for category_name in ("가족·생활", "보건·안전", "사회·문화", "경제·고용·주거")
        ]
    )
    subcategory = category.rename(
        columns={
            "category": "subcategory",
            "category_contribution": "subcategory_contribution",
            "category_weight_total": "subcategory_weight_total",
            "category_score": "subcategory_score",
        }
    )
    subcategory["category"] = "가족·생활"
    indicator = pd.DataFrame(
        [
            {
                "region": region,
                "year": year,
                "indicator_id": "indicator",
                "indicator_weighted_contribution": 1.0,
            }
            for region, year in keys
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "region": region,
                "year": year,
                "final_index_pooled": 50.0,
                "rank_pooled": 1,
                "final_index_yearly": 51.0,
                "rank_yearly": 2,
                "score_diff_yearly_minus_pooled": 1.0,
                "abs_score_diff": 1.0,
                "rank_diff_yearly_minus_pooled": 1,
                "abs_rank_diff": 1,
            }
            for region, year in keys
        ]
    )
    sensitivity = comparison[["region", "year", "abs_score_diff", "abs_rank_diff"]].copy()
    return {
        "pooled_final": final.copy(),
        "yearly_final": final.assign(final_index=51.0, rank=2),
        "pooled_indicator": indicator,
        "pooled_subcategory": subcategory,
        "pooled_category": category,
        "yearly_comparison": comparison,
        "family_sensitivity": sensitivity,
        "family_weight_transfer": sensitivity,
    }


def test_structural_index_artifacts_are_complete_and_nationwide_is_excluded():
    artifacts = _synthetic_artifacts()

    validation = validate_structural_index_artifacts(artifacts)
    summary = build_index_summary(artifacts)

    assert validation["final_index_rows"] == 153
    assert len(summary) == 153
    assert "전국" not in set(summary["region"])
    assert summary[["pooled_index", "yearly_index"]].notna().all().all()


def test_structural_index_validation_rejects_nationwide_final_row():
    artifacts = _synthetic_artifacts()
    artifacts["pooled_final"] = pd.concat(
        [
            artifacts["pooled_final"],
            artifacts["pooled_final"].iloc[[0]].assign(region="전국"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="전국 행"):
        validate_structural_index_artifacts(artifacts)


def test_structural_index_validation_rejects_missing_category_region_year():
    artifacts = _synthetic_artifacts()
    artifacts["pooled_category"] = artifacts["pooled_category"].loc[
        ~(
            artifacts["pooled_category"]["region"].eq("제주")
            & artifacts["pooled_category"]["year"].eq(2024)
        )
    ]

    with pytest.raises(ValueError, match="고유 지역·연도 조합 수"):
        validate_structural_index_artifacts(artifacts)


def test_structural_index_validation_rejects_missing_yearly_comparison_summary_column():
    artifacts = _synthetic_artifacts()
    artifacts["yearly_comparison"] = artifacts["yearly_comparison"].drop(columns="abs_score_diff")

    with pytest.raises(ValueError, match="yearly_comparison 필수 컬럼 누락"):
        validate_structural_index_artifacts(artifacts)


def test_region_component_comparison_rejects_unknown_region():
    artifacts = _synthetic_artifacts()

    with pytest.raises(ValueError, match="구성점수에 지역"):
        plot_region_component_comparison(artifacts, region="없는 지역")


def test_region_contribution_heatmap_rejects_unknown_region():
    artifacts = _synthetic_artifacts()

    with pytest.raises(ValueError, match="구성점수에 지역"):
        plot_region_contribution_heatmap(artifacts, region="없는 지역")
