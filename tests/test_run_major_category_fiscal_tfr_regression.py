import numpy as np
import pandas as pd

import pytest

from scripts.run_major_category_fiscal_tfr_regression import (
    aggregate_to_major_category,
    build_major_category_mapping,
    run_mutually_controlled_model,
)


def _structural_scores() -> pd.DataFrame:
    rows = [
        {"subcategory": "고용여건", "category": "경제·고용·주거"},
        {"subcategory": "주거 안정성", "category": "경제·고용·주거"},
        {"subcategory": "경제적 여건", "category": "경제·고용·주거"},
        {"subcategory": "돌봄여건", "category": "가족·생활"},
        {"subcategory": "여가 인프라", "category": "가족·생활"},
        {"subcategory": "가사수행 격차", "category": "가족·생활"},
        {"subcategory": "의료서비스 여건", "category": "보건·안전"},
        {"subcategory": "산후조리 여건", "category": "보건·안전"},
        {"subcategory": "아동안전 수준", "category": "보건·안전"},
        {"subcategory": "일·가정 양립 여건", "category": "사회·문화"},
        {"subcategory": "사회적 가치관", "category": "사회·문화"},
    ]
    return pd.DataFrame(rows)


def test_build_major_category_mapping_uses_structural_taxonomy():
    mapping = build_major_category_mapping(_structural_scores())

    assert mapping["1-1. 고용여건"] == "경제·고용·주거"
    assert mapping["2-1. 돌봄 여건"] == "가족·생활"
    assert mapping["3-3. 아동안전 수준"] == "보건·안전"
    assert mapping["4-2. 사회적 가치관"] == "사회·문화"
    assert len(mapping) == 11


def test_aggregate_to_major_category_sums_lag_budget_within_category():
    category_map = {"1-1. 고용여건": "경제·고용·주거", "1-2. 주거안정성": "경제·고용·주거"}
    sample = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2018, 2018],
            "세부영역": ["1-1. 고용여건", "1-2. 주거안정성"],
            "합계출산율": [0.9, 0.9],
            "인구1인당_실질예산_전년도": [100.0, 50.0],
            "인구1인당_실질예산_전전년도": [80.0, 40.0],
        }
    )

    result = aggregate_to_major_category(sample, category_map)

    row = result.iloc[0]
    assert row["대영역"] == "경제·고용·주거"
    assert row["합계출산율"] == pytest.approx(0.9)
    assert row["인구1인당_실질예산_전년도"] == pytest.approx(150.0)
    assert row["인구1인당_실질예산_전전년도"] == pytest.approx(120.0)


def test_aggregate_to_major_category_propagates_missing_lag_as_group_missing():
    category_map = {"1-1. 고용여건": "경제·고용·주거", "1-2. 주거안정성": "경제·고용·주거"}
    sample = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2016, 2016],
            "세부영역": ["1-1. 고용여건", "1-2. 주거안정성"],
            "합계출산율": [0.9, 0.9],
            "인구1인당_실질예산_전년도": [np.nan, 50.0],
            "인구1인당_실질예산_전전년도": [np.nan, np.nan],
        }
    )

    result = aggregate_to_major_category(sample, category_map)

    assert pd.isna(result.iloc[0]["인구1인당_실질예산_전년도"])


def test_aggregate_to_major_category_rejects_inconsistent_tfr_within_group():
    category_map = {"1-1. 고용여건": "경제·고용·주거", "1-2. 주거안정성": "경제·고용·주거"}
    sample = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2018, 2018],
            "세부영역": ["1-1. 고용여건", "1-2. 주거안정성"],
            "합계출산율": [0.9, 1.1],
            "인구1인당_실질예산_전년도": [100.0, 50.0],
            "인구1인당_실질예산_전전년도": [80.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="합계출산율이 세부영역마다 다릅니다"):
        aggregate_to_major_category(sample, category_map)


def _synthetic_major_category_sample() -> pd.DataFrame:
    categories = ["경제·고용·주거", "가족·생활", "보건·안전", "사회·문화"]
    coefficients = {"경제·고용·주거": 2.0, "가족·생활": -1.0, "보건·안전": 0.0, "사회·문화": 0.0}
    regions = ["서울", "부산", "대구", "광주"]
    years = list(range(2018, 2024))
    rng = np.random.default_rng(7)

    n = len(regions) * len(years)
    log_predictors = {category: rng.normal(size=n) for category in categories}

    rows = []
    index = 0
    for region_index, region in enumerate(regions):
        for year_index, year in enumerate(years):
            region_effect = region_index * 0.4
            year_effect = year_index * 0.2
            row = {"지역": region, "연도": year}
            outcome = region_effect + year_effect
            for category in categories:
                log_predictor = log_predictors[category][index]
                row[f"인구1인당_실질예산_전년도__{category}"] = np.expm1(log_predictor)
                outcome += coefficients[category] * log_predictor
            row["합계출산율"] = outcome
            rows.append(row)
            index += 1
    return pd.DataFrame(rows)


def test_run_mutually_controlled_model_recovers_category_specific_coefficients():
    sample = _synthetic_major_category_sample()
    categories = ["경제·고용·주거", "가족·생활", "보건·안전", "사회·문화"]

    result = run_mutually_controlled_model(
        sample,
        categories=categories,
        lag_column_template="인구1인당_실질예산_전년도__{category}",
    )

    coefficient_by_category = dict(zip(result["모형"], result["계수"], strict=True))
    assert coefficient_by_category["경제·고용·주거"] == pytest.approx(2.0, abs=0.05)
    assert coefficient_by_category["가족·생활"] == pytest.approx(-1.0, abs=0.05)
