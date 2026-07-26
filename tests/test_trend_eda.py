"""지역×연도 추세 EDA 준비 유틸 테스트."""

import pandas as pd
import pytest

from src.features.trend_eda import (
    classify_basic_plan_period,
    prepare_budget_trends,
    reshape_structural_indicators,
)


def test_classify_basic_plan_period():
    assert classify_basic_plan_period(2020).startswith("제3차")
    assert classify_basic_plan_period(2021).startswith("제4차")
    assert classify_basic_plan_period(2025) == "분석기간 외"


def test_reshape_structural_indicators_preserves_missing_and_excludes_nationwide_from_changes():
    df = pd.DataFrame(
        {
            "지역": ["전국", "서울", "부산"],
            "대영역": ["영역"] * 3,
            "세부영역": ["세부"] * 3,
            "세부지표": ["지표"] * 3,
            "검증상태": ["완료"] * 3,
            "2020": [1.5, 1.0, "-"],
            "2021": [2.5, 2.0, 3.0],
        }
    )

    result = reshape_structural_indicators(
        df,
        expected_regions=["서울", "부산"],
        years=[2020, 2021],
    )

    assert len(result) == 6
    busan_2020 = result.query("지역 == '부산' and 연도 == 2020").iloc[0]
    seoul_2021 = result.query("지역 == '서울' and 연도 == 2021").iloc[0]
    nationwide = result.loc[result["지역"].eq("전국")]
    assert pd.isna(busan_2020["측정값"])
    assert busan_2020["결측상태"] == "원자료 결측"
    assert seoul_2021["전년대비변화"] == 1.0
    assert nationwide["전년대비변화"].isna().all()


def test_reshape_structural_indicators_rejects_missing_region():
    df = pd.DataFrame(
        {
            "지역": ["전국", "서울"],
            "대영역": ["영역"] * 2,
            "세부영역": ["세부"] * 2,
            "세부지표": ["지표"] * 2,
            "검증상태": ["완료"] * 2,
            "2020": [1.5, 1.0],
        }
    )

    with pytest.raises(ValueError, match="17개 시도 구성 불일치"):
        reshape_structural_indicators(
            df,
            expected_regions=["서울", "부산"],
            years=[2020],
        )


def test_prepare_budget_trends_adds_yoy_and_plan_period():
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "연도": [2020, 2021, 2020, 2021],
            "당해계획예산_백만원": [100.0, 150.0, 200.0, 180.0],
            "원자료_누락주의": [pd.NA] * 4,
        }
    )

    result = prepare_budget_trends(
        panel,
        expected_regions=["서울", "부산"],
        expected_years=[2020, 2021],
    )

    seoul_2021 = result.query("지역 == '서울' and 연도 == 2021").iloc[0]
    assert seoul_2021["전년대비증감률_pct"] == pytest.approx(50.0)
    assert seoul_2021["기본계획기간"].startswith("제4차")
