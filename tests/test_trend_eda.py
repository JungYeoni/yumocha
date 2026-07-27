"""지역×연도 추세 EDA 준비 유틸 테스트."""

import pandas as pd
import pytest

from src.features.trend_eda import (
    build_structural_region_summary,
    classify_basic_plan_period,
    prepare_budget_trends,
    render_structural_region_report,
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


def test_structural_region_summary_respects_lower_is_better_direction():
    structural_long = pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "연도": [2020, 2021, 2020, 2021],
            "대영역": ["사회·문화"] * 4,
            "세부영역": ["일·가정 양립"] * 4,
            "세부지표": ["근로시간"] * 4,
            "측정값": [180.0, 170.0, 175.0, 178.0],
            "실측여부": [True] * 4,
            "급등락후보": [False, True, False, False],
        }
    )

    result = build_structural_region_summary(
        structural_long,
        region_order=["서울", "부산"],
    )

    seoul = result.loc[result["지역"].eq("서울")].iloc[0]
    busan = result.loc[result["지역"].eq("부산")].iloc[0]
    assert seoul["방향성기준결과"] == "개선"
    assert seoul["기준연도순위"] == 1
    assert seoul["급등락후보연도"] == "2021"
    assert busan["방향성기준결과"] == "악화"


def test_structural_region_report_contains_every_region_row():
    summary = pd.DataFrame(
        {
            "대영역": ["경제"] * 2,
            "세부영역": ["고용"] * 2,
            "세부지표": ["청년고용률"] * 2,
            "방향성": ["높을수록 양호"] * 2,
            "비교시작연도": [2020] * 2,
            "비교기준연도": [2021] * 2,
            "지역": ["서울", "부산"],
            "시작값": [60.0, 55.0],
            "기준값": [61.0, 57.0],
            "변화량": [1.0, 2.0],
            "방향성기준결과": ["개선", "개선"],
            "기준연도순위": [1, 2],
            "실측연도수": [2, 2],
            "결측연도수": [0, 0],
            "급등락후보연도": ["-", "-"],
        }
    )

    report = render_structural_region_report(summary)

    assert "## 청년고용률" in report
    assert "| 서울 | 60 | 61 | 1 | 개선 | 1 | 2/0 | - |" in report
    assert "| 부산 | 55 | 57 | 2 | 개선 | 2 | 2/0 | - |" in report
