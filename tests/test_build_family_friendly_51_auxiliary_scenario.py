import numpy as np
import pandas as pd

import pytest

from scripts.build_family_friendly_51_auxiliary_scenario import (
    RAKING_METHOD,
    RAKING_YEARS,
    apply_raking_candidates_to_panel,
    build_raking_candidates,
    remove_resolved_rows_from_mapping,
    render_report,
)
from scripts.build_family_friendly_candidates import INDICATOR_ID, REGION_ORDER
from scripts.build_family_friendly_national_candidates import NATIONAL_CUMULATIVE_TOTALS


def _sample_regions() -> list[str]:
    return ["서울", "부산", "대구"]


def _sample_denominators(regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            2017: [1_050, 2_050, 1_550],
            2019: [1_100, 2_100, 1_600],
        },
        index=regions,
    )


def test_build_raking_candidates_sums_to_national_total_for_2019():
    regions = _sample_regions()
    count_2018 = pd.Series([10.0, 20.0, 5.0], index=regions)
    count_2020 = pd.Series([12.0, 24.0, 7.0], index=regions)
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_raking_candidates(
            2019, count_2018, count_2020, denominators, qa, before_year=2018, after_year=2020
        )
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 3
    assert result["연도"].eq(2019).all()
    assert result["방법"].eq(RAKING_METHOD).all()
    assert np.isclose(result["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[2019], atol=1e-6)

    interpolated_seoul = (10.0 + 12.0) / 2
    raking_factor = NATIONAL_CUMULATIVE_TOTALS[2019] / ((10 + 12) / 2 + (20 + 24) / 2 + (5 + 7) / 2)
    expected_seoul_numerator = interpolated_seoul * raking_factor
    seoul_row = result.loc[result["지역"].eq("서울")].iloc[0]
    assert np.isclose(seoul_row["추정_분자"], expected_seoul_numerator, rtol=1e-9)
    assert np.isclose(seoul_row["추정_비율"], expected_seoul_numerator / 1_100 * 100, rtol=1e-9)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()


def test_build_raking_candidates_sums_to_national_total_for_2017():
    regions = _sample_regions()
    count_2016 = pd.Series([9.0, 18.0, 4.0], index=regions)
    count_2018 = pd.Series([10.0, 20.0, 5.0], index=regions)
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_raking_candidates(
            2017, count_2016, count_2018, denominators, qa, before_year=2016, after_year=2018
        )
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 3
    assert result["연도"].eq(2017).all()
    assert result["방법"].eq(RAKING_METHOD).all()
    assert np.isclose(result["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[2017], atol=1e-6)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()


def _sample_candidates() -> pd.DataFrame:
    rows = []
    for year in RAKING_YEARS:
        for i, region in enumerate(REGION_ORDER):
            rows.append(
                {
                    "지역": region,
                    "지표_id": INDICATOR_ID,
                    "연도": year,
                    "추정_분자": 10.0 + i,
                    "사업체수_분모": 1_000,
                    "추정_비율": (10.0 + i) / 1_000 * 100,
                    "방법": RAKING_METHOD,
                    "근거": "테스트용 근거",
                    "반영유형": "raking 추정치 반영(본계열, 이슈 #70 팀 결정 2026-08-06)",
                    "관측상태": "추정",
                    "QA_상태": "PASS",
                }
            )
    return pd.DataFrame(rows)


def _sample_panel() -> pd.DataFrame:
    rows = []
    for year in RAKING_YEARS:
        for region in REGION_ORDER:
            rows.append(
                {
                    "지역": region,
                    "지표_id": INDICATOR_ID,
                    "연도": year,
                    "측정값": float("nan"),
                    "원본행존재": True,
                    "관측상태": "결측",
                }
            )
    rows.append(
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2018,
            "측정값": 1.23,
            "원본행존재": True,
            "관측상태": "관측",
        }
    )
    return pd.DataFrame(rows)


def test_apply_raking_candidates_to_panel_fills_only_the_34_cells():
    panel = _sample_panel()
    candidates = _sample_candidates()

    updated, audit = apply_raking_candidates_to_panel(panel, candidates)

    assert len(audit) == 34
    assert audit["QA_상태"].eq("PASS").all()
    assert audit["반영전값"].isna().all()
    assert audit["관측상태"].eq("추정").all()
    for year in RAKING_YEARS:
        for i, region in enumerate(REGION_ORDER):
            row = updated.loc[updated["지역"].eq(region) & updated["연도"].eq(year)].iloc[0]
            assert row["측정값"] == pytest.approx((10.0 + i) / 1_000 * 100)
            assert row["관측상태"] == "추정"

    unrelated = updated.loc[updated["연도"].eq(2018)].iloc[0]
    assert unrelated["측정값"] == 1.23
    assert unrelated["관측상태"] == "관측"


def test_apply_raking_candidates_to_panel_rejects_existing_value():
    panel = _sample_panel()
    panel.loc[panel["지역"].eq("서울") & panel["연도"].eq(2017), "측정값"] = 999.0
    candidates = _sample_candidates()

    with pytest.raises(ValueError, match="이미 존재"):
        apply_raking_candidates_to_panel(panel, candidates)


def _sample_mapping() -> pd.DataFrame:
    rows = []
    for year in RAKING_YEARS:
        for region in REGION_ORDER:
            rows.append(
                {
                    "지역": region,
                    "지표_id": INDICATOR_ID,
                    "연도": year,
                    "block_imputation": True,
                    "imputation_policy": "pending_review",
                    "auxiliary_scenario_policy": "auxiliary_raking_composition_ratio",
                }
            )
    rows.append(
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2016,
            "block_imputation": True,
            "imputation_policy": "pending_review",
            "auxiliary_scenario_policy": "auxiliary_raking_composition_ratio",
        }
    )
    return pd.DataFrame(rows)


def test_remove_resolved_rows_from_mapping_drops_only_2017_2019():
    mapping = _sample_mapping()
    candidates = _sample_candidates()

    updated, removed = remove_resolved_rows_from_mapping(mapping, candidates)

    assert len(removed) == 34
    assert len(updated) == 1
    assert updated.iloc[0]["연도"] == 2016


def test_render_report_does_not_crash_when_panel_not_applied():
    candidates = _sample_candidates()
    qa = pd.DataFrame(
        [{"구분": "테스트", "검증항목": "항목", "기대값": 1, "실제값": 1, "판정": "PASS"}]
    )
    empty_panel_audit = pd.DataFrame()

    report = render_report(candidates, qa, empty_panel_audit)

    assert "패널 반영 행 수: 0건" in report
    assert "반영 전 기존 값 존재(위반) 건수: 0" in report
