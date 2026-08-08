import pandas as pd
import pytest

from scripts.build_budget_domain_composition_visualizations import (
    MAJOR_ORDER,
    REAL_BUDGET,
    INPUT_SUBAREAS,
    SUBAREA_ORDER,
    build_composition_tables,
    validate_input,
)


def _panel() -> pd.DataFrame:
    rows = []
    for region_index in range(17):
        for year in range(2016, 2025):
            for subarea_index, subarea in enumerate(INPUT_SUBAREAS, start=1):
                rows.append(
                    {
                        "지역": f"지역{region_index}",
                        "연도": year,
                        "세부영역": subarea,
                        REAL_BUDGET: float(subarea_index * 100),
                        "CPI_기준연도": 2024,
                        "예산결측_사업수": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_input_rejects_unknown_subarea() -> None:
    panel = _panel()
    panel.loc[0, "세부영역"] = "알 수 없음"
    with pytest.raises(ValueError, match="11개 세부영역"):
        validate_input(panel)


def test_composition_tables_have_expected_rows_and_shares() -> None:
    detail, major = build_composition_tables(_panel())
    assert len(detail) == 9 * len(SUBAREA_ORDER)
    assert len(major) == 9 * len(MAJOR_ORDER)
    assert detail.groupby("연도")["연도내_세부영역비중_pct"].sum().round(10).eq(100).all()
    assert major.groupby("연도")["연도내_대영역비중_pct"].sum().round(10).eq(100).all()


def test_major_and_subarea_totals_are_identical() -> None:
    detail, major = build_composition_tables(_panel())
    detail_totals = detail.groupby("연도")["실질계획예산_2024년가격_백만원"].sum()
    major_totals = major.groupby("연도")["실질계획예산_2024년가격_백만원"].sum()
    pd.testing.assert_series_equal(detail_totals, major_totals)


def test_indicator_system_outside_budget_is_excluded() -> None:
    detail, major = build_composition_tables(_panel())
    assert "지표체계 외" not in detail["세부영역"].unique()
    expected = sum(range(1, len(SUBAREA_ORDER) + 1)) * 100 * 17
    assert detail.loc[detail["연도"].eq(2016), "실질계획예산_2024년가격_백만원"].sum() == expected
    assert major.loc[major["연도"].eq(2016), "실질계획예산_2024년가격_백만원"].sum() == expected
