import numpy as np
import pandas as pd
import pytest

from scripts.build_final_analysis_visualizations import (
    REAL_PC,
    build_composite_fiscal_panel,
    build_structural_tfr_panel,
    validate_moving_results,
)


def test_structural_tfr_panel_is_complete_and_standardized() -> None:
    structural = pd.DataFrame(
        [(f"지역{r}", year, r + year) for r in range(17) for year in range(2016, 2025)],
        columns=["region", "year", "pooled_index"],
    )
    tfr = pd.DataFrame(
        [
            (f"지역{r}", year, 0.7 + r * 0.01 + year * 0.001)
            for r in range(17)
            for year in range(2016, 2025)
        ],
        columns=["지역", "연도", "합계출산율"],
    )
    result = build_structural_tfr_panel(structural, tfr)
    assert len(result) == 153
    assert np.allclose(result.groupby("지역")["구조환경_시도내_z"].mean(), 0)
    assert np.allclose(result.groupby("지역")["TFR_시도내_z"].mean(), 0)
    assert result.groupby("연도")["구조환경_연도별순위"].nunique().eq(17).all()
    assert result.groupby("연도")["TFR_연도별순위"].nunique().eq(17).all()
    assert set(result["구조환경_연도별순위점수"].unique()) >= {0, 100}
    assert set(result["TFR_연도별순위점수"].unique()) >= {0, 100}
    assert result["절대순위차이"].between(0, 16).all()


def test_structural_tfr_panel_rejects_wrong_year_set() -> None:
    structural = pd.DataFrame(
        [(f"지역{r}", year, r + year) for r in range(17) for year in range(2015, 2024)],
        columns=["region", "year", "pooled_index"],
    )
    tfr = structural.rename(
        columns={"region": "지역", "year": "연도", "pooled_index": "합계출산율"}
    )
    with pytest.raises(ValueError, match="완전 패널"):
        build_structural_tfr_panel(structural, tfr)


def _fiscal_panel() -> pd.DataFrame:
    rows = []
    areas = [f"영역{index}" for index in range(11)] + ["지표체계 외"]
    for region_index in range(17):
        for year in range(2016, 2025):
            for area_index, area in enumerate(areas, start=1):
                rows.append(
                    {
                        "지역": f"지역{region_index}",
                        "연도": year,
                        "세부영역": area,
                        REAL_PC: float(region_index + year + area_index),
                        "예산결측_사업수": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_composite_fiscal_panel_excludes_outside_and_builds_ranks() -> None:
    result = build_composite_fiscal_panel(_fiscal_panel())
    assert len(result) == 153
    assert result["종합재정대응점수_0_100"].between(0, 100).all()
    assert result.groupby("연도")["연도별_종합재정대응순위"].nunique().eq(17).all()
    first = result.loc[result["지역"].eq("지역0") & result["연도"].eq(2016)].iloc[0]
    expected = sum(float(2016 + area_index) for area_index in range(1, 12))
    assert first["지표체계내_실질인구1인당예산_원"] == expected


def test_composite_fiscal_panel_rejects_duplicate_and_constant_totals() -> None:
    fiscal = _fiscal_panel()
    with pytest.raises(ValueError, match="키가 중복"):
        build_composite_fiscal_panel(pd.concat([fiscal, fiscal.iloc[[0]]], ignore_index=True))

    constant = fiscal.copy()
    constant[REAL_PC] = 1.0
    with pytest.raises(ValueError, match="변동이 있어야"):
        build_composite_fiscal_panel(constant)


def test_validate_moving_results_requires_both_lags_and_11_areas() -> None:
    rows = []
    for version in ["3개년평균_t+1", "3개년평균_t+2"]:
        for area in range(11):
            rows.append(
                {
                    "모형버전": version,
                    "모형": f"영역{area}",
                    "계수": 0.01,
                    "95%신뢰구간_하한": -0.01,
                    "95%신뢰구간_상한": 0.03,
                    "FDR_q값": 0.2,
                    "FDR_0.05_유의": False,
                    "관측치": 85,
                }
            )
    result = pd.DataFrame(rows)
    validate_moving_results(result)
    with pytest.raises(ValueError, match="총 22행"):
        validate_moving_results(result.iloc[:-1])
    mismatched = result.copy()
    mismatched.loc[
        mismatched["모형버전"].eq("3개년평균_t+2") & mismatched["모형"].eq("영역10"), "모형"
    ] = "다른영역"
    with pytest.raises(ValueError, match="모형 집합"):
        validate_moving_results(mismatched)
