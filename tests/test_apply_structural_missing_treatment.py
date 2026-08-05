import numpy as np
import pandas as pd

import pytest

from scripts.apply_structural_missing_treatment import (
    attach_block_imputation_flag,
    build_processed_panels,
    build_sensitivity_comparison,
    validate_outputs,
)


def _sample_panel() -> pd.DataFrame:
    # 지표A/서울: 중간 결측 2개 — 2020(차단, pending_review), 2021(비차단, linear_interpolation)
    # 지표A/부산: 선행 결측 2019(비차단=boundary_carry 후보)
    rows = []
    for year, value in zip([2019, 2020, 2021, 2022], [1.0, np.nan, np.nan, 4.0]):
        rows.append(
            {"지역": "서울", "지표_id": "지표A", "지표명": "지표A", "연도": year, "측정값": value}
        )
    for year, value in zip([2019, 2020, 2021, 2022], [np.nan, 2.0, 3.0, 4.0]):
        rows.append(
            {"지역": "부산", "지표_id": "지표A", "지표명": "지표A", "연도": year, "측정값": value}
        )
    return pd.DataFrame(rows)


def _sample_mapping() -> pd.DataFrame:
    rows = [
        {
            "지역": "서울",
            "지표_id": "지표A",
            "연도": 2020,
            "imputation_policy": "pending_review",
            "block_imputation": True,
            "policy_risk_level": "not_applicable",
        },
        {
            "지역": "서울",
            "지표_id": "지표A",
            "연도": 2021,
            "imputation_policy": "linear_interpolation",
            "block_imputation": False,
            "policy_risk_level": "not_applicable",
        },
        {
            "지역": "부산",
            "지표_id": "지표A",
            "연도": 2019,
            "imputation_policy": "boundary_carry",
            "block_imputation": False,
            "policy_risk_level": "medium",
        },
    ]
    return pd.DataFrame(rows)


def test_attach_block_imputation_flag_merges_and_defaults_observed_to_false():
    panel = _sample_panel()
    mapping = _sample_mapping()

    merged = attach_block_imputation_flag(panel, mapping)

    seoul_2020 = merged.loc[merged["지역"].eq("서울") & merged["연도"].eq(2020)].iloc[0]
    assert bool(seoul_2020["block_imputation"]) is True
    seoul_2022 = merged.loc[merged["지역"].eq("서울") & merged["연도"].eq(2022)].iloc[0]
    assert bool(seoul_2022["block_imputation"]) is False


def test_attach_block_imputation_flag_rejects_observed_row_in_mapping():
    panel = _sample_panel()
    mapping = _sample_mapping().copy()
    mapping.loc[len(mapping)] = {
        "지역": "서울",
        "지표_id": "지표A",
        "연도": 2022,
        "imputation_policy": "linear_interpolation",
        "block_imputation": False,
        "policy_risk_level": "not_applicable",
    }

    with pytest.raises(ValueError, match="실측 행이 결측정책 매핑에 존재"):
        attach_block_imputation_flag(panel, mapping)


def test_build_processed_panels_blocks_pending_and_holds_boundary():
    panel = _sample_panel()
    mapping = _sample_mapping()
    merged = attach_block_imputation_flag(panel, mapping)

    main_panel, alternative_panel = build_processed_panels(
        merged, expected_years=[2019, 2020, 2021, 2022]
    )

    seoul_2020_main = main_panel.loc[
        main_panel["지역"].eq("서울") & main_panel["연도"].eq(2020)
    ].iloc[0]
    assert pd.isna(seoul_2020_main["processed_value"])  # 차단된 셀은 본계열에서도 NA

    seoul_2021_main = main_panel.loc[
        main_panel["지역"].eq("서울") & main_panel["연도"].eq(2021)
    ].iloc[0]
    # 2020이 구조적 결측(차단)이라 세그먼트가 2019|2020-2021-2022로 끊긴다. 2021의 세그먼트에는
    # 관측값이 2022 하나뿐이라(2019는 다른 세그먼트) 선형보간에 필요한 좌우 관측값을 못 채워
    # NA로 남고 처리전략도 EXCLUDE_ANALYSIS_PERIOD로 강등된다 — 구조적 결측 경계 때문에
    # 보간 불가한 결측은 원본 NA를 보존한다는 모듈 설계와 일치한다.
    assert pd.isna(seoul_2021_main["processed_value"])
    assert not seoul_2021_main["include_in_analysis"]

    busan_2019_main = main_panel.loc[
        main_panel["지역"].eq("부산") & main_panel["연도"].eq(2019)
    ].iloc[0]
    assert busan_2019_main["processed_value"] == 2.0  # hold: 첫 관측값(2020=2.0) 유지

    busan_2019_alt = alternative_panel.loc[
        alternative_panel["지역"].eq("부산") & alternative_panel["연도"].eq(2019)
    ].iloc[0]
    assert pd.isna(busan_2019_alt["processed_value"])  # 대안본은 경계 결측을 제외


def test_build_sensitivity_comparison_filters_to_boundary_carry_only():
    panel = _sample_panel()
    mapping = _sample_mapping()
    merged = attach_block_imputation_flag(panel, mapping)
    main_panel, alternative_panel = build_processed_panels(
        merged, expected_years=[2019, 2020, 2021, 2022]
    )

    comparison = build_sensitivity_comparison(main_panel, alternative_panel, mapping)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["지역"] == "부산"
    assert row["연도"] == 2019
    assert row["processed_value_본계열_carry"] == 2.0
    assert pd.isna(row["processed_value_대안_관측구간제한"])


def test_validate_outputs_passes_for_consistent_synthetic_data():
    panel = _sample_panel()
    mapping = _sample_mapping()
    merged = attach_block_imputation_flag(panel, mapping)
    main_panel, alternative_panel = build_processed_panels(
        merged, expected_years=[2019, 2020, 2021, 2022]
    )
    comparison = build_sensitivity_comparison(main_panel, alternative_panel, mapping)

    qa = validate_outputs(panel, mapping, main_panel, alternative_panel, comparison)

    assert qa["판정"].eq("PASS").all()
