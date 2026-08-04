from pathlib import Path

import pandas as pd
import pytest

from scripts.build_structural_missing_policy import (
    KEY_COLUMNS,
    build_policy_mapping,
    load_policy_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "structural_missing_policy.yaml"
MAPPING_PATH = ROOT / "reports" / "20260804_구조환경지표_결측정책_전수매핑.csv"
QA_PATH = ROOT / "reports" / "20260804_구조환경지표_결측정책_매핑_QA.csv"


def _minimal_config(years):
    return {
        "rules": [
            {
                "rule_id": "test_rule",
                "indicator_ids": ["indicator"],
                "regions": ["서울"],
                "years": years,
                "missing_cause": "조사 비실시 연도",
                "cause_status": "confirmed",
                "imputation_policy": "linear_interpolation",
                "block_imputation": False,
                "analysis_included": True,
                "evidence": "test",
                "note": "test",
            }
        ]
    }


def test_policy_rule_expands_only_the_missing_key():
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울", "서울"],
            "지표_id": ["indicator"] * 3,
            "지표명": ["지표"] * 3,
            "연도": [2020, 2021, 2022],
            "측정값": [1.0, None, 3.0],
            "원본행존재": [True, True, True],
        }
    )

    mapping = build_policy_mapping(panel, _minimal_config([2021]))

    assert len(mapping) == 1
    assert mapping.loc[0, KEY_COLUMNS].tolist() == ["서울", "indicator", 2021]
    assert mapping.loc[0, "missing_position"] == "intermediate_missing"
    assert mapping.loc[0, "imputation_policy"] == "linear_interpolation"
    assert not bool(mapping.loc[0, "block_imputation"])


def test_policy_rule_rejects_observed_rows():
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "지표_id": ["indicator", "indicator"],
            "지표명": ["지표", "지표"],
            "연도": [2020, 2021],
            "측정값": [1.0, None],
            "원본행존재": [True, True],
        }
    )

    with pytest.raises(ValueError, match="실측 행을 선택했습니다"):
        build_policy_mapping(panel, _minimal_config([2020, 2021]))


def test_committed_policy_mapping_is_complete_and_conservative():
    config = load_policy_config(CONFIG_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    assert len(mapping) == config["panel"]["expected_missing"] == 1236
    assert not mapping.duplicated(KEY_COLUMNS).any()
    assert set(mapping["missing_cause"]) <= set(config["allowed_values"]["missing_cause"])
    assert set(mapping["cause_status"]) <= set(config["allowed_values"]["cause_status"])
    assert set(mapping["imputation_policy"]) <= set(config["allowed_values"]["imputation_policy"])

    unresolved = mapping["cause_status"].eq("unresolved")
    assert unresolved.sum() == 65
    assert mapping.loc[unresolved, "imputation_policy"].eq("pending_review").all()
    assert mapping.loc[unresolved, "block_imputation"].all()
    assert (~mapping.loc[unresolved, "analysis_included_after_imputation"]).all()

    family_friendly = mapping["지표_id"].eq("family_friendly_certification_rate")
    youth_regular_national = mapping["지표_id"].eq("youth_regular_employment_rate") & mapping[
        "지역"
    ].eq("전국")
    assert (family_friendly & unresolved).sum() == 56
    assert (youth_regular_national & unresolved).sum() == 9


def test_block_imputation_is_independent_from_missing_cause():
    mapping = pd.read_csv(MAPPING_PATH)
    survey_missing = mapping.loc[mapping["missing_cause"].eq("조사 비실시 연도")]

    assert set(survey_missing["block_imputation"]) == {False, True}
    nonlinear = mapping["지표_id"].eq("housework_gender_equality")
    assert mapping.loc[nonlinear, "imputation_policy"].eq("pending_review").all()
    assert mapping.loc[nonlinear, "block_imputation"].all()


def test_auxiliary_boundaries_are_scenario_only():
    mapping = pd.read_csv(MAPPING_PATH)
    auxiliary = mapping["auxiliary_scenario_policy"].eq("auxiliary_boundary_interpolation")

    assert auxiliary.sum() == 72
    assert mapping.loc[auxiliary, "imputation_policy"].eq("boundary_carry").all()
    assert (
        ~mapping.loc[auxiliary, "imputation_policy"].eq("auxiliary_boundary_interpolation")
    ).all()
    assert set(mapping.loc[auxiliary, "auxiliary_boundary_year"]) == {2015.0, 2025.0}


def test_boundary_carry_records_distance_direction_and_long_range_risk():
    panel = pd.DataFrame(
        {
            "지역": ["서울"] * 5,
            "지표_id": ["indicator"] * 5,
            "지표명": ["지표"] * 5,
            "연도": [2016, 2017, 2018, 2019, 2020],
            "측정값": [None, None, None, None, 10.0],
            "원본행존재": [True] * 5,
        }
    )
    config = _minimal_config([2016, 2017, 2018, 2019])
    config["rules"][0]["imputation_policy"] = "boundary_carry"

    mapping = build_policy_mapping(panel, config)

    assert mapping["nearest_observed_year"].eq(2020).all()
    assert mapping["distance_from_observation"].tolist() == [4, 3, 2, 1]
    assert mapping["carry_direction"].eq("backward").all()
    assert mapping["consecutive_missing_years"].eq(4).all()
    assert mapping["long_range_backcast"].all()
    assert mapping["sensitivity_required"].all()
    assert mapping["policy_risk_level"].eq("high").all()


def test_issue_82_provincial_blockers_total_119():
    mapping = pd.read_csv(MAPPING_PATH)
    provincial_blockers = mapping["지역"].ne("전국") & mapping["block_imputation"]
    family_friendly = provincial_blockers & mapping["지표_id"].eq(
        "family_friendly_certification_rate"
    )
    nonlinear_housework = provincial_blockers & mapping["지표_id"].eq("housework_gender_equality")

    assert family_friendly.sum() == 51
    assert nonlinear_housework.sum() == 68
    assert provincial_blockers.sum() == 119


def test_boundary_carry_artifact_risk_counts():
    mapping = pd.read_csv(MAPPING_PATH)
    boundary = mapping.loc[mapping["imputation_policy"].eq("boundary_carry")]

    assert len(boundary) == 676
    assert boundary["sensitivity_required"].all()
    assert boundary["long_range_backcast"].sum() == 544
    assert boundary["policy_risk_level"].value_counts().to_dict() == {
        "high": 544,
        "medium": 132,
    }


def test_policy_qa_artifact_has_no_failures():
    qa = pd.read_csv(QA_PATH)

    assert len(qa) > 0
    assert qa["판정"].eq("PASS").all()
