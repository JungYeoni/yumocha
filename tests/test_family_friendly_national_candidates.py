from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from scripts.build_family_friendly_national_candidates import (
    INDICATOR_ID,
    NATIONAL_CUMULATIVE_TOTALS,
    TARGET_YEARS,
    apply_national_observations_to_panel,
    build_national_candidates,
    remove_resolved_rows_from_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "reports" / "20260805_가족친화_전국_직접복원_후보.csv"
QA_PATH = ROOT / "reports" / "20260805_가족친화_전국_직접복원_QA.csv"


def _sample_candidates() -> pd.DataFrame:
    denominators = {2016: 1_000, 2017: 2_000, 2019: 3_000, 2020: 4_000, 2021: 5_000}
    return build_national_candidates(denominators, [])


def _sample_panel() -> pd.DataFrame:
    rows = [
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": year,
            "측정값": float("nan"),
            "원본행존재": False,
        }
        for year in TARGET_YEARS
    ]
    rows.append(
        {"지역": "서울", "지표_id": INDICATOR_ID, "연도": 2018, "측정값": 1.23, "원본행존재": True}
    )
    rows.append(
        {
            "지역": "전국",
            "지표_id": "youth_employment_rate",
            "연도": 2016,
            "측정값": 42.0,
            "원본행존재": True,
        }
    )
    return pd.DataFrame(rows)


def _sample_mapping() -> pd.DataFrame:
    rows = [
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": year,
            "block_imputation": True,
            "imputation_policy": "pending_review",
        }
        for year in TARGET_YEARS
    ]
    rows.append(
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2016,
            "block_imputation": True,
            "imputation_policy": "pending_review",
        }
    )
    return pd.DataFrame(rows)


def test_build_national_candidates_computes_expected_ratios():
    denominators = {2016: 1_000, 2017: 2_000, 2019: 3_000, 2020: 4_000, 2021: 5_000}
    qa_records: list[dict[str, object]] = []

    candidates = build_national_candidates(denominators, qa_records)

    assert len(candidates) == 5
    assert set(candidates["연도"]) == set(TARGET_YEARS)
    assert candidates["지역"].eq("전국").all()
    assert candidates["관측상태"].eq("관측").all()
    for year in TARGET_YEARS:
        row = candidates.loc[candidates["연도"].eq(year)].iloc[0]
        expected = NATIONAL_CUMULATIVE_TOTALS[year] / denominators[year] * 100
        assert np.isclose(row["측정값"], expected, rtol=0, atol=1e-15)
    assert pd.DataFrame(qa_records)["판정"].eq("PASS").all()


def test_committed_national_candidate_artifacts_are_complete():
    candidates = pd.read_csv(CANDIDATE_PATH)
    qa = pd.read_csv(QA_PATH)

    assert len(candidates) == 5
    assert set(candidates["연도"]) == set(TARGET_YEARS)
    assert candidates["지역"].eq("전국").all()
    assert candidates["QA_상태"].eq("PASS").all()
    formula = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    assert np.allclose(candidates["측정값"], formula, rtol=0, atol=1e-15)
    assert candidates.set_index("연도")["공식_분자"].to_dict() == NATIONAL_CUMULATIVE_TOTALS
    assert qa["판정"].eq("PASS").all()

    # 2020/2021 cross-check against the already-confirmed province-level national totals.
    assert candidates.set_index("연도").loc[2020, "공식_분자"] == 4_340
    assert candidates.set_index("연도").loc[2021, "공식_분자"] == 4_918


def test_apply_national_observations_to_panel_fills_only_the_five_missing_cells():
    panel = _sample_panel()
    candidates = _sample_candidates()

    updated, audit = apply_national_observations_to_panel(panel, candidates)

    assert len(audit) == 5
    assert audit["QA_상태"].eq("PASS").all()
    assert audit["반영전값"].isna().all()
    for year in TARGET_YEARS:
        mask = (
            updated["지역"].eq("전국")
            & updated["지표_id"].eq(INDICATOR_ID)
            & updated["연도"].eq(year)
        )
        row = updated.loc[mask].iloc[0]
        expected = candidates.set_index("연도").loc[year, "측정값"]
        assert np.isclose(row["측정값"], expected, rtol=0, atol=1e-15)
        assert row["원본행존재"] is True or row["원본행존재"] == 1

    # unrelated rows are untouched
    seoul_row = updated.loc[updated["지역"].eq("서울")].iloc[0]
    assert seoul_row["측정값"] == 1.23
    other_indicator_row = updated.loc[updated["지표_id"].eq("youth_employment_rate")].iloc[0]
    assert other_indicator_row["측정값"] == 42.0


def test_apply_national_observations_to_panel_rejects_conflicting_existing_value():
    panel = _sample_panel()
    panel.loc[panel["연도"].eq(2016), "측정값"] = 999.0  # disagrees with the candidate
    candidates = _sample_candidates()

    with pytest.raises(ValueError, match="예상과 다릅니다"):
        apply_national_observations_to_panel(panel, candidates)


def test_remove_resolved_rows_from_mapping_drops_exactly_the_five_national_rows():
    mapping = _sample_mapping()
    candidates = _sample_candidates()

    updated, removed = remove_resolved_rows_from_mapping(mapping, candidates)

    assert len(removed) == 5
    assert removed["지역"].eq("전국").all()
    assert len(updated) == len(mapping) - 5
    # the unrelated 서울 row must survive
    assert (updated["지역"].eq("서울")).any()


def test_remove_resolved_rows_from_mapping_rejects_unexpected_state():
    mapping = _sample_mapping()
    mapping.loc[mapping["연도"].eq(2016) & mapping["지역"].eq("전국"), "block_imputation"] = False
    candidates = _sample_candidates()

    with pytest.raises(ValueError, match="pending_review/block_imputation"):
        remove_resolved_rows_from_mapping(mapping, candidates)
