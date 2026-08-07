import unicodedata
from pathlib import Path

import pandas as pd

from scripts.verify_provisional_against_audited_wide import (
    _find_normalized,
    compare_detail_to_audited_wide,
)


def _sample_detail() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "원본행": 1, "세부사업명": "사업A", "예산액": 100.0},
            {"지역": "서울", "연도": 2021, "원본행": 2, "세부사업명": "사업B", "예산액": 200.0},
        ]
    )


def test_compare_passes_when_everything_matches():
    detail = _sample_detail()
    audited = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "원본행": 1, "세부사업명": "사업A", "당해예산": 100.0},
            {"지역": "서울", "연도": 2021, "원본행": 2, "세부사업명": "사업B", "당해예산": 200.0},
        ]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    assert qa["판정"].eq("PASS").sum() >= 4
    assert mismatches.loc[mismatches["불일치유형"].eq("예산 불일치(미확인)")].empty


def test_compare_flags_unexpected_budget_mismatch():
    detail = _sample_detail()
    audited = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "원본행": 1, "세부사업명": "사업A", "당해예산": 999.0},
            {"지역": "서울", "연도": 2021, "원본행": 2, "세부사업명": "사업B", "당해예산": 200.0},
        ]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    row = qa.loc[qa["검사항목"].eq("예산 불일치(알려진 보정 제외)")].iloc[0]
    assert row["판정"] == "FAIL"
    assert row["실제값"] == 1


def test_compare_treats_known_correction_as_expected_not_a_failure():
    detail = pd.DataFrame(
        [
            {
                "지역": "경북",
                "연도": 2018,
                "원본행": 8200,
                "세부사업명": "여성지도자 육성",
                "예산액": 20.0,
            }
        ]
    )
    audited = pd.DataFrame(
        [
            {
                "지역": "경북",
                "연도": 2018,
                "원본행": 8200,
                "세부사업명": "여성지도자 육성",
                "당해예산": -20.0,
            }
        ]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    unexpected_row = qa.loc[qa["검사항목"].eq("예산 불일치(알려진 보정 제외)")].iloc[0]
    assert unexpected_row["판정"] == "PASS"
    known_row = qa.loc[qa["검사항목"].eq("예산 불일치(알려진 보정만, 참고용)")].iloc[0]
    assert known_row["실제값"] == 1


def test_compare_treats_known_subtotal_row_as_expected_not_a_failure():
    detail = pd.DataFrame(
        [{"지역": "전남", "연도": 2017, "원본행": 1, "세부사업명": "사업A", "예산액": 100.0}]
    )
    audited = pd.DataFrame(
        [
            {"지역": "전남", "연도": 2017, "원본행": 1, "세부사업명": "사업A", "당해예산": 100.0},
            {
                "지역": "전남",
                "연도": 2017,
                "원본행": 9505,
                "세부사업명": "소 계",
                "당해예산": 500.0,
            },
        ]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    unexpected_row = qa.loc[qa["검사항목"].str.contains("알려진 소계 행 제외")].iloc[0]
    assert unexpected_row["판정"] == "PASS"
    known_row = qa.loc[qa["검사항목"].str.contains("알려진 소계 행만")].iloc[0]
    assert known_row["실제값"] == 1
    assert mismatches.loc[mismatches["불일치유형"].eq("감사wide에만 존재(미확인)")].empty


def test_compare_flags_unexpected_key_only_in_audited_wide_even_if_row_9505():
    """같은 키(전남/2017/9505)라도 세부사업명이 소계가 아니면 알려진 예외로 봐주지 않는다."""
    detail = pd.DataFrame(
        [{"지역": "전남", "연도": 2017, "원본행": 1, "세부사업명": "사업A", "예산액": 100.0}]
    )
    audited = pd.DataFrame(
        [
            {"지역": "전남", "연도": 2017, "원본행": 1, "세부사업명": "사업A", "당해예산": 100.0},
            {
                "지역": "전남",
                "연도": 2017,
                "원본행": 9505,
                "세부사업명": "실제사업명",
                "당해예산": 500.0,
            },
        ]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    unexpected_row = qa.loc[qa["검사항목"].str.contains("알려진 소계 행 제외")].iloc[0]
    assert unexpected_row["판정"] == "FAIL"
    assert unexpected_row["실제값"] == 1


def test_compare_flags_key_present_only_on_one_side():
    detail = _sample_detail()
    audited = pd.DataFrame(
        [{"지역": "서울", "연도": 2021, "원본행": 1, "세부사업명": "사업A", "당해예산": 100.0}]
    )
    qa, mismatches = compare_detail_to_audited_wide(detail, audited)
    row = qa.loc[qa["검사항목"].str.contains("감사wide에 없음")].iloc[0]
    assert row["판정"] == "FAIL"
    assert row["실제값"] == 1


def test_find_normalized_matches_across_nfc_and_nfd(tmp_path: Path):
    nfd_name = unicodedata.normalize("NFD", "2022_충북_세부사업_정제.csv")
    (tmp_path / nfd_name).write_text("dummy", encoding="utf-8")

    found = _find_normalized(tmp_path, "2022_충북_세부사업_정제.csv")

    assert found is not None
    assert found.name == nfd_name
