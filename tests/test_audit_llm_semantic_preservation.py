from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_llm_semantic_preservation import build_audit_candidates, load_year_wide_files


def test_build_audit_candidates_keeps_only_flagged_rows():
    frame = pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "1",
                "세부사업명": "정상 사업",
                "주요내용": "서울시민에게 월 30만원 지원",
                "주요내용_정제": "서울시민에게 월 30만 원 지원",
            },
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "2",
                "세부사업명": "환각 사업",
                "주요내용": "",
                "주요내용_정제": "사업예산 1,200만원 지원",
            },
        ]
    )

    result = build_audit_candidates(frame)

    assert result["원본행"].tolist() == ["2"]
    assert result["자동검출사유"].tolist() == ["빈 원문 변경"]
    assert result["숫자보존"].tolist() == [False]


def test_build_audit_candidates_rejects_duplicate_grain():
    row = {
        "연도": 2021,
        "지역": "서울",
        "원본행": "1",
        "세부사업명": "사업",
        "주요내용": "내용",
        "주요내용_정제": "내용",
    }
    frame = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="키 중복"):
        build_audit_candidates(frame)


def test_load_year_wide_files_rejects_path_scope_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("scripts.audit_llm_semantic_preservation.REGIONS", ("서울",))
    region_root = tmp_path / "서울"
    region_root.mkdir()
    pd.DataFrame(
        [
            {
                "연도": 2024,
                "지역": "부산",
                "원본행": "1",
                "세부사업명": "사업",
                "주요내용": "내용",
                "주요내용_정제": "내용",
            }
        ]
    ).to_csv(region_root / "2023_서울_세부사업_정제.csv", index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="연도·지역 불일치"):
        load_year_wide_files(2023, data_root=tmp_path)
