from pathlib import Path

import pandas as pd
import pytest

from scripts.build_semantic_change_impact import attach_latest_budget


def test_attach_latest_budget_validates_key_and_cleaned_text(tmp_path: Path):
    data_root = tmp_path / "data"
    region_dir = data_root / "서울"
    region_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "10",
                "세부사업명": "사업",
                "주요내용_정제": "수정후",
                "사업분류재정구분": "자체",
                "당해예산": 100,
                "전년도예산": 90,
                "증감액": 10,
                "증감율": 11.1,
            }
        ]
    ).to_csv(
        region_dir / "2021_서울_세부사업_정제.csv",
        index=False,
        encoding="utf-8-sig",
    )
    changed = pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "10",
                "세부사업명": "사업",
                "주요내용": "원문",
                "수정전_주요내용_정제": "수정전",
                "수정후_주요내용_정제": "수정후",
                "변경유형": "오류 수정",
                "판단근거": "근거",
            }
        ]
    )

    result = attach_latest_budget(changed, data_root)

    assert result.loc[0, "당해예산"] == 100
    assert result.loc[0, "예산연결상태"] == "키 일치·예산 유지"


def test_attach_latest_budget_rejects_stale_cleaned_text(tmp_path: Path):
    data_root = tmp_path / "data"
    region_dir = data_root / "서울"
    region_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "10",
                "세부사업명": "사업",
                "주요내용_정제": "다른 값",
                "사업분류재정구분": "자체",
                "당해예산": 100,
                "전년도예산": 90,
                "증감액": 10,
                "증감율": 11.1,
            }
        ]
    ).to_csv(
        region_dir / "2021_서울_세부사업_정제.csv",
        index=False,
        encoding="utf-8-sig",
    )
    changed = pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": "서울",
                "원본행": "10",
                "세부사업명": "사업",
                "주요내용": "원문",
                "수정전_주요내용_정제": "수정전",
                "수정후_주요내용_정제": "수정후",
                "변경유형": "오류 수정",
                "판단근거": "근거",
            }
        ]
    )

    with pytest.raises(ValueError, match="수정후 정제문"):
        attach_latest_budget(changed, data_root)
