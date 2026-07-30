from pathlib import Path

import pandas as pd

from scripts.verify_2023_hallucination_recovery import verify_recovery


def test_verify_recovery_detects_broader_blank_input_change(
    tmp_path: Path,
    monkeypatch,
):
    regions = ("서울",)
    monkeypatch.setattr(
        "scripts.verify_2023_hallucination_recovery.REGIONS",
        regions,
    )
    data_root = tmp_path / "interim"
    region_root = data_root / "서울"
    region_root.mkdir(parents=True)

    wide = pd.DataFrame(
        [
            {
                "연도": 2023,
                "지역": "서울",
                "원본행": "1",
                "세부사업명": "빈 원문 사업",
                "주요내용": "",
                "주요내용_정제": "빈 원문 사업",
            }
        ]
    )
    long = pd.concat(
        [
            wide.assign(예산구분="당해예산"),
            wide.assign(예산구분="전년도예산"),
        ],
        ignore_index=True,
    )
    checkpoint = pd.DataFrame({"주요내용_정제": ["빈 원문 사업"]}, index=[0])
    wide.to_csv(
        region_root / "2023_서울_세부사업_정제.csv",
        index=False,
        encoding="utf-8-sig",
    )
    long.to_csv(
        region_root / "2023_서울_세부사업_정제_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checkpoint.to_csv(
        data_root / "2023_llm_정제_체크포인트.csv",
        encoding="utf-8-sig",
    )

    result = verify_recovery(data_root)

    assert result["checkpoint_wide_cleaned_multiset_match"] is True
    assert result["wide_long_cleaned_mismatch"] == 0
    assert result["blank_original_cleaned_nonblank_rows"] == 1
    assert result["broader_blank_input_recovery_complete"] is False
