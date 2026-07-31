from pathlib import Path

import pandas as pd

from scripts.apply_2021_semantic_corrections import (
    CORRECTIONS,
    apply_corrections,
)


def test_apply_corrections_updates_checkpoint_wide_and_long(tmp_path: Path, monkeypatch):
    correction = CORRECTIONS[0]
    monkeypatch.setattr(
        "scripts.apply_2021_semantic_corrections.CORRECTIONS",
        (correction,),
    )
    data_root = tmp_path / "interim"
    region_dir = data_root / correction.region
    region_dir.mkdir(parents=True)
    row = {
        "연도": 2021,
        "지역": correction.region,
        "원본행": correction.source_row,
        "세부사업명": correction.name,
        "주요내용": correction.expected_original,
        "주요내용_정제": correction.expected_cleaned,
    }
    wide = pd.DataFrame([row])
    long = pd.DataFrame([row, row])
    checkpoint = pd.DataFrame(
        {"주요내용_정제": [f"• {correction.expected_cleaned}"]},
        index=[10],
    )
    wide_path = region_dir / f"2021_{correction.region}_세부사업_정제.csv"
    long_path = region_dir / f"2021_{correction.region}_세부사업_정제_long.csv"
    checkpoint_path = data_root / "2021_llm_정제_체크포인트.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    checkpoint.to_csv(checkpoint_path, encoding="utf-8-sig")

    result = apply_corrections(data_root)

    assert len(result) == 1
    updated_wide = pd.read_csv(wide_path, encoding="utf-8-sig")
    updated_long = pd.read_csv(long_path, encoding="utf-8-sig")
    updated_checkpoint = pd.read_csv(checkpoint_path, encoding="utf-8-sig", index_col=0)
    assert updated_wide["주요내용_정제"].tolist() == [correction.corrected_cleaned]
    assert updated_long["주요내용_정제"].tolist() == [
        correction.corrected_cleaned,
        correction.corrected_cleaned,
    ]
    assert updated_checkpoint["주요내용_정제"].tolist() == [correction.corrected_cleaned]

    repeated = apply_corrections(data_root)

    assert len(repeated) == 1
