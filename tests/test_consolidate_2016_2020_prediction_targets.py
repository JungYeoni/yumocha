"""2016-2020년 전용 TF-IDF 예측 대상 취합 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.consolidate_2016_2020_prediction_targets import PREDICTION_YEARS, save_outputs
from scripts.consolidate_tfidf_prediction_targets import (
    SOURCE_COLUMNS,
    consolidate_prediction_targets,
)


def _write_source(source_dir: Path, *, year: int, region: str, original_rows: list[int]) -> None:
    rows = []
    for original_row in original_rows:
        row = {column: None for column in SOURCE_COLUMNS}
        row.update(
            {
                "연도": year,
                "지역": region,
                "세부사업명": f"사업 {original_row}",
                "주요내용_정제": "지원내용",
                "원본행": original_row,
            }
        )
        rows.append(row)
    region_dir = source_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(region_dir / f"{year}_{region}_세부사업_정제.csv", index=False)


def test_prediction_years_excludes_2021_and_later():
    assert PREDICTION_YEARS == [2016, 2017, 2018, 2019, 2020]


def test_consolidate_and_save_restricted_to_2016_2020(tmp_path):
    for year in PREDICTION_YEARS:
        _write_source(tmp_path, year=year, region="서울", original_rows=[1])
    _write_source(tmp_path, year=2022, region="서울", original_rows=[1])

    combined, qa = consolidate_prediction_targets(
        tmp_path, years=PREDICTION_YEARS, regions=["서울"]
    )
    paths = save_outputs(combined, qa, tmp_path / "out")

    assert set(combined["연도"]) == set(PREDICTION_YEARS)
    assert 2022 not in combined["연도"].tolist()
    reloaded = pd.read_csv(paths["분류본"])
    assert set(reloaded["연도"]) == set(PREDICTION_YEARS)
