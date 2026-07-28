"""TF-IDF 예측 대상 데이터 취합 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.consolidate_tfidf_prediction_targets import (
    PREDICTION_COLUMNS,
    SOURCE_COLUMNS,
    consolidate_prediction_targets,
    read_prediction_source,
)


def _write_source(
    source_dir: Path,
    *,
    year: int,
    region: str,
    original_rows: list[int],
    missing_content_rows: set[int] | None = None,
) -> Path:
    missing_content_rows = missing_content_rows or set()
    rows = []
    for original_row in original_rows:
        row = {column: None for column in SOURCE_COLUMNS}
        row.update(
            {
                "연도": year,
                "지역": region,
                "대분류": "Ⅰ. 공통사업",
                "중분류": "1. 사업",
                "세부사업명": f"사업 {original_row}",
                "주요내용_정제": (None if original_row in missing_content_rows else "지원내용"),
                "원본행": original_row,
            }
        )
        rows.append(row)
    region_dir = source_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)
    path = region_dir / f"{year}_{region}_세부사업_정제.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_read_prediction_source_builds_text_and_preserves_missing_content(tmp_path):
    path = _write_source(
        tmp_path,
        year=2020,
        region="서울",
        original_rows=[1, 2],
        missing_content_rows={2},
    )

    result = read_prediction_source(path, year=2020, region="서울")

    assert result["분류텍스트"].tolist() == ["사업 1 지원내용", "사업 2"]
    assert result["주요내용_정제_결측"].tolist() == [False, True]


def test_consolidate_prediction_targets_keeps_complete_key_and_schema(tmp_path):
    for year in [2020, 2022]:
        for region in ["서울", "부산"]:
            _write_source(
                tmp_path,
                year=year,
                region=region,
                original_rows=[1, 2],
            )

    combined, qa = consolidate_prediction_targets(
        tmp_path,
        years=[2020, 2022],
        regions=["서울", "부산"],
    )

    assert len(combined) == 8
    assert len(qa) == 4
    assert not combined.duplicated(["연도", "지역", "원본행"]).any()
    assert set(PREDICTION_COLUMNS).issubset(combined.columns)
    assert combined[["연도", "지역"]].drop_duplicates().shape[0] == 4


def test_consolidate_prediction_targets_rejects_2021(tmp_path):
    with pytest.raises(ValueError, match="2021년은 예측 대상"):
        consolidate_prediction_targets(tmp_path, years=[2021], regions=["서울"])


def test_consolidate_prediction_targets_rejects_missing_file(tmp_path):
    _write_source(tmp_path, year=2020, region="서울", original_rows=[1])

    with pytest.raises(FileNotFoundError, match="부산"):
        consolidate_prediction_targets(
            tmp_path,
            years=[2020],
            regions=["서울", "부산"],
        )


def test_consolidate_prediction_targets_rejects_duplicate_key(tmp_path):
    _write_source(tmp_path, year=2020, region="서울", original_rows=[1, 1])

    with pytest.raises(ValueError, match="파일 내부 키 중복"):
        consolidate_prediction_targets(
            tmp_path,
            years=[2020],
            regions=["서울"],
        )


def test_read_prediction_source_rejects_wrong_region(tmp_path):
    path = _write_source(tmp_path, year=2020, region="서울", original_rows=[1])
    frame = pd.read_csv(path)
    frame["지역"] = "부산"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="경로 지역과 데이터 지역"):
        read_prediction_source(path, year=2020, region="서울")
