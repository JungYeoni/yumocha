"""2021년 영역분류 라벨 취합 스크립트 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.consolidate_2021_area_labels import (
    BASE_COLUMNS,
    MAJOR_BY_SUBCATEGORY,
    consolidate_labels,
    normalize_text,
    read_label_file,
    refresh_from_sources,
    region_from_filename,
    validate_source_keys,
)


def test_normalize_text_handles_missing_unicode_and_whitespace():
    assert normalize_text(pd.NA) == ""
    assert normalize_text("  주요내용\n 정제  ") == "주요내용 정제"
    assert normalize_text("가") == "가"


def test_region_from_filename_finds_exact_region():
    path = Path("(완료) 라벨링_2021_서울_세부사업_정제.xlsx")
    assert region_from_filename(path) == "서울"


def test_region_from_filename_rejects_unknown_region():
    with pytest.raises(ValueError, match="지역을 하나로 판별"):
        region_from_filename(Path("라벨링_2021_전국.xlsx"))


def test_every_subcategory_maps_to_one_canonical_major_category():
    assert len(MAJOR_BY_SUBCATEGORY) == 12
    assert MAJOR_BY_SUBCATEGORY["2-1. 돌봄 여건"] == "2. 가족·생활"
    assert MAJOR_BY_SUBCATEGORY["4-1. 일·가정 양립 여건"] == "4. 사회·문화"
    assert MAJOR_BY_SUBCATEGORY["지표체계 외"] == "지표체계 외"


def test_read_label_file_stores_normalized_region(tmp_path):
    path = tmp_path / "(완료) 라벨링_2021_서울_세부사업_정제.xlsx"
    row = {column: None for column in BASE_COLUMNS}
    row.update(
        {
            "대영역": "2. 가족·생활",
            "세부영역": "2-1. 돌봄 여건",
            "연도": 2021,
            "지역": "  서울  ",
            "세부사업명": "돌봄 지원",
            "원본행": 1,
        }
    )
    pd.DataFrame([row]).to_excel(path, index=False)

    result = read_label_file(path)

    assert result["지역"].tolist() == ["서울"]


def _write_source_keys(
    source_dir: Path,
    *,
    region: str,
    original_rows: list[int],
) -> None:
    region_dir = source_dir / region
    region_dir.mkdir(parents=True)
    pd.DataFrame({"지역": [region] * len(original_rows), "원본행": original_rows}).to_csv(
        region_dir / f"2021_{region}_세부사업_정제.csv", index=False
    )


def _write_label_rows(
    input_dir: Path,
    *,
    region: str,
    original_rows: list[int],
) -> None:
    input_dir.mkdir(parents=True)
    rows = []
    for original_row in original_rows:
        row = {column: None for column in BASE_COLUMNS}
        row.update(
            {
                "대영역": "2. 가족·생활",
                "세부영역": "2-1. 돌봄 여건",
                "연도": 2021,
                "지역": region,
                "세부사업명": f"돌봄 지원 {original_row}",
                "원본행": original_row,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_excel(
        input_dir / f"(완료) 라벨링_2021_{region}_세부사업_정제.xlsx",
        index=False,
    )


def test_validate_source_keys_rejects_missing_label_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.consolidate_2021_area_labels.REGION_ORDER",
        ["서울"],
    )
    input_dir = tmp_path / "labels"
    source_dir = tmp_path / "source"
    _write_label_rows(input_dir, region="서울", original_rows=[1])
    _write_source_keys(source_dir, region="서울", original_rows=[1, 2])

    with pytest.raises(ValueError, match="라벨누락.*2"):
        consolidate_labels(input_dir, source_dir)


def test_validate_source_keys_rejects_unexpected_label_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.consolidate_2021_area_labels.REGION_ORDER",
        ["서울"],
    )
    input_dir = tmp_path / "labels"
    source_dir = tmp_path / "source"
    _write_label_rows(input_dir, region="서울", original_rows=[1, 2])
    _write_source_keys(source_dir, region="서울", original_rows=[1])

    with pytest.raises(ValueError, match="예상밖라벨.*2"):
        consolidate_labels(input_dir, source_dir)


def test_validate_source_keys_rejects_missing_key(tmp_path):
    combined = pd.DataFrame({"지역": ["서울"], "원본행": [pd.NA]})

    with pytest.raises(ValueError, match="지역·원본행 결측"):
        validate_source_keys(combined, tmp_path)


def test_refresh_from_sources_keeps_labels_and_uses_latest_cleaned_text(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "scripts.consolidate_2021_area_labels.REGION_ORDER",
        ["서울"],
    )
    source_dir = tmp_path / "source"
    source_path = source_dir / "서울" / "2021_서울_세부사업_정제.csv"
    source_path.parent.mkdir(parents=True)
    source_row = {column: None for column in BASE_COLUMNS if column not in {"대영역", "세부영역"}}
    source_row.update(
        {
            "연도": 2021,
            "지역": "서울",
            "원본행": 1,
            "세부사업명": "최신 사업명",
            "주요내용_정제": "최신 정제문",
        }
    )
    pd.DataFrame([source_row]).to_csv(source_path, index=False)
    label_row = {column: None for column in BASE_COLUMNS}
    label_row.update(
        {
            "대영역": "2. 가족·생활",
            "세부영역": "2-1. 돌봄 여건",
            "연도": 2021,
            "지역": "서울",
            "원본행": 1,
            "세부사업명": "옛 사업명",
            "주요내용_정제": "옛 정제문",
            "라벨원본파일": "서울.xlsx",
        }
    )

    result = refresh_from_sources(pd.DataFrame([label_row]), source_dir)

    assert result.loc[0, "대영역"] == "2. 가족·생활"
    assert result.loc[0, "세부영역"] == "2-1. 돌봄 여건"
    assert result.loc[0, "주요내용_정제"] == "최신 정제문"
