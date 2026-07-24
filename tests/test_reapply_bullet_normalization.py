from pathlib import Path
import unicodedata

import pandas as pd
import pytest

import scripts.reapply_bullet_normalization as bullet_script
from scripts.reapply_bullet_normalization import reapply_bullet_normalization


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_reapply_bullet_normalization_dry_run_and_write_with_backup(tmp_path):
    region_dir = tmp_path / "서울"
    wide_path = region_dir / unicodedata.normalize("NFD", "2022_서울_세부사업_정제.csv")
    long_path = region_dir / "2022_서울_세부사업_정제_long.csv"
    row = {
        "원본행": 10,
        "세부사업명": "사업 A",
        "주요내용": "지원대상 : 시민 • 지원내용 : 상담 3회",
        "주요내용_정제": "시민 상담 • 3회",
    }
    _write_csv(wide_path, [row])
    _write_csv(long_path, [row, row])

    dry_run = reapply_bullet_normalization(tmp_path, years=(2022,))

    assert dry_run["검사파일"] == 2
    assert dry_run["변경파일"] == 2
    assert dry_run["변경셀"] == 6
    assert "•" in pd.read_csv(wide_path, encoding="utf-8-sig").loc[0, "주요내용"]

    written = reapply_bullet_normalization(tmp_path, years=(2022,), write=True)

    assert written["변경파일"] == 2
    assert written["백업경로"] is not None
    normalized = pd.read_csv(wide_path, encoding="utf-8-sig")
    assert normalized.loc[0, "주요내용"] == "지원대상 : 시민 지원내용 : 상담 3회"
    assert normalized.loc[0, "주요내용_정제"] == "시민 상담 3회"

    backup_dir = Path(str(written["백업경로"]))
    backup = pd.read_csv(backup_dir / "서울" / wide_path.name, encoding="utf-8-sig")
    assert "•" in backup.loc[0, "주요내용"]


def test_reapply_bullet_normalization_preserves_na_like_text(tmp_path):
    region_dir = tmp_path / "서울"
    wide_path = region_dir / "2022_서울_세부사업_정제.csv"
    long_path = region_dir / "2022_서울_세부사업_정제_long.csv"
    row = {
        "원본행": 1,
        "세부사업명": "NA",
        "주요내용": "지원대상 : 시민 • 지원내용 : 상담 1회",
        "주요내용_정제": "NULL",
    }
    _write_csv(wide_path, [row])
    _write_csv(long_path, [row, row])

    reapply_bullet_normalization(tmp_path, years=(2022,), write=True)

    normalized = pd.read_csv(
        wide_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False
    )
    normalized_long = pd.read_csv(
        long_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False
    )
    assert normalized.loc[0, "세부사업명"] == "NA"
    assert normalized.loc[0, "주요내용_정제"] == "NULL"
    assert normalized_long["세부사업명"].tolist() == ["NA", "NA"]
    assert normalized_long["주요내용_정제"].tolist() == ["NULL", "NULL"]


def test_reapply_bullet_normalization_fails_when_sync_column_missing(tmp_path):
    region_dir = tmp_path / "서울"
    wide_path = region_dir / "2022_서울_세부사업_정제.csv"
    long_path = region_dir / "2022_서울_세부사업_정제_long.csv"
    wide_row = {
        "세부사업명": "사업 A",
        "주요내용": "지원대상 : 시민 • 지원내용 : 상담 3회",
        "주요내용_정제": "시민 상담 • 3회",
    }
    long_row = {**wide_row, "주요내용": "지원대상 : 다른 시민 • 지원내용 : 상담 5회"}
    _write_csv(wide_path, [wide_row])
    _write_csv(long_path, [long_row, long_row])

    with pytest.raises(ValueError, match="동기화 검증에 필요한 컬럼이 없습니다"):
        reapply_bullet_normalization(tmp_path, years=(2022,))


def test_reapply_bullet_normalization_rolls_back_wide_when_long_replace_fails(
    tmp_path,
    monkeypatch,
):
    region_dir = tmp_path / "서울"
    wide_path = region_dir / "2022_서울_세부사업_정제.csv"
    long_path = region_dir / "2022_서울_세부사업_정제_long.csv"
    row = {
        "원본행": 10,
        "세부사업명": "사업 A",
        "주요내용": "시민 • 상담 3회",
        "주요내용_정제": "시민 • 상담 3회",
    }
    _write_csv(wide_path, [row])
    _write_csv(long_path, [row, row])

    original_replace = bullet_script.os.replace
    replace_count = 0

    def fail_second_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("long 교체 실패")
        original_replace(source, destination)

    monkeypatch.setattr(bullet_script.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="long 교체 실패"):
        reapply_bullet_normalization(tmp_path, years=(2022,), write=True)

    restored_wide = pd.read_csv(
        wide_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False
    )
    untouched_long = pd.read_csv(
        long_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False
    )
    assert "•" in restored_wide.loc[0, "주요내용"]
    assert "•" in untouched_long.loc[0, "주요내용"]
