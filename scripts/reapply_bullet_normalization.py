"""과거 연도 wide·long CSV에 최신 불릿 정규화를 재적용한다.

기본 실행은 파일을 변경하지 않는 검증 모드다. 실제 반영은 ``--write``를
지정하며, 변경 전 파일은 ``data/interim/.bullet_normalization_backups`` 아래에
실행 시각별로 백업한다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.llm_refine import extract_quantities, numbers_preserved  # noqa: E402
from src.features.text_match import BULLET_PATTERN, dedup_label  # noqa: E402

DEFAULT_YEARS = (2016, 2017, 2021, 2022, 2023, 2024)
TEXT_COLUMNS = ("주요내용", "주요내용_정제")
SYNC_COLUMNS = ("원본행", "세부사업명", *TEXT_COLUMNS)


def _read_csv(path: Path) -> pd.DataFrame:
    # NA/NULL/None/빈 값 등을 결측치로 바꾸지 않고 원문 그대로 읽는다.
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def _normalized_name(path: Path) -> str:
    return unicodedata.normalize("NFC", path.name)


def _normalize_frame(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    changed_cells = 0

    for column in TEXT_COLUMNS:
        if column not in result:
            continue

        before = result[column]
        bullet_rows = before.astype("string").str.contains(BULLET_PATTERN, na=False)
        after = before.copy()
        after.loc[bullet_rows] = before.loc[bullet_rows].apply(dedup_label)
        changed = ~before.astype("string").fillna("<NA>").eq(after.astype("string").fillna("<NA>"))

        for old, new in zip(before.loc[changed], after.loc[changed], strict=True):
            if not numbers_preserved(old, new):
                raise ValueError(f"숫자 순서가 변경됐습니다: {path} / {column}")
            if extract_quantities(old) != extract_quantities(new):
                raise ValueError(f"수량 표현이 변경됐습니다: {path} / {column}")

        result[column] = after
        changed_cells += int(changed.sum())

        residual = (
            result[column]
            .astype("string")
            .str.contains(
                BULLET_PATTERN,
                na=False,
            )
        )
        if residual.any():
            raise ValueError(f"불릿이 남아 있습니다: {path} / {column} / {int(residual.sum())}건")

    if len(result) != len(df) or list(result.columns) != list(df.columns):
        raise ValueError(f"행 수 또는 스키마가 변경됐습니다: {path}")

    return result, changed_cells


def _row_counter(df: pd.DataFrame, columns: Sequence[str]) -> Counter[tuple[str, ...]]:
    normalized = df[columns].astype("string").fillna("<NA>")
    return Counter(map(tuple, normalized.itertuples(index=False, name=None)))


def _validate_wide_long_sync(frames: dict[Path, pd.DataFrame]) -> None:
    paths_by_name = {(path.parent, _normalized_name(path)): path for path in frames}

    for long_path in frames:
        normalized_name = _normalized_name(long_path)
        if not normalized_name.endswith("_long.csv"):
            continue
        wide_name = normalized_name.removesuffix("_long.csv") + ".csv"
        if (long_path.parent, wide_name) not in paths_by_name:
            raise FileNotFoundError(f"long에 대응하는 wide 파일이 없습니다: {long_path}")

    for wide_path, wide in frames.items():
        normalized_name = _normalized_name(wide_path)
        if normalized_name.endswith("_long.csv"):
            continue

        long_name = normalized_name.removesuffix(".csv") + "_long.csv"
        long_path = paths_by_name.get((wide_path.parent, long_name))
        if long_path is None:
            raise FileNotFoundError(f"wide에 대응하는 long 파일이 없습니다: {wide_path}")

        long = frames[long_path]
        if len(long) != len(wide) * 2:
            raise ValueError(
                f"wide·long 행 수가 맞지 않습니다: {wide_path} ({len(wide)} / {len(long)})"
            )

        missing_wide = [column for column in SYNC_COLUMNS if column not in wide]
        missing_long = [column for column in SYNC_COLUMNS if column not in long]
        if missing_wide or missing_long:
            raise ValueError(
                "wide·long 동기화 검증에 필요한 컬럼이 없습니다: "
                f"{wide_path} (wide 누락: {missing_wide}, long 누락: {missing_long})"
            )

        columns = list(SYNC_COLUMNS)
        wide_rows = _row_counter(wide, columns)
        long_rows = _row_counter(long, columns)
        expected_long_rows = Counter({row: count * 2 for row, count in wide_rows.items()})
        if long_rows != expected_long_rows:
            raise ValueError(f"wide·long 텍스트가 동기화되지 않았습니다: {wide_path}")


def _write_with_backup(
    path: Path,
    df: pd.DataFrame,
    *,
    interim_dir: Path,
    backup_dir: Path,
) -> None:
    backup_path = backup_dir / path.relative_to(interim_dir)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def reapply_bullet_normalization(
    interim_dir: Path,
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
    write: bool = False,
) -> dict[str, object]:
    """대상 CSV를 정규화·검증하고 선택적으로 백업 후 덮어쓴다."""
    year_prefixes = tuple(f"{year}_" for year in years)
    valid_suffixes = ("_세부사업_정제.csv", "_세부사업_정제_long.csv")
    paths = sorted(
        path
        for path in interim_dir.glob("*/*.csv")
        if _normalized_name(path).startswith(year_prefixes)
        and _normalized_name(path).endswith(valid_suffixes)
    )
    if not paths:
        raise FileNotFoundError(f"대상 정제 CSV가 없습니다: {interim_dir}")

    frames: dict[Path, pd.DataFrame] = {}
    changed_by_path: dict[Path, int] = {}
    for path in paths:
        frame, changed_cells = _normalize_frame(_read_csv(path), path)
        frames[path] = frame
        if changed_cells:
            changed_by_path[path] = changed_cells

    _validate_wide_long_sync(frames)

    backup_dir: Path | None = None
    if write and changed_by_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = interim_dir / ".bullet_normalization_backups" / timestamp
        for path in changed_by_path:
            _write_with_backup(
                path,
                frames[path],
                interim_dir=interim_dir,
                backup_dir=backup_dir,
            )

    return {
        "검사파일": len(paths),
        "변경파일": len(changed_by_path),
        "변경셀": sum(changed_by_path.values()),
        "쓰기모드": write,
        "백업경로": str(backup_dir) if backup_dir else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="검증 후 변경 파일을 백업하고 실제 CSV에 반영합니다.",
    )
    args = parser.parse_args()

    summary = reapply_bullet_normalization(
        args.interim_dir.resolve(),
        years=tuple(args.years),
        write=args.write,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
