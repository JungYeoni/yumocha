"""과거 정제 CSV의 불릿 정규화와 wide·long 동기화 검증."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.features.llm_refine import extract_quantities, numbers_preserved
from src.features.text_match import BULLET_PATTERN, dedup_label

TEXT_COLUMNS = ("주요내용", "주요내용_정제")
SYNC_COLUMNS = ("원본행", "세부사업명", *TEXT_COLUMNS)


def read_text_csv(path: Path) -> pd.DataFrame:
    """NA 유사 문자열과 빈 값을 변환하지 않고 CSV를 읽는다."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def normalized_filename(path: Path) -> str:
    """macOS 분해형 한글 파일명도 비교할 수 있도록 NFC 이름을 반환한다."""
    return unicodedata.normalize("NFC", path.name)


def normalize_bullet_frame(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    """불릿이 검출된 텍스트 셀만 정규화하고 숫자·수량 보존을 검증한다."""
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

        residual = result[column].astype("string").str.contains(BULLET_PATTERN, na=False)
        if residual.any():
            raise ValueError(f"불릿이 남아 있습니다: {path} / {column} / {int(residual.sum())}건")

    if len(result) != len(df) or list(result.columns) != list(df.columns):
        raise ValueError(f"행 수 또는 스키마가 변경됐습니다: {path}")

    return result, changed_cells


def _row_counter(df: pd.DataFrame, columns: Sequence[str]) -> Counter[tuple[str, ...]]:
    normalized = df[list(columns)].astype("string").fillna("<NA>")
    return Counter(map(tuple, normalized.itertuples(index=False, name=None)))


def pair_wide_long_paths(paths: Sequence[Path]) -> list[tuple[Path, Path]]:
    """NFC 파일명을 기준으로 모든 wide·long 경로 쌍을 반환한다."""
    paths_by_name = {(path.parent, normalized_filename(path)): path for path in paths}
    pairs: list[tuple[Path, Path]] = []

    for path in paths:
        name = normalized_filename(path)
        if name.endswith("_long.csv"):
            wide_name = name.removesuffix("_long.csv") + ".csv"
            if (path.parent, wide_name) not in paths_by_name:
                raise FileNotFoundError(f"long에 대응하는 wide 파일이 없습니다: {path}")
            continue

        long_name = name.removesuffix(".csv") + "_long.csv"
        long_path = paths_by_name.get((path.parent, long_name))
        if long_path is None:
            raise FileNotFoundError(f"wide에 대응하는 long 파일이 없습니다: {path}")
        pairs.append((path, long_path))

    return pairs


def validate_wide_long_sync(frames: dict[Path, pd.DataFrame]) -> None:
    """wide·long의 필수 텍스트 컬럼, 행 수와 값 구성이 같은지 검증한다."""
    for wide_path, long_path in pair_wide_long_paths(list(frames)):
        wide = frames[wide_path]
        long = frames[long_path]

        missing_wide = [column for column in SYNC_COLUMNS if column not in wide]
        missing_long = [column for column in SYNC_COLUMNS if column not in long]
        if missing_wide or missing_long:
            raise ValueError(
                "wide·long 동기화 검증에 필요한 컬럼이 없습니다: "
                f"{wide_path} (wide 누락: {missing_wide}, long 누락: {missing_long})"
            )

        if len(long) != len(wide) * 2:
            raise ValueError(
                f"wide·long 행 수가 맞지 않습니다: {wide_path} ({len(wide)} / {len(long)})"
            )

        wide_rows = _row_counter(wide, SYNC_COLUMNS)
        long_rows = _row_counter(long, SYNC_COLUMNS)
        expected_long_rows = Counter({row: count * 2 for row, count in wide_rows.items()})
        if long_rows != expected_long_rows:
            raise ValueError(f"wide·long 텍스트가 동기화되지 않았습니다: {wide_path}")


__all__ = [
    "normalize_bullet_frame",
    "normalized_filename",
    "pair_wide_long_paths",
    "read_text_csv",
    "validate_wide_long_sync",
]
