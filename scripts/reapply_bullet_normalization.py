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
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.bullet_normalization import (  # noqa: E402
    normalize_bullet_frame,
    normalized_filename,
    pair_wide_long_paths,
    read_text_csv,
    validate_wide_long_sync,
)

DEFAULT_YEARS = (2016, 2017, 2021, 2022, 2023, 2024)


def _write_pair_with_backup(
    paths: tuple[Path, ...],
    frames: dict[Path, pd.DataFrame],
    *,
    interim_dir: Path,
    backup_dir: Path,
) -> None:
    """wide·long 변경 파일을 함께 준비하고 교체 실패 시 원본으로 복원한다."""
    backup_paths: dict[Path, Path] = {}
    temporary_paths: dict[Path, Path] = {}
    replaced_paths: list[Path] = []
    try:
        for path in paths:
            backup_path = backup_dir / path.relative_to(interim_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            backup_paths[path] = backup_path

            temporary_path = path.with_name(f".{path.name}.{backup_dir.name}.tmp")
            frames[path].to_csv(temporary_path, index=False, encoding="utf-8-sig")
            temporary_paths[path] = temporary_path

        for path in paths:
            os.replace(temporary_paths[path], path)
            replaced_paths.append(path)
    except Exception as error:
        rollback_errors: list[str] = []
        for path in replaced_paths:
            try:
                shutil.copy2(backup_paths[path], path)
            except Exception as rollback_error:
                rollback_errors.append(f"{path}: {type(rollback_error).__name__}")
        if rollback_errors:
            raise RuntimeError(f"wide·long 롤백 실패: {rollback_errors}") from error
        raise
    finally:
        for temporary_path in temporary_paths.values():
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
        if normalized_filename(path).startswith(year_prefixes)
        and normalized_filename(path).endswith(valid_suffixes)
    )
    if not paths:
        raise FileNotFoundError(f"대상 정제 CSV가 없습니다: {interim_dir}")

    frames: dict[Path, pd.DataFrame] = {}
    changed_by_path: dict[Path, int] = {}
    for path in paths:
        frame, changed_cells = normalize_bullet_frame(read_text_csv(path), path)
        frames[path] = frame
        if changed_cells:
            changed_by_path[path] = changed_cells

    validate_wide_long_sync(frames)

    backup_dir: Path | None = None
    if write and changed_by_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = interim_dir / ".bullet_normalization_backups" / timestamp
        for wide_path, long_path in pair_wide_long_paths(paths):
            changed_pair = tuple(path for path in (wide_path, long_path) if path in changed_by_path)
            if changed_pair:
                _write_pair_with_backup(
                    changed_pair,
                    frames,
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
