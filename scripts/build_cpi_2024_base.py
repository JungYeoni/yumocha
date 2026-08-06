"""연도별 소비자물가지수(2020=100)를 이슈 #81이 요구하는 2024=100 기준으로 재기준화한다.

`data/lookup/연도별_소비자물가지수.csv`(통계청 소비자물가조사)는 2020=100
기준이지만, 이슈 #81은 "실질예산 기준: 2024년 가격"을 요구한다.
`src/provisional/adjust.py`의 `read_cpi()`는 `unit` 인자와 기준연도 지수값이
정확히 100이어야 통과하므로, 원본을 그대로 `--cpi-base-year 2024`로 넘기면
검증에서 걸린다. 지수 재기준화(연쇄가중 없는 단순 비율 조정)는 표준적인
절차이므로 새 값을 만들지 않고 원본을 그대로 비율만 바꾼다.

산식: `재기준화_지수_연도 = 원본_지수_연도 / 원본_지수_2024 × 100`

이 변환은 연도 간 전년대비 상승률(%)을 바꾸지 않는다(모든 연도에 같은
상수를 곱하는 것과 수학적으로 같음).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "lookup" / "연도별_소비자물가지수.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "lookup" / "연도별_소비자물가지수_2024기준.csv"
TARGET_BASE_YEAR = 2024
SOURCE_BASE_YEAR = 2020


def rebase_cpi(source: pd.DataFrame, *, target_base_year: int) -> pd.DataFrame:
    required = {"연도", "소비자물가지수", "기준연도"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"CPI 원본 필수 열 누락: {sorted(missing)}")

    source_units = set(source["기준연도"].astype(str))
    expected_source_unit = f"{SOURCE_BASE_YEAR}=100"
    if source_units != {expected_source_unit}:
        raise ValueError(f"CPI 원본 기준연도 표기가 예상과 다릅니다: {source_units}")

    if target_base_year not in set(source["연도"]):
        raise ValueError(f"CPI 원본에 목표 기준연도 {target_base_year}가 없습니다.")

    frame = source.copy()
    base_index = float(frame.loc[frame["연도"].eq(target_base_year), "소비자물가지수"].iloc[0])
    frame["소비자물가지수"] = (frame["소비자물가지수"] / base_index * 100).round(6)
    frame["기준연도"] = f"{target_base_year}=100"

    rebased_at_target = float(
        frame.loc[frame["연도"].eq(target_base_year), "소비자물가지수"].iloc[0]
    )
    if rebased_at_target != 100.0:
        raise ValueError(f"재기준화 후 목표 연도 지수가 100이 아닙니다: {rebased_at_target}")
    if frame["소비자물가지수"].le(0).any():
        raise ValueError("재기준화 후 0 이하 지수가 있습니다.")

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-base-year", type=int, default=TARGET_BASE_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input, encoding="utf-8-sig")
    rebased = rebase_cpi(source, target_base_year=args.target_base_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rebased.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"재기준화 CPI 저장: {args.output} ({len(rebased)}행, 기준연도={args.target_base_year})")


if __name__ == "__main__":
    main()
