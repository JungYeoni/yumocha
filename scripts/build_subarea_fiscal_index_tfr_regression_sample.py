"""재정대응지수 기반 #98 TFR 회귀표본을 만든다.

재정대응지수(z-score)를 전년도·전전년도 시차로 만들어 TFR과 결합한다.
지수는 이미 표준화된 값이므로 회귀에서 추가 log1p 변환을 하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_재정대응지수.csv"
DEFAULT_SAMPLE = REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_재정반응성_회귀표본.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"
INDEX_COLUMN = "세부영역_재정대응지수_z"
LAG1_COLUMN = f"{INDEX_COLUMN}_전년도"
LAG2_COLUMN = f"{INDEX_COLUMN}_전전년도"


def build_index_lags(index_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"지역", "연도", "세부영역", INDEX_COLUMN}
    missing = required - set(index_panel.columns)
    if missing:
        raise KeyError(f"재정대응지수 필수 컬럼 누락: {sorted(missing)}")
    panel = index_panel[index_panel["세부영역"].ne("지표체계 외")].copy()
    if panel.duplicated(["지역", "연도", "세부영역"]).any():
        raise ValueError("재정대응지수 지역·연도·세부영역 키 중복")
    panel = panel.sort_values(["지역", "세부영역", "연도"])
    years = panel.groupby(["지역", "세부영역"])["연도"].agg(list)
    expected = list(range(int(panel["연도"].min()), int(panel["연도"].max()) + 1))
    if years.map(lambda values: values != expected).any():
        raise ValueError("재정대응지수 패널에 지역·세부영역별 연도 공백이 있습니다.")
    grouped = panel.groupby(["지역", "세부영역"], sort=False)[INDEX_COLUMN]
    panel[LAG1_COLUMN] = grouped.shift(1)
    panel[LAG2_COLUMN] = grouped.shift(2)
    return panel[["지역", "연도", "세부영역", LAG1_COLUMN, LAG2_COLUMN]]


def build_sample(index_panel: pd.DataFrame, fiscal_sample: pd.DataFrame) -> pd.DataFrame:
    lags = build_index_lags(index_panel)
    required = {"지역", "연도", "세부영역", "합계출산율"}
    missing = required - set(fiscal_sample.columns)
    if missing:
        raise KeyError(f"기존 회귀표본 필수 컬럼 누락: {sorted(missing)}")
    outcome = fiscal_sample[["지역", "연도", "세부영역", "합계출산율"]].copy()
    result = outcome.merge(lags, on=["지역", "연도", "세부영역"], how="left", validate="one_to_one")
    if len(result) != len(outcome) or result[[LAG1_COLUMN, LAG2_COLUMN]].isna().all(
        axis=1
    ).sum() == len(result):
        raise ValueError("재정대응지수 시차변수 결합에 실패했습니다.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-panel", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--fiscal-sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.index_panel.is_file() or not args.fiscal_sample.is_file():
        raise FileNotFoundError("재정대응지수 패널 또는 기존 회귀표본이 없습니다.")
    result = build_sample(pd.read_csv(args.index_panel), pd.read_csv(args.fiscal_sample))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "2016-2024_세부영역별_재정대응지수_TFR_회귀표본.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"저장: {output} ({len(result)}행, 1년시차 완비 {result[LAG1_COLUMN].notna().sum()}행)")


if __name__ == "__main__":
    main()
