"""재정대응지수(z-score) 기반 세부영역별 TFR 회귀를 실행한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_subarea_fiscal_index_tfr_regression_sample import (
    LAG1_COLUMN,
    LAG2_COLUMN,
)
from scripts.run_subarea_fiscal_tfr_regression import run_subarea_models

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_재정대응지수_TFR_회귀표본.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"


def main() -> None:
    if not DEFAULT_SAMPLE.is_file():
        raise FileNotFoundError(f"지수 회귀표본이 없습니다: {DEFAULT_SAMPLE}")
    sample = pd.read_csv(DEFAULT_SAMPLE)
    variants = {
        "지수기반_기본모형(F_z,t-1)": run_subarea_models(
            sample, lag_column=LAG1_COLUMN, transform="identity"
        ),
        "지수기반_강건성체크_2년시차(F_z,t-2)": run_subarea_models(
            sample, lag_column=LAG2_COLUMN, transform="identity"
        ),
    }
    results = []
    for label, table in variants.items():
        table = table.copy()
        table.insert(0, "모형버전", label)
        results.append(table)
    combined = pd.concat(results, ignore_index=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_OUTPUT_DIR / "2016-2024_세부영역별_재정대응지수_TFR_고정효과_결과.csv"
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    print(combined[["모형버전", "모형", "계수", "p값", "관측치"]].round(4).to_string(index=False))
    print(f"저장: {output}")


if __name__ == "__main__":
    main()
