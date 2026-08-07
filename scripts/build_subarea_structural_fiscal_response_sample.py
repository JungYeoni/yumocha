"""#102 구조환경 t → 재정대응예산 t+1/t+2 세부영역 패널을 만든다.

2026-08-07 팀 합의:
- X: t년 세부영역별 구조환경지수(이동평균 없음)
- Y: t+2년 세부영역별 실질 1인당 재정대응예산
- 재정대응지수(z-score)가 아닌 예산액 자체를 사용한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISCAL = REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_인구1인당_실질예산액.csv"
DEFAULT_STRUCTURAL = (
    REPO_ROOT / "data/processed/structural_index/structural_index_pooled_subcategory_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"

LAG_YEARS = 2
STRUCTURAL_PREDICTOR = "구조환경지수_t"
FISCAL_OUTCOME = "인구1인당_실질예산_t+2_원"
LOG_FISCAL_OUTCOME = "log1p_인구1인당_실질예산_t+2"


def fiscal_outcome_columns(lag_years: int) -> tuple[str, str]:
    """시차에 맞는 예산 수준·로그 컬럼명을 반환한다."""
    return (
        f"인구1인당_실질예산_t+{lag_years}_원",
        f"log1p_인구1인당_실질예산_t+{lag_years}",
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{label} 필수 컬럼 누락: {missing}")


def build_structural_fiscal_response_sample(
    fiscal: pd.DataFrame,
    structural_scores: pd.DataFrame,
    *,
    lag_years: int = LAG_YEARS,
) -> pd.DataFrame:
    """t년 구조환경과 t+lag년 실질 1인당 예산을 일대일로 정렬한다."""
    if lag_years <= 0:
        raise ValueError("예산 시차는 1년 이상이어야 합니다.")

    _require_columns(
        fiscal,
        {"지역", "연도", "세부영역", "인구1인당_실질예산_원"},
        label="재정대응예산",
    )
    _require_columns(
        structural_scores,
        {"region", "year", "subcategory", "subcategory_score"},
        label="구조환경지수",
    )

    expected_structural = set(FISCAL_TO_STRUCTURAL_SUBCATEGORY.values())
    observed_structural = set(structural_scores["subcategory"].dropna().unique())
    if observed_structural != expected_structural:
        raise ValueError(
            "구조환경지수 subcategory 표기가 매핑표와 다릅니다: "
            f"구조환경에만={sorted(observed_structural - expected_structural)}, "
            f"매핑표에만={sorted(expected_structural - observed_structural)}"
        )

    fiscal_outcome, log_fiscal_outcome = fiscal_outcome_columns(lag_years)
    structural_to_fiscal = {
        structural: fiscal_name
        for fiscal_name, structural in FISCAL_TO_STRUCTURAL_SUBCATEGORY.items()
    }
    structural = structural_scores.rename(
        columns={
            "region": "지역",
            "year": "구조환경연도",
            "subcategory_score": STRUCTURAL_PREDICTOR,
        }
    ).copy()
    structural["세부영역"] = structural["subcategory"].map(structural_to_fiscal)
    structural["예산연도"] = structural["구조환경연도"] + lag_years
    structural = structural[["지역", "구조환경연도", "예산연도", "세부영역", STRUCTURAL_PREDICTOR]]

    fiscal_panel = fiscal.loc[fiscal["세부영역"].ne("지표체계 외")].copy()
    fiscal_panel = fiscal_panel.rename(
        columns={"연도": "예산연도", "인구1인당_실질예산_원": fiscal_outcome}
    )[["지역", "예산연도", "세부영역", fiscal_outcome]]

    for frame, label, keys in (
        (structural, "구조환경", ["지역", "구조환경연도", "세부영역"]),
        (fiscal_panel, "재정대응예산", ["지역", "예산연도", "세부영역"]),
    ):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} 패널 키가 중복됩니다: {keys}")

    budget_years = set(pd.to_numeric(fiscal_panel["예산연도"], errors="raise").astype(int))
    structural = structural.loc[structural["예산연도"].isin(budget_years)].copy()
    result = structural.merge(
        fiscal_panel,
        on=["지역", "예산연도", "세부영역"],
        how="left",
        validate="one_to_one",
    )

    numeric_columns = [STRUCTURAL_PREDICTOR, fiscal_outcome]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric_columns].isna().any().any():
        missing_rows = result.loc[result[numeric_columns].isna().any(axis=1)].head(10)
        raise ValueError(
            f"구조환경·예산 결합 후 필수값이 누락됩니다: {missing_rows.to_dict(orient='records')}"
        )
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("구조환경·예산 표본에 무한값이 있습니다.")
    if result[fiscal_outcome].lt(0).any():
        raise ValueError("실질 1인당 예산에 음수가 있어 log1p 변환할 수 없습니다.")

    result[log_fiscal_outcome] = np.log1p(result[fiscal_outcome])
    if result.duplicated(["지역", "구조환경연도", "예산연도", "세부영역"]).any():
        raise ValueError("최종 재정반응성 패널 키가 중복됩니다.")

    return result.sort_values(["세부영역", "지역", "구조환경연도"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiscal", type=Path, default=DEFAULT_FISCAL)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag-years", type=int, choices=(1, 2), default=LAG_YEARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fiscal.is_file() or not args.structural.is_file():
        raise FileNotFoundError("실질 1인당 예산 또는 구조환경지수 패널이 없습니다.")

    result = build_structural_fiscal_response_sample(
        pd.read_csv(args.fiscal), pd.read_csv(args.structural), lag_years=args.lag_years
    )
    expected_years = 8 if args.lag_years == 1 else 7
    expected_rows = 17 * expected_years * 11
    if len(result) != expected_rows:
        raise ValueError(f"결과 행 수 불일치: 기대={expected_rows}, 실제={len(result)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.lag_years == 2 else "_t+1"
    output = args.output_dir / f"2016-2024_세부영역별_구조환경_재정대응_반응성_표본{suffix}.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"저장: {output} ({len(result)}행, "
        f"구조환경 {result['구조환경연도'].min()}~{result['구조환경연도'].max()}, "
        f"예산 {result['예산연도'].min()}~{result['예산연도'].max()})"
    )


if __name__ == "__main__":
    main()
