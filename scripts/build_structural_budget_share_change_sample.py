"""#108 구조환경 변화와 후행 예산비중 변화의 대응성 표본을 만든다.

시간 배열은 구조환경 ``t→t+1``과 같은 영역의 계획예산 비중
``t+2→t+3``을 연결한다. 예산비중은 지역×연도 전체 저출생 대응
계획예산(지표체계 외 포함)에서 해당 영역이 차지하는 비율이다.
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
DEFAULT_STRUCTURAL = (
    REPO_ROOT / "data/processed/structural_index/structural_index_pooled_subcategory_scores.csv"
)
DEFAULT_BUDGET_SHARE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_지역연도_세부영역별_계획예산비중.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data/processed/analysis/2016-2024_구조환경변화_후행예산비중변화_표본.csv"
)

STRUCTURAL_CHANGE = "구조환경지수_변화_t_t1"
BUDGET_SHARE_CHANGE = "계획예산비중_변화_t2_t3_pp"
BASE_YEARS = set(range(2016, 2022))


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := required - set(frame.columns):
        raise KeyError(f"{label} 필수 컬럼 누락: {sorted(missing)}")


def build_change_response_sample(
    structural_scores: pd.DataFrame, budget_shares: pd.DataFrame
) -> pd.DataFrame:
    """구조환경 1년 변화와 2년 뒤 시작하는 예산비중 1년 변화를 정렬한다."""
    _require_columns(
        structural_scores,
        {"region", "year", "subcategory", "subcategory_score"},
        "구조환경지수",
    )
    _require_columns(
        budget_shares,
        {"지역", "연도", "세부영역", "계획예산비중_pct", "예산누락주의"},
        "예산비중",
    )
    structural_to_fiscal = {
        structural: fiscal for fiscal, structural in FISCAL_TO_STRUCTURAL_SUBCATEGORY.items()
    }
    observed = set(structural_scores["subcategory"].dropna().unique())
    expected = set(structural_to_fiscal)
    if observed != expected:
        raise ValueError(
            "구조환경 세부영역이 매핑표와 다릅니다: "
            f"구조환경에만={sorted(observed - expected)}, 매핑표에만={sorted(expected - observed)}"
        )

    structural = structural_scores.rename(columns={"region": "지역", "year": "기준연도"}).copy()
    structural["세부영역"] = structural["subcategory"].map(structural_to_fiscal)
    structural = structural[["지역", "기준연도", "세부영역", "subcategory_score"]]
    if structural.duplicated(["지역", "기준연도", "세부영역"]).any():
        raise ValueError("구조환경 지역×연도×세부영역 키가 중복됩니다.")
    structural = structural.sort_values(["지역", "세부영역", "기준연도"])
    grouped_structural = structural.groupby(["지역", "세부영역"], sort=False)
    structural["구조환경지수_t1"] = grouped_structural["subcategory_score"].shift(-1)
    structural["구조환경_t1연도"] = grouped_structural["기준연도"].shift(-1)
    structural = structural.loc[
        structural["구조환경_t1연도"].eq(structural["기준연도"] + 1)
        & structural["기준연도"].isin(BASE_YEARS)
    ].copy()
    structural = structural.rename(columns={"subcategory_score": "구조환경지수_t"})
    structural[STRUCTURAL_CHANGE] = structural["구조환경지수_t1"] - structural["구조환경지수_t"]

    budget = budget_shares.loc[budget_shares["세부영역"].ne("지표체계 외")].copy()
    budget = budget[["지역", "연도", "세부영역", "계획예산비중_pct", "예산누락주의"]]
    if budget.duplicated(["지역", "연도", "세부영역"]).any():
        raise ValueError("예산비중 지역×연도×세부영역 키가 중복됩니다.")
    budget = budget.sort_values(["지역", "세부영역", "연도"])
    grouped_budget = budget.groupby(["지역", "세부영역"], sort=False)
    budget["계획예산비중_t3_pct"] = grouped_budget["계획예산비중_pct"].shift(-1)
    budget["예산누락주의_t3"] = grouped_budget["예산누락주의"].shift(-1)
    budget["예산_t3연도"] = grouped_budget["연도"].shift(-1)
    budget = budget.loc[budget["예산_t3연도"].eq(budget["연도"] + 1)].copy()
    budget["기준연도"] = budget["연도"] - 2
    budget = budget.rename(
        columns={"연도": "예산_t2연도", "계획예산비중_pct": "계획예산비중_t2_pct"}
    )
    budget[BUDGET_SHARE_CHANGE] = budget["계획예산비중_t3_pct"] - budget["계획예산비중_t2_pct"]
    budget["예산누락주의_두연도"] = budget["예산누락주의"] | budget["예산누락주의_t3"]

    result = structural.merge(
        budget[
            [
                "지역",
                "기준연도",
                "세부영역",
                "예산_t2연도",
                "예산_t3연도",
                "계획예산비중_t2_pct",
                "계획예산비중_t3_pct",
                BUDGET_SHARE_CHANGE,
                "예산누락주의_두연도",
            ]
        ],
        on=["지역", "기준연도", "세부영역"],
        how="inner",
        validate="one_to_one",
    )
    numeric = ["구조환경지수_t", "구조환경지수_t1", STRUCTURAL_CHANGE, BUDGET_SHARE_CHANGE]
    if (
        result[numeric].isna().any().any()
        or not np.isfinite(result[numeric].to_numpy(dtype=float)).all()
    ):
        raise ValueError("변화량 표본에 결측 또는 무한값이 있습니다.")
    if not result["예산_t2연도"].eq(result["기준연도"] + 2).all():
        raise ValueError("예산 t+2 연도 정렬이 잘못되었습니다.")
    if not result["예산_t3연도"].eq(result["기준연도"] + 3).all():
        raise ValueError("예산 t+3 연도 정렬이 잘못되었습니다.")
    if result.duplicated(["지역", "기준연도", "세부영역"]).any():
        raise ValueError("최종 변화량 표본 키가 중복됩니다.")
    return result.sort_values(["세부영역", "지역", "기준연도"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--budget-share", type=Path, default=DEFAULT_BUDGET_SHARE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_change_response_sample(
        pd.read_csv(args.structural), pd.read_csv(args.budget_share)
    )
    expected_rows = 17 * len(BASE_YEARS) * 11
    if len(result) != expected_rows:
        raise ValueError(f"결과 행 수 불일치: 기대={expected_rows}, 실제={len(result)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"저장: {args.output} ({len(result)}행)")


if __name__ == "__main__":
    main()
