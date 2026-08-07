"""#98 탐색적 비교 — 세부영역(11개) 대신 대영역(4개) 단위로 재정-TFR 관련성을 본다.

2026-08-07 팀 결정 범위에는 없는 탐색적 비교다(#98 이슈 본문의 확정 사항이
아님). 세부영역별 회귀가 세부영역당 136개 관측치로 검정력이 낮다는 한계를
어느 정도 완화할 수 있는지, 대영역 단위로 묶으면 결론이 달라지는지 확인하기
위한 비교용 분석이다.

대영역 taxonomy는 새로 정의하지 않고 구조환경지수(#82/#96) 산출물의
category 컬럼을 그대로 재사용한다(재정 세부영역 <-> 구조환경 subcategory
매핑은 scripts/build_subarea_fiscal_response_regression_sample.py의
FISCAL_TO_STRUCTURAL_SUBCATEGORY를 그대로 쓴다).

두 가지 버전을 비교한다:
1. 단순 대영역 회귀: 세부영역 회귀(run_subarea_fiscal_tfr_regression.py)와
   완전히 동일한 방식으로 4개 대영역 각각 F_i,t-1을 합산해 개별 추정한다.
   관측치 수는 세부영역과 동일하게 대영역당 136개다 — 세부영역을 묶어도
   지역×연도 패널 자체가 늘어나지는 않는다.
2. 상호통제 다변량 회귀: 4개 대영역의 F_i,t-1을 하나의 모형에 동시에 넣어
   서로를 통제한다("이 대영역 예산의 효과가 다른 대영역 예산과 상관되어
   생기는 착시인지" 확인). 표본은 여전히 136개이며 늘어나지 않는다 — 대영역
   더미×F 상호작용으로 관측치를 4배(약 544개)로 "늘리는" 방식은 종속변수
   (TFR)가 대영역과 무관하게 지역·연도에서만 결정되므로 같은 TFR값을
   4번 복제해 별도 목적함수로 근사하는 것과 같아 해석이 불명확해 채택하지
   않았다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGRESSION_SAMPLE = (
    REPO_ROOT / "data" / "processed" / "analysis" / "2016-2024_세부영역별_재정반응성_회귀표본.csv"
)
DEFAULT_STRUCTURAL_SUBCATEGORY = (
    REPO_ROOT
    / "data"
    / "processed"
    / "structural_index"
    / "structural_index_pooled_subcategory_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"

OUTCOME = "합계출산율"
LAG1_COLUMN = "인구1인당_실질예산_전년도"
LAG2_COLUMN = "인구1인당_실질예산_전전년도"


def build_major_category_mapping(structural_scores: pd.DataFrame) -> dict[str, str]:
    """재정 세부영역(11개) -> 대영역(4개) 매핑을 구조환경지수 category 컬럼에서 만든다."""
    from scripts.build_subarea_fiscal_response_regression_sample import (
        FISCAL_TO_STRUCTURAL_SUBCATEGORY,
    )

    pairs = structural_scores[["subcategory", "category"]].drop_duplicates()
    required_subcategories = set(FISCAL_TO_STRUCTURAL_SUBCATEGORY.values())
    missing_category = sorted(
        pairs.loc[
            pairs["subcategory"].isin(required_subcategories) & pairs["category"].isna(),
            "subcategory",
        ].unique()
    )
    if missing_category:
        raise ValueError(f"category가 없는 subcategory: {missing_category}")
    conflicting = sorted(
        pairs.loc[pairs.duplicated("subcategory", keep=False), "subcategory"].unique()
    )
    if conflicting:
        raise ValueError(f"subcategory가 여러 category에 매핑됩니다: {conflicting}")
    subcategory_to_category = pairs.set_index("subcategory")["category"].to_dict()
    missing = required_subcategories - set(subcategory_to_category)
    if missing:
        raise ValueError(f"구조환경지수 category 정보가 없는 subcategory: {sorted(missing)}")

    return {
        fiscal_subarea: subcategory_to_category[structural_subcategory]
        for fiscal_subarea, structural_subcategory in FISCAL_TO_STRUCTURAL_SUBCATEGORY.items()
    }


def _sum_or_missing(values: pd.Series) -> float:
    if values.isna().any():
        return np.nan
    return float(values.sum())


def aggregate_to_major_category(sample: pd.DataFrame, category_map: dict[str, str]) -> pd.DataFrame:
    """세부영역별 F_i,t-1·F_i,t-2를 대영역 단위로 합산한다.

    구성 세부영역 중 하나라도 결측이면 대영역 합계도 결측으로 둔다(암묵적
    skipna 합산은 실제보다 예산을 과소평가하게 만든다).
    """
    working = sample.copy()
    working["대영역"] = working["세부영역"].map(category_map)
    if working["대영역"].isna().any():
        unmapped = sorted(working.loc[working["대영역"].isna(), "세부영역"].unique())
        raise ValueError(f"대영역 매핑이 없는 세부영역: {unmapped}")

    tfr_nunique = working.groupby(["지역", "연도"])[OUTCOME].nunique()
    if (tfr_nunique > 1).any():
        raise ValueError("합계출산율이 세부영역마다 다릅니다 — 집계 전 원본 데이터를 확인하십시오.")

    grouped = working.groupby(["지역", "연도", "대영역"], sort=False)
    result = grouped.agg(
        **{
            OUTCOME: (OUTCOME, "first"),
            LAG1_COLUMN: (LAG1_COLUMN, _sum_or_missing),
            LAG2_COLUMN: (LAG2_COLUMN, _sum_or_missing),
        }
    ).reset_index()
    return result


def run_mutually_controlled_model(
    sample: pd.DataFrame,
    *,
    categories: list[str],
    lag_column_template: str,
) -> pd.DataFrame:
    """대영역 4개의 log1p(F_i,t-1)을 한 모형에 동시에 넣어 서로 통제하며 추정한다."""
    from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

    working = sample.copy()
    predictor_columns: dict[str, str] = {}
    for category in categories:
        raw_column = lag_column_template.format(category=category)
        predictor_column = f"log1p_{raw_column}"
        working[predictor_column] = np.log1p(working[raw_column])
        predictor_columns[category] = predictor_column

    usable = working.dropna(subset=list(predictor_columns.values())).copy()

    summaries: list[dict[str, object]] = []
    for category in categories:
        predictor = predictor_columns[category]
        controls = [predictor_columns[c] for c in categories if c != category]
        model, fitted_sample = fit_two_way_fixed_effects(
            usable,
            outcome=OUTCOME,
            predictor=predictor,
            controls=controls,
        )
        summaries.append(
            summarize_fixed_effects(
                model,
                fitted_sample,
                model_name=category,
                outcome=OUTCOME,
                predictor=predictor,
                excluded_quality_rows=False,
            )
        )
    return pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-sample", type=Path, default=DEFAULT_REGRESSION_SAMPLE)
    parser.add_argument(
        "--structural-subcategory-scores", type=Path, default=DEFAULT_STRUCTURAL_SUBCATEGORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    from scripts.run_subarea_fiscal_tfr_regression import run_subarea_models

    args = parse_args()
    for path in (args.regression_sample, args.structural_subcategory_scores):
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    sample = pd.read_csv(args.regression_sample)
    structural_scores = pd.read_csv(args.structural_subcategory_scores)

    category_map = build_major_category_mapping(structural_scores)
    aggregated = aggregate_to_major_category(sample, category_map)

    categories = sorted(set(category_map.values()))

    simple_result = run_subarea_models(aggregated, lag_column=LAG1_COLUMN, group_col="대영역")
    simple_result.insert(0, "모형버전", "단순_대영역(F_t-1)")

    wide = aggregated.pivot_table(
        index=["지역", "연도"], columns="대영역", values=[LAG1_COLUMN], aggfunc="first"
    )
    wide.columns = [f"{LAG1_COLUMN}__{category}" for _, category in wide.columns]
    wide = wide.reset_index()
    tfr_by_region_year = aggregated[["지역", "연도", OUTCOME]].drop_duplicates()
    wide = wide.merge(tfr_by_region_year, on=["지역", "연도"], validate="one_to_one")

    controlled_result = run_mutually_controlled_model(
        wide, categories=categories, lag_column_template=f"{LAG1_COLUMN}__{{category}}"
    )
    controlled_result.insert(0, "모형버전", "상호통제_대영역(F_t-1)")

    combined = pd.concat([simple_result, controlled_result], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_대영역별_재정_TFR_비교_고정효과_결과.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = ["모형버전", "모형", "계수", "군집표준오차", "p값", "관측치"]
    print(combined[display_columns].round(4).to_string(index=False))
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
