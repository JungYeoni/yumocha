"""#110 구조환경 군집별 전체 재정대응예산과 후행 TFR의 관계를 추정한다.

11개 세부영역의 실질 1인당 3개년 평균예산을 지역×연도 단위로 합산하고,
고정된 구조환경 k=2·k=3 군집별로 t+1·t+2 합계출산율과의 조건부 관련성을
추정한다. 부분표본 결과와 전체표본 예산×군집 상호작용 결과를 함께 저장한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_cluster_fiscal_tfr_regression import (
    _coefficient_contrast,
    _fit_coefficient_only,
    add_bh,
    merge_clusters,
)
from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_SAMPLE = ANALYSIS_DIR / "2016-2024_세부영역별_3개년평균예산_TFR_회귀표본.csv"
DEFAULT_CLUSTERS = ANALYSIS_DIR / "2016-2024_시도별_구조환경_군집.csv"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR

BUDGET = "인구1인당_실질예산_3개년평균"
TOTAL_BUDGET = "전체_인구1인당_실질예산_3개년평균"
PREDICTOR = "log1p_전체_3개년평균예산"
EXPECTED_SUBAREA_COUNT = 11


def build_total_budget_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """세부영역 패널을 지역×연도 전체예산 패널로 집계한다."""
    required = {
        "지역",
        "연도",
        "세부영역",
        BUDGET,
        "합계출산율_t+1",
        "합계출산율_t+2",
    }
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"전체예산 표본 필수 컬럼 누락: {missing}")
    if sample.duplicated(["지역", "연도", "세부영역"]).any():
        raise ValueError("세부영역 회귀표본에 지역×연도×세부영역 중복이 있습니다.")
    if sample[BUDGET].dropna().lt(0).any():
        raise ValueError("실질 1인당 3개년 평균예산에는 음수가 올 수 없습니다.")

    expected_subareas = set(sample["세부영역"].dropna().unique())
    if len(expected_subareas) != EXPECTED_SUBAREA_COUNT:
        raise ValueError(
            f"전체예산은 정확히 {EXPECTED_SUBAREA_COUNT}개 세부영역이어야 합니다: "
            f"관측={len(expected_subareas)}"
        )
    observed_sets = sample.groupby(["지역", "연도"])["세부영역"].agg(set)
    incomplete = observed_sets.loc[observed_sets.ne(expected_subareas)]
    if not incomplete.empty:
        raise ValueError(f"지역×연도별 세부영역 구성이 불완전합니다: {list(incomplete.index[:5])}")

    for outcome in ("합계출산율_t+1", "합계출산율_t+2"):
        if sample.groupby(["지역", "연도"])[outcome].nunique(dropna=False).gt(1).any():
            raise ValueError(f"지역×연도 안에서 {outcome} 값이 일치하지 않습니다.")

    def complete_sum(values: pd.Series) -> float:
        return float(values.sum()) if values.notna().all() else float("nan")

    total = (
        sample.groupby(["지역", "연도"], as_index=False)
        .agg(
            **{
                TOTAL_BUDGET: (BUDGET, complete_sum),
                "합계출산율_t+1": ("합계출산율_t+1", "first"),
                "합계출산율_t+2": ("합계출산율_t+2", "first"),
                "세부영역수": ("세부영역", "nunique"),
            }
        )
        .sort_values(["지역", "연도"])
        .reset_index(drop=True)
    )
    total[PREDICTOR] = np.log1p(total[TOTAL_BUDGET])
    return total


def validate_full_panel(total: pd.DataFrame) -> None:
    """공식 입력이 17개 시도×2016~2024년 균형패널인지 확인한다."""
    expected_years = set(range(2016, 2025))
    if total["지역"].nunique() != 17 or set(total["연도"].unique()) != expected_years:
        raise ValueError("공식 전체예산 표본은 17개 시도×2016~2024년이어야 합니다.")
    counts = total.groupby("지역")["연도"].nunique()
    if len(total) != 17 * 9 or not counts.eq(9).all():
        raise ValueError("공식 전체예산 표본의 지역별 연도 구성이 불완전합니다.")


def run_subgroup_total_models(sample: pd.DataFrame, *, cluster_count: int) -> pd.DataFrame:
    """각 구조환경 군집 부분표본에서 전체예산 계수를 추정한다."""
    cluster_column = f"군집_{cluster_count}개"
    rows: list[dict[str, object]] = []
    for lag in (1, 2):
        outcome = f"합계출산율_t+{lag}"
        for cluster_id in range(1, cluster_count + 1):
            usable = sample.loc[sample[cluster_column].eq(cluster_id)].dropna(
                subset=[outcome, PREDICTOR]
            )
            region_count = usable["지역"].nunique()
            if region_count == 2:
                coefficient, observations, regions, years, residual_df = _fit_coefficient_only(
                    usable,
                    outcome=outcome,
                    predictor=PREDICTOR,
                )
                row = {
                    "계수": coefficient,
                    "군집표준오차": np.nan,
                    "p값": np.nan,
                    "95%신뢰구간_하한": np.nan,
                    "95%신뢰구간_상한": np.nan,
                    "관측치": observations,
                    "지역수": regions,
                    "연도수": years,
                    "잔차자유도": residual_df,
                    "추론가능": False,
                    "분석구분": "2개 시도 계수만 탐색",
                }
            else:
                model, fitted = fit_two_way_fixed_effects(
                    usable,
                    outcome=outcome,
                    predictor=PREDICTOR,
                )
                row = summarize_fixed_effects(
                    model,
                    fitted,
                    model_name="전체 재정대응예산",
                    outcome=outcome,
                    predictor=PREDICTOR,
                    excluded_quality_rows=False,
                )
                row["추론가능"] = True
                row["분석구분"] = "군집별 부분표본 FE"
            row.update({"군집수": cluster_count, "군집": cluster_id, "시차": f"t+{lag}"})
            rows.append(row)
    return pd.DataFrame(rows)


def run_total_interaction_models(
    sample: pd.DataFrame, *, cluster_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체표본에서 군집별 전체예산 계수와 군집 간 차이를 직접 검정한다."""
    cluster_column = f"군집_{cluster_count}개"
    coefficient_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    interaction_names = {
        cluster_id: f"전체예산×구조환경군집{cluster_count}_{cluster_id}"
        for cluster_id in range(2, cluster_count + 1)
    }
    for lag in (1, 2):
        outcome = f"합계출산율_t+{lag}"
        usable = sample.dropna(subset=[outcome, PREDICTOR]).copy()
        observed_clusters = set(usable[cluster_column].unique())
        expected_clusters = set(range(1, cluster_count + 1))
        if observed_clusters != expected_clusters:
            raise ValueError(
                f"k={cluster_count} 상호작용 표본의 군집 누락: 관측={sorted(observed_clusters)}"
            )
        for cluster_id, interaction_name in interaction_names.items():
            usable[interaction_name] = usable[PREDICTOR] * usable[cluster_column].eq(
                cluster_id
            ).astype(int)
        model, fitted = fit_two_way_fixed_effects(
            usable,
            outcome=outcome,
            predictor=PREDICTOR,
            controls=list(interaction_names.values()),
        )
        cluster_weights = {
            cluster_id: {
                PREDICTOR: 1.0,
                **({interaction_names[cluster_id]: 1.0} if cluster_id in interaction_names else {}),
            }
            for cluster_id in range(1, cluster_count + 1)
        }
        for cluster_id, weights in cluster_weights.items():
            coefficient, se, p_value, lower, upper = _coefficient_contrast(model, weights)
            coefficient_rows.append(
                {
                    "군집수": cluster_count,
                    "시차": f"t+{lag}",
                    "군집": cluster_id,
                    "계수": coefficient,
                    "군집표준오차": se,
                    "p값": p_value,
                    "95%신뢰구간_하한": lower,
                    "95%신뢰구간_상한": upper,
                    "관측치": int(model.nobs),
                    "지역수": int(fitted["지역"].nunique()),
                    "연도수": int(fitted["연도"].nunique()),
                }
            )
        for first in range(1, cluster_count):
            for second in range(first + 1, cluster_count + 1):
                weights = cluster_weights[second].copy()
                for name, weight in cluster_weights[first].items():
                    weights[name] = weights.get(name, 0.0) - weight
                difference, se, p_value, lower, upper = _coefficient_contrast(model, weights)
                contrast_rows.append(
                    {
                        "군집수": cluster_count,
                        "시차": f"t+{lag}",
                        "군집쌍": f"{second}-{first}",
                        "계수차이": difference,
                        "군집표준오차": se,
                        "p값": p_value,
                        "95%신뢰구간_하한": lower,
                        "95%신뢰구간_상한": upper,
                        "관측치": int(model.nobs),
                        "지역수": int(fitted["지역"].nunique()),
                        "연도수": int(fitted["연도"].nunique()),
                    }
                )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(contrast_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = build_total_budget_sample(pd.read_csv(args.sample))
    validate_full_panel(total)
    total = merge_clusters(total, pd.read_csv(args.clusters))
    subgroup = pd.concat(
        [run_subgroup_total_models(total, cluster_count=k) for k in (2, 3)],
        ignore_index=True,
    )
    subgroup = add_bh(subgroup, ["군집수"])
    coefficient_tables = []
    contrast_tables = []
    for cluster_count in (2, 3):
        coefficients, contrasts = run_total_interaction_models(total, cluster_count=cluster_count)
        coefficient_tables.append(coefficients)
        contrast_tables.append(contrasts)
    interaction_coefficients = pd.concat(coefficient_tables, ignore_index=True)
    interaction_contrasts = pd.concat(contrast_tables, ignore_index=True)
    interaction_coefficients = add_bh(interaction_coefficients, ["군집수"])
    interaction_contrasts = add_bh(interaction_contrasts, ["군집수"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "2016-2024_구조환경군집별_전체3개년평균예산_TFR_회귀표본.csv": total,
        "2016-2024_구조환경군집별_전체3개년평균예산_TFR_부분표본결과.csv": subgroup,
        "2016-2024_구조환경군집_전체예산상호작용_TFR_계수.csv": interaction_coefficients,
        "2016-2024_구조환경군집_전체예산상호작용_TFR_계수차이.csv": interaction_contrasts,
    }
    for filename, table in outputs.items():
        table.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")
    print(subgroup[["군집수", "군집", "시차", "계수", "p값", "관측치", "지역수"]])
    print(interaction_contrasts[["군집수", "군집쌍", "시차", "계수차이", "p값"]])


if __name__ == "__main__":
    main()
