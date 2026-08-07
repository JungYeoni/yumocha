"""#103 구조환경 2개 군집별 3개년 평균예산–t+1·t+2 TFR 관계를 추정한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scripts.run_subarea_fiscal_tfr_regression import run_subarea_models
from src.modeling.fiscal_response import fit_two_way_fixed_effects

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_3개년평균예산_TFR_회귀표본.csv"
)
DEFAULT_CLUSTERS = REPO_ROOT / "data/processed/analysis/2016-2024_시도별_구조환경_군집.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"
BUDGET = "인구1인당_실질예산_3개년평균"
PREDICTOR = "log1p_3개년평균예산"
INTERACTION = "예산×구조환경군집2"


def merge_clusters(sample: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """17개 지역의 고정 군집을 회귀표본에 many-to-one으로 결합한다."""
    required = {"region", "군집_2개", "군집_3개"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"군집 파일 필수 컬럼 누락: {missing}")
    if clusters["region"].duplicated().any() or len(clusters) != 17:
        raise ValueError("군집 파일은 17개 시도별 한 행이어야 합니다.")
    merged = sample.merge(
        clusters[["region", "군집_2개", "군집_3개"]].rename(columns={"region": "지역"}),
        on="지역",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        raise ValueError("회귀표본에 군집이 매칭되지 않은 지역이 있습니다.")
    return merged.drop(columns="_merge")


def add_bh(table: pd.DataFrame, group_columns: list[str], p_column: str = "p값") -> pd.DataFrame:
    result = table.copy()
    result["FDR_q값"] = np.nan
    result["FDR_0.05_유의"] = False
    groupby_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, index in result.groupby(groupby_key, sort=False).groups.items():
        valid_index = result.loc[index, p_column].dropna().index
        if len(valid_index) == 0:
            continue
        rejected, adjusted, _, _ = multipletests(result.loc[valid_index, p_column], method="fdr_bh")
        result.loc[valid_index, "FDR_q값"] = adjusted
        result.loc[valid_index, "FDR_0.05_유의"] = rejected
    return result


def run_subgroup_models(sample: pd.DataFrame, cluster_count: int = 2) -> pd.DataFrame:
    """지정한 구조환경 군집에서 같은 세부영역별 모형을 각각 반복한다."""
    tables = []
    cluster_column = f"군집_{cluster_count}개"
    for lag in (1, 2):
        outcome_column = f"합계출산율_t+{lag}"
        for cluster_id in range(1, cluster_count + 1):
            subset = sample.loc[sample[cluster_column].eq(cluster_id)].copy()
            subset["합계출산율"] = subset[outcome_column]
            subset = subset.dropna(subset=["합계출산율", BUDGET])
            region_count = subset["지역"].nunique()
            if region_count < 3:
                counts = subset.groupby("세부영역").size()
                unavailable = pd.DataFrame({"모형": list(dict.fromkeys(sample["세부영역"]))})
                unavailable["계수"] = np.nan
                unavailable["p값"] = np.nan
                unavailable["관측치"] = unavailable["모형"].map(counts).fillna(0).astype(int)
                unavailable["지역수"] = region_count
                unavailable["연도수"] = subset["연도"].nunique()
                unavailable["추정가능"] = False
                unavailable["추정불가사유"] = "시도 군집표준오차 산출에 필요한 독립 지역 수 부족"
                unavailable.insert(0, "군집", cluster_id)
                unavailable.insert(0, "시차", f"t+{lag}")
                tables.append(unavailable)
                continue
            table = run_subarea_models(subset, lag_column=BUDGET)
            table["추정가능"] = True
            table["추정불가사유"] = ""
            table.insert(0, "군집", cluster_id)
            table.insert(0, "시차", f"t+{lag}")
            tables.append(table)
    return add_bh(pd.concat(tables, ignore_index=True), ["시차", "군집"])


def _coefficient_contrast(
    model, weights: dict[str, float]
) -> tuple[float, float, float, float, float]:
    """회귀계수 선형결합의 계수·표준오차·p값·신뢰구간을 계산한다."""
    names = list(model.model.exog_names)
    vector = np.zeros(len(names))
    for name, weight in weights.items():
        vector[names.index(name)] = weight
    params = np.asarray(model.params)
    covariance = np.asarray(model.cov_params())
    coefficient = float(vector @ params)
    variance = float(vector @ covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    degrees = float(getattr(model, "df_resid_inference", model.df_resid))
    critical = float(stats.t.ppf(0.975, degrees))
    p_value = (
        float(2 * stats.t.sf(abs(coefficient / standard_error), degrees))
        if standard_error > 0
        else float("nan")
    )
    return (
        coefficient,
        standard_error,
        p_value,
        coefficient - critical * standard_error,
        coefficient + critical * standard_error,
    )


def run_interaction_models(sample: pd.DataFrame) -> pd.DataFrame:
    """전체 표본에서 군집별 예산계수와 두 계수의 차이를 직접 추정한다."""
    rows = []
    for lag in (1, 2):
        outcome = f"합계출산율_t+{lag}"
        for subarea, group in sample.groupby("세부영역", sort=False):
            usable = group.dropna(subset=[outcome, BUDGET]).copy()
            if set(usable["군집_2개"].unique()) != {1, 2}:
                rows.append(
                    {
                        "시차": f"t+{lag}",
                        "세부영역": subarea,
                        "군집간_차이_p값": np.nan,
                        "추정가능": False,
                        "추정불가사유": "세부영역 회귀표본에 2개 군집이 모두 존재하지 않음",
                        "관측치": len(usable),
                        "지역수": usable["지역"].nunique(),
                        "연도수": usable["연도"].nunique(),
                    }
                )
                continue
            usable[PREDICTOR] = np.log1p(usable[BUDGET])
            usable[INTERACTION] = usable[PREDICTOR] * usable["군집_2개"].eq(2).astype(int)
            model, fitted = fit_two_way_fixed_effects(
                usable,
                outcome=outcome,
                predictor=PREDICTOR,
                controls=[INTERACTION],
            )
            beta1, se1, p1, lower1, upper1 = _coefficient_contrast(model, {PREDICTOR: 1.0})
            delta, se_delta, p_delta, _, _ = _coefficient_contrast(model, {INTERACTION: 1.0})
            beta2, se2, p2, lower2, upper2 = _coefficient_contrast(
                model, {PREDICTOR: 1.0, INTERACTION: 1.0}
            )
            rows.append(
                {
                    "시차": f"t+{lag}",
                    "세부영역": subarea,
                    "군집1_계수": beta1,
                    "군집1_군집표준오차": se1,
                    "군집1_p값": p1,
                    "군집1_95%신뢰구간_하한": lower1,
                    "군집1_95%신뢰구간_상한": upper1,
                    "군집2_계수": beta2,
                    "군집2_군집표준오차": se2,
                    "군집2_p값": p2,
                    "군집2_95%신뢰구간_하한": lower2,
                    "군집2_95%신뢰구간_상한": upper2,
                    "군집간_계수차": delta,
                    "군집간_차이_표준오차": se_delta,
                    "군집간_차이_p값": p_delta,
                    "관측치": int(model.nobs),
                    "지역수": int(fitted["지역"].nunique()),
                    "연도수": int(fitted["연도"].nunique()),
                    "추정가능": True,
                    "추정불가사유": "",
                }
            )
    return add_bh(pd.DataFrame(rows), ["시차"], p_column="군집간_차이_p값")


def run_three_cluster_interaction_models(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3개 군집별 계수와 모든 군집 쌍의 계수 차이를 추정한다."""
    coefficient_rows = []
    contrast_rows = []
    interaction_names = {2: "예산×구조환경군집3_2", 3: "예산×구조환경군집3_3"}
    for lag in (1, 2):
        outcome = f"합계출산율_t+{lag}"
        for subarea, group in sample.groupby("세부영역", sort=False):
            usable = group.dropna(subset=[outcome, BUDGET]).copy()
            if set(usable["군집_3개"].unique()) != {1, 2, 3}:
                reason = "세부영역 회귀표본에 3개 군집이 모두 존재하지 않음"
                for cluster_id in (1, 2, 3):
                    coefficient_rows.append(
                        {
                            "시차": f"t+{lag}",
                            "세부영역": subarea,
                            "군집": cluster_id,
                            "p값": np.nan,
                            "추정가능": False,
                            "추정불가사유": reason,
                            "관측치": len(usable),
                            "지역수": usable["지역"].nunique(),
                            "연도수": usable["연도"].nunique(),
                        }
                    )
                for first, second in ((1, 2), (1, 3), (2, 3)):
                    contrast_rows.append(
                        {
                            "시차": f"t+{lag}",
                            "세부영역": subarea,
                            "군집쌍": f"{second}-{first}",
                            "p값": np.nan,
                            "추정가능": False,
                            "추정불가사유": reason,
                        }
                    )
                continue
            usable[PREDICTOR] = np.log1p(usable[BUDGET])
            for cluster_id, interaction_name in interaction_names.items():
                usable[interaction_name] = usable[PREDICTOR] * usable["군집_3개"].eq(
                    cluster_id
                ).astype(int)
            model, fitted = fit_two_way_fixed_effects(
                usable,
                outcome=outcome,
                predictor=PREDICTOR,
                controls=list(interaction_names.values()),
            )
            cluster_weights = {
                1: {PREDICTOR: 1.0},
                2: {PREDICTOR: 1.0, interaction_names[2]: 1.0},
                3: {PREDICTOR: 1.0, interaction_names[3]: 1.0},
            }
            for cluster_id, weights in cluster_weights.items():
                coefficient, se, p_value, lower, upper = _coefficient_contrast(model, weights)
                coefficient_rows.append(
                    {
                        "시차": f"t+{lag}",
                        "세부영역": subarea,
                        "군집": cluster_id,
                        "계수": coefficient,
                        "군집표준오차": se,
                        "p값": p_value,
                        "95%신뢰구간_하한": lower,
                        "95%신뢰구간_상한": upper,
                        "관측치": int(model.nobs),
                        "지역수": int(fitted["지역"].nunique()),
                        "연도수": int(fitted["연도"].nunique()),
                        "추정가능": True,
                        "추정불가사유": "",
                    }
                )
            for first, second in ((1, 2), (1, 3), (2, 3)):
                weights = cluster_weights[second].copy()
                for name, weight in cluster_weights[first].items():
                    weights[name] = weights.get(name, 0.0) - weight
                difference, se, p_value, lower, upper = _coefficient_contrast(model, weights)
                contrast_rows.append(
                    {
                        "시차": f"t+{lag}",
                        "세부영역": subarea,
                        "군집쌍": f"{second}-{first}",
                        "계수차이": difference,
                        "군집표준오차": se,
                        "p값": p_value,
                        "95%신뢰구간_하한": lower,
                        "95%신뢰구간_상한": upper,
                        "추정가능": True,
                        "추정불가사유": "",
                    }
                )
    coefficients = add_bh(pd.DataFrame(coefficient_rows), ["시차", "군집"])
    contrasts = add_bh(pd.DataFrame(contrast_rows), ["시차", "군집쌍"])
    return coefficients, contrasts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = merge_clusters(pd.read_csv(args.sample), pd.read_csv(args.clusters))
    subgroup = run_subgroup_models(sample)
    interaction = run_interaction_models(sample)
    subgroup_three = run_subgroup_models(sample, cluster_count=3)
    interaction_three, contrasts_three = run_three_cluster_interaction_models(sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subgroup.to_csv(
        args.output_dir / "2016-2024_구조환경_2개군집별_3개년평균예산_TFR_결과.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interaction.to_csv(
        args.output_dir / "2016-2024_구조환경군집_예산상호작용_TFR_결과.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subgroup_three.to_csv(
        args.output_dir / "2016-2024_구조환경_3개군집별_3개년평균예산_TFR_결과.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interaction_three.to_csv(
        args.output_dir / "2016-2024_구조환경_3개군집_예산상호작용_TFR_계수.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrasts_three.to_csv(
        args.output_dir / "2016-2024_구조환경_3개군집_예산상호작용_TFR_계수차이.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("[군집별 BH 보정 후 유의]")
    print(
        subgroup.loc[
            subgroup["FDR_0.05_유의"], ["시차", "군집", "모형", "계수", "p값", "FDR_q값"]
        ].to_string(index=False)
    )
    print("[3개 군집 부분표본 BH 보정 후 유의—탐색 전용]")
    print(
        subgroup_three.loc[
            subgroup_three["FDR_0.05_유의"],
            ["시차", "군집", "모형", "계수", "p값", "FDR_q값", "지역수"],
        ].to_string(index=False)
    )
    print("[3개 군집 계수차이 BH 보정 후 유의—탐색 전용]")
    print(
        contrasts_three.loc[
            contrasts_three["FDR_0.05_유의"],
            ["시차", "군집쌍", "세부영역", "계수차이", "p값", "FDR_q값"],
        ].to_string(index=False)
    )
    print("[군집 간 차이 BH 보정 후 유의]")
    print(
        interaction.loc[
            interaction["FDR_0.05_유의"],
            ["시차", "세부영역", "군집간_계수차", "군집간_차이_p값", "FDR_q값"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
