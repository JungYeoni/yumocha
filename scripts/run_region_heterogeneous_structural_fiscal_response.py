"""#102 구조환경 t → 재정대응예산 t+1/t+2 고정효과 모형을 추정한다.

세부영역별 전국 공통 반응계수와 지역별 구조환경 상호작용 계수를 분리해
산출한다. 지역별 계수는 짧은 패널의 탐색적 결과로만 해석한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scripts.build_subarea_structural_fiscal_response_sample import (
    LOG_FISCAL_OUTCOME,
    STRUCTURAL_PREDICTOR,
    fiscal_outcome_columns,
)
from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_구조환경_재정대응_반응성_표본.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"
REGION_COL = "지역"
YEAR_COL = "예산연도"
GROUP_COL = "세부영역"


def run_common_response_models(
    sample: pd.DataFrame,
    *,
    outcome: str = LOG_FISCAL_OUTCOME,
) -> pd.DataFrame:
    """세부영역별 전국 공통 구조환경 반응계수를 추정한다."""
    required = {REGION_COL, YEAR_COL, GROUP_COL, outcome, STRUCTURAL_PREDICTOR}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise KeyError(f"공통 반응모형 필수 컬럼 누락: {missing}")

    summaries: list[dict[str, object]] = []
    for subarea, group in sample.groupby(GROUP_COL, sort=False):
        usable = group.dropna(subset=[outcome, STRUCTURAL_PREDICTOR]).copy()
        model, fitted_sample = fit_two_way_fixed_effects(
            usable,
            outcome=outcome,
            predictor=STRUCTURAL_PREDICTOR,
            region_col=REGION_COL,
            year_col=YEAR_COL,
        )
        summary = summarize_fixed_effects(
            model,
            fitted_sample,
            model_name=subarea,
            outcome=outcome,
            predictor=STRUCTURAL_PREDICTOR,
            excluded_quality_rows=False,
            region_col=REGION_COL,
            year_col=YEAR_COL,
        )
        summary["구조환경1점당_예산변화율_pct"] = float(np.expm1(summary["계수"]) * 100)
        summaries.append(summary)
    result = pd.DataFrame(summaries)
    rejected, adjusted, _, _ = multipletests(result["p값"], alpha=0.05, method="fdr_bh")
    result["FDR_q값"] = adjusted
    result["FDR_0.05_유의"] = rejected
    return result


def _unidentified_rows(
    *,
    subarea: str,
    regions: list[str],
    observation_count: int,
    year_count: int,
    design_columns: int,
    design_rank: int,
    outcome: str = LOG_FISCAL_OUTCOME,
) -> list[dict[str, object]]:
    reason = (
        "지역별 구조환경 기울기와 지역·연도 고정효과 간 선형종속으로 "
        "지역별 절대 반응계수를 유일하게 추정할 수 없음"
    )
    return [
        {
            GROUP_COL: subarea,
            REGION_COL: region,
            "종속변수": outcome,
            "설명변수": STRUCTURAL_PREDICTOR,
            "추정가능": False,
            "추정제외사유": reason,
            "지역별_반응계수": np.nan,
            "구조환경1점당_예산변화율_pct": np.nan,
            "군집표준오차": np.nan,
            "t값": np.nan,
            "p값": np.nan,
            "95%신뢰구간_하한": np.nan,
            "95%신뢰구간_상한": np.nan,
            "관측치": observation_count,
            "지역수": len(regions),
            "연도수": year_count,
            "설계행렬_모수수": design_columns,
            "설계행렬_계수": design_rank,
            "지역고정효과": True,
            "연도고정효과": True,
            "지역군집표준오차": True,
        }
        for region in regions
    ]


def fit_region_interaction_model(
    sample: pd.DataFrame,
    *,
    outcome: str = LOG_FISCAL_OUTCOME,
    predictor: str = STRUCTURAL_PREDICTOR,
    group_col: str = GROUP_COL,
) -> pd.DataFrame:
    """세부영역별 지역별 기울기와 신뢰구간을 반환한다.

    설계행렬이 완전계수가 아닌 영역은 임의의 제약으로 계수를 만들지 않고,
    지역별 계수를 결측으로 기록하며 식별 불가 사유를 남긴다.
    """
    required = {REGION_COL, YEAR_COL, group_col, outcome, predictor}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise KeyError(f"지역 상호작용 모형 필수 컬럼 누락: {missing}")

    results: list[dict[str, object]] = []
    for group_label, group in sample.groupby(group_col, sort=False):
        usable = group.dropna(subset=[outcome, predictor]).copy()
        usable[outcome] = pd.to_numeric(usable[outcome], errors="coerce")
        usable[predictor] = pd.to_numeric(usable[predictor], errors="coerce")
        if usable[[outcome, predictor]].isna().any().any():
            raise ValueError(f"{group_label}에 비수치 회귀값이 있습니다.")
        if usable.duplicated([REGION_COL, YEAR_COL]).any():
            raise ValueError(f"{group_label}의 지역·예산연도 키가 중복됩니다.")

        regions = sorted(usable[REGION_COL].unique())
        years = sorted(usable[YEAR_COL].unique())
        if len(regions) < 3 or len(years) < 3:
            raise ValueError(f"{group_label}의 지역·연도 표본이 부족합니다.")

        region_full = pd.get_dummies(usable[REGION_COL], drop_first=False, dtype=float)
        region_fe = pd.get_dummies(usable[REGION_COL], drop_first=True, dtype=float)
        year_fe = pd.get_dummies(usable[YEAR_COL].astype(str), drop_first=True, dtype=float)
        predictor_values = usable[predictor].to_numpy(dtype=float)

        design = pd.DataFrame({"const": np.ones(len(usable))})
        slope_names: dict[str, str] = {}
        for region in regions:
            name = f"지역기울기_{region}"
            slope_names[region] = name
            design[name] = predictor_values * region_full[region].to_numpy(dtype=float)
        design = pd.concat(
            [
                design,
                region_fe.add_prefix("지역FE_").reset_index(drop=True),
                year_fe.add_prefix("연도FE_").reset_index(drop=True),
            ],
            axis=1,
        )
        matrix = design.to_numpy(dtype=float)
        design_rank = int(np.linalg.matrix_rank(matrix))
        if design_rank < design.shape[1]:
            results.extend(
                _unidentified_rows(
                    subarea=str(group_label),
                    regions=regions,
                    observation_count=len(usable),
                    year_count=len(years),
                    design_columns=design.shape[1],
                    design_rank=design_rank,
                    outcome=outcome,
                )
            )
            continue

        model = sm.OLS(usable[outcome].to_numpy(dtype=float), design).fit(
            cov_type="cluster",
            cov_kwds={"groups": usable[REGION_COL].to_numpy(), "use_correction": True},
            use_t=True,
        )
        names = list(design.columns)
        covariance = np.asarray(model.cov_params())
        parameters = np.asarray(model.params)
        inference_df = float(getattr(model, "df_resid_inference", model.df_resid))
        critical = float(stats.t.ppf(0.975, df=inference_df))

        for region in regions:
            parameter_index = names.index(slope_names[region])
            coefficient = float(parameters[parameter_index])
            variance = float(covariance[parameter_index, parameter_index])
            standard_error = float(np.sqrt(variance)) if variance > 0 else np.nan
            t_value = coefficient / standard_error if np.isfinite(standard_error) else np.nan
            p_value = (
                float(2 * stats.t.sf(abs(t_value), df=inference_df))
                if np.isfinite(t_value)
                else np.nan
            )
            results.append(
                {
                    GROUP_COL: group_label,
                    REGION_COL: region,
                    "종속변수": outcome,
                    "설명변수": predictor,
                    "추정가능": True,
                    "추정제외사유": "",
                    "지역별_반응계수": coefficient,
                    "구조환경1점당_예산변화율_pct": float(np.expm1(coefficient) * 100),
                    "군집표준오차": standard_error,
                    "t값": t_value,
                    "p값": p_value,
                    "95%신뢰구간_하한": coefficient - critical * standard_error,
                    "95%신뢰구간_상한": coefficient + critical * standard_error,
                    "관측치": int(model.nobs),
                    "지역수": len(regions),
                    "연도수": len(years),
                    "설계행렬_모수수": design.shape[1],
                    "설계행렬_계수": design_rank,
                    "지역고정효과": True,
                    "연도고정효과": True,
                    "지역군집표준오차": True,
                }
            )
    result = pd.DataFrame(results)
    estimable = result["추정가능"] & result["p값"].notna()
    result["FDR_q값"] = np.nan
    result["FDR_0.05_유의"] = False
    if estimable.any():
        rejected, adjusted, _, _ = multipletests(
            result.loc[estimable, "p값"], alpha=0.05, method="fdr_bh"
        )
        result.loc[estimable, "FDR_q값"] = adjusted
        result.loc[estimable, "FDR_0.05_유의"] = rejected
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag-years", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lag_years == 1 and args.sample == DEFAULT_SAMPLE:
        args.sample = DEFAULT_SAMPLE.with_name(f"{DEFAULT_SAMPLE.stem}_t+1.csv")
    if not args.sample.is_file():
        raise FileNotFoundError(f"분석 표본이 없습니다: {args.sample}")
    sample = pd.read_csv(args.sample)
    _, outcome = fiscal_outcome_columns(args.lag_years)

    common = run_common_response_models(sample, outcome=outcome)
    regional = fit_region_interaction_model(sample, outcome=outcome)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.lag_years == 2 else "_t+1"
    common_output = (
        args.output_dir / f"2016-2024_세부영역별_구조환경_재정대응_공통반응계수{suffix}.csv"
    )
    regional_output = (
        args.output_dir / f"2016-2024_세부영역별_시도별_구조환경_재정대응_반응계수{suffix}.csv"
    )
    common.to_csv(common_output, index=False, encoding="utf-8-sig")
    regional.to_csv(regional_output, index=False, encoding="utf-8-sig")

    display_columns = ["모형", "계수", "군집표준오차", "p값", "관측치", "지역수", "연도수"]
    print("세부영역별 공통 반응계수")
    print(common[display_columns].round(4).to_string(index=False))
    unidentified = regional.loc[~regional["추정가능"]]
    if not unidentified.empty:
        print(
            "\n지역별 계수 식별 불가 영역: " + ", ".join(sorted(unidentified[GROUP_COL].unique()))
        )
    print(f"\n저장: {common_output} ({len(common)}행)")
    print(f"저장: {regional_output} ({len(regional)}행)")


if __name__ == "__main__":
    main()
