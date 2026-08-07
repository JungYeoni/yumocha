"""재정반응성 지역·연도 고정효과 모형 유틸."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"고정효과 모형 필수 컬럼 누락: {missing}")


def fit_two_way_fixed_effects(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictor: str,
    controls: Sequence[str] | None = None,
    region_col: str = "지역",
    year_col: str = "연도",
) -> tuple[RegressionResultsWrapper, pd.DataFrame]:
    """지역·연도 고정효과 OLS를 지역 군집표준오차로 추정한다.

    결측 관측치는 암묵적으로 제거하지 않고 오류로 처리한다. 호출자가 분석
    표본을 명시적으로 확정한 뒤 전달해야 모형별 표본 차이를 추적할 수 있다.
    군집 수가 적은 점을 고려해 statsmodels의 유한표본 보정과 t분포 추론을
    사용한다. controls는 강건성체크용 통제변수(예: S_i,t-1)를 추가할 때
    쓰며, 생략하면 기존과 동일한 단변량 고정효과 모형이다.
    """

    control_columns = list(controls) if controls is not None else []
    required = {outcome, predictor, region_col, year_col, *control_columns}
    _require_columns(data, required)
    if data.empty:
        raise ValueError("고정효과 모형 분석 표본이 비어 있습니다.")
    if data.duplicated([region_col, year_col]).any():
        duplicates = (
            data.loc[data.duplicated([region_col, year_col], keep=False), [region_col, year_col]]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"고정효과 모형 지역×연도 키 중복: {duplicates}")

    numeric_columns = [outcome, predictor, *control_columns]
    sample = data[[region_col, year_col, *numeric_columns]].copy()
    for column in numeric_columns:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    invalid = sample[numeric_columns].isna().any(axis=1) | ~np.isfinite(
        sample[numeric_columns].to_numpy(dtype=float)
    ).all(axis=1)
    if invalid.any():
        rows = sample.loc[invalid].head(10).to_dict(orient="records")
        raise ValueError(f"고정효과 모형에 결측 또는 비수치 관측치가 있습니다: {rows}")

    region_count = sample[region_col].nunique()
    year_count = sample[year_col].nunique()
    if region_count < 2:
        raise ValueError("지역 군집표준오차에는 지역이 2개 이상 필요합니다.")
    if year_count < 2:
        raise ValueError("연도 고정효과에는 연도가 2개 이상 필요합니다.")
    if sample[predictor].nunique() < 2:
        raise ValueError(f"설명변수 {predictor}에 변동이 없습니다.")
    for column in control_columns:
        if sample[column].nunique() < 2:
            raise ValueError(f"통제변수 {column}에 변동이 없습니다.")

    fixed_effects = pd.get_dummies(
        sample[[region_col, year_col]].astype({year_col: "string"}),
        columns=[region_col, year_col],
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            sample[[predictor, *control_columns]].astype(float).reset_index(drop=True),
            fixed_effects.reset_index(drop=True),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("고정효과 설계행렬이 완전계수가 아닙니다.")
    if len(sample) <= design.shape[1]:
        raise ValueError(
            f"고정효과 모형 잔차 자유도가 없습니다: 관측치={len(sample)}, 모수={design.shape[1]}"
        )

    model = sm.OLS(
        sample[outcome].to_numpy(dtype=float),
        design,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": sample[region_col].to_numpy(),
            "use_correction": True,
        },
        use_t=True,
    )
    return model, sample


def summarize_fixed_effects(
    model: RegressionResultsWrapper,
    sample: pd.DataFrame,
    *,
    model_name: str,
    outcome: str,
    predictor: str,
    excluded_quality_rows: bool,
    region_col: str = "지역",
    year_col: str = "연도",
) -> dict[str, object]:
    """관심 설명변수의 추정 결과와 표본 정보를 한 행으로 정리한다."""

    parameter_names = list(model.model.exog_names)
    if predictor not in parameter_names:
        raise ValueError(f"모형 결과에 설명변수 {predictor}가 없습니다.")
    index = parameter_names.index(predictor)
    coefficient = float(model.params.iloc[index])
    variance = float(np.asarray(model.cov_params())[index, index])
    if not np.isfinite(variance) or variance < 0:
        raise ValueError(f"설명변수 {predictor}의 군집분산을 계산할 수 없습니다.")
    standard_error = float(np.sqrt(variance))
    inference_df = float(getattr(model, "df_resid_inference", model.df_resid))
    t_value = coefficient / standard_error
    p_value = float(2 * stats.t.sf(abs(t_value), df=inference_df))
    critical_value = float(stats.t.ppf(0.975, df=inference_df))

    return {
        "모형": model_name,
        "종속변수": outcome,
        "설명변수": predictor,
        "누락주의관측치_제외": excluded_quality_rows,
        "계수": coefficient,
        "군집표준오차": standard_error,
        "t값": t_value,
        "p값": p_value,
        "95%신뢰구간_하한": coefficient - critical_value * standard_error,
        "95%신뢰구간_상한": coefficient + critical_value * standard_error,
        "관측치": int(model.nobs),
        "지역수": int(sample[region_col].nunique()),
        "연도수": int(sample[year_col].nunique()),
        "잔차자유도": float(model.df_resid),
        "추론자유도": inference_df,
        "지역고정효과": True,
        "연도고정효과": True,
        "지역군집표준오차": True,
    }


def run_fiscal_response_models(
    data: pd.DataFrame,
    *,
    outcomes: Mapping[str, str] | Sequence[str],
    predictor: str = "직전1년_출산율하락도",
    quality_col: str = "원자료_누락주의",
    region_col: str = "지역",
    year_col: str = "연도",
) -> tuple[pd.DataFrame, dict[str, RegressionResultsWrapper]]:
    """여러 종속변수의 전체표본·누락주의 제외 고정효과 모형을 실행한다."""

    if isinstance(outcomes, Mapping):
        outcome_specs = list(outcomes.items())
    else:
        outcome_specs = [(outcome, outcome) for outcome in outcomes]
    if not outcome_specs:
        raise ValueError("추정할 종속변수가 없습니다.")

    required = {
        predictor,
        quality_col,
        region_col,
        year_col,
        *(outcome for _, outcome in outcome_specs),
    }
    _require_columns(data, required)

    summaries: list[dict[str, object]] = []
    models: dict[str, RegressionResultsWrapper] = {}
    for sample_label, exclude_quality_rows in (("전체표본", False), ("누락주의제외", True)):
        model_data = (
            data.loc[data[quality_col].isna()].copy() if exclude_quality_rows else data.copy()
        )
        for outcome_label, outcome in outcome_specs:
            model_name = f"{outcome_label}_{sample_label}"
            model, sample = fit_two_way_fixed_effects(
                model_data,
                outcome=outcome,
                predictor=predictor,
                region_col=region_col,
                year_col=year_col,
            )
            summaries.append(
                summarize_fixed_effects(
                    model,
                    sample,
                    model_name=model_name,
                    outcome=outcome,
                    predictor=predictor,
                    excluded_quality_rows=exclude_quality_rows,
                    region_col=region_col,
                    year_col=year_col,
                )
            )
            models[model_name] = model

    return pd.DataFrame(summaries), models
