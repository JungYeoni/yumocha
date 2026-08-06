"""#70 결측처리 완료 패널로 #82 구조환경지수 두 시나리오를 산출한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.structural_index import (
    DEFAULT_STRUCTURAL_REGIONS,
    DEFAULT_STRUCTURAL_YEARS,
    REAL_COST_INDICATOR_IDS,
    compute_structural_index,
    deflate_structural_cost_indicators,
    load_structural_index_weights,
    load_structural_indicator_manifest,
    prepare_processed_structural_panel,
    run_structural_index_scenarios,
    standardize_structural_indicators,
    validate_structural_index_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/구조환경지표_28개_결측처리후_본계열패널.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/structural_index"
DEFAULT_REPORT = REPO_ROOT / "reports/methodology/20260806_구조환경지수_실제패널_산출_QA.md"
DEFAULT_CPI = REPO_ROOT / "data/lookup/연도별_소비자물가지수.csv"
FAMILY_FRIENDLY_INDICATOR_ID = "family_friendly_certification_rate"
PARENTAL_LEAVE_INDICATOR_ID = "parental_leave_usage"
FAMILY_FRIENDLY_EXCLUDED_YEARS = (2016, 2017)


def compare_scenarios(results: dict) -> pd.DataFrame:
    """pooled와 yearly 최종지수·연도 내 순위를 동일 지역·연도에서 비교한다."""
    frames = {}
    for method in ("pooled", "yearly"):
        frames[method] = (
            results[method]
            .final_index[["region", "year", "final_index", "rank"]]
            .rename(
                columns={
                    "final_index": f"final_index_{method}",
                    "rank": f"rank_{method}",
                }
            )
        )
    comparison = frames["pooled"].merge(
        frames["yearly"], on=["region", "year"], how="inner", validate="one_to_one"
    )
    comparison["score_diff_yearly_minus_pooled"] = (
        comparison["final_index_yearly"] - comparison["final_index_pooled"]
    )
    comparison["abs_score_diff"] = comparison["score_diff_yearly_minus_pooled"].abs()
    comparison["rank_diff_yearly_minus_pooled"] = (
        comparison["rank_yearly"] - comparison["rank_pooled"]
    )
    comparison["abs_rank_diff"] = comparison["rank_diff_yearly_minus_pooled"].abs()
    return comparison.sort_values(["year", "region"]).reset_index(drop=True)


def run_family_friendly_weight_transfer_scenarios(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    methods=("pooled", "yearly"),
    expected_regions=DEFAULT_STRUCTURAL_REGIONS,
    expected_years=DEFAULT_STRUCTURAL_YEARS,
) -> dict:
    """2016·2017 가족친화 가중치 몫을 육아휴직 점수로 이전해 지수를 산출한다."""
    prepared = panel.copy()
    target = prepared["indicator_id"].eq(FAMILY_FRIENDLY_INDICATOR_ID) & prepared["year"].isin(
        FAMILY_FRIENDLY_EXCLUDED_YEARS
    )
    fit_values = prepared.loc[
        prepared["indicator_id"].eq(FAMILY_FRIENDLY_INDICATOR_ID)
        & ~prepared["year"].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS)
        & prepared["region"].ne("전국"),
        "value",
    ].astype(float)
    if fit_values.empty:
        raise ValueError("가족친화 지표의 2018년 이후 표준화 기준값이 없습니다.")

    # 완전 격자 검증은 유지하되, 제외 연도 값이 pooled 최솟값·최댓값에 영향을 주지
    # 않도록 유효 관측구간의 최솟값을 임시 앵커로 사용한다. 최종 점수는 아래에서
    # 육아휴직 점수로 대체되므로 이 임시값은 지수에 직접 사용되지 않는다.
    prepared.loc[target, "value"] = float(fit_values.min())
    indicator_ids = weights["id"].astype(str).tolist()
    results = {}
    for method in methods:
        standardized = standardize_structural_indicators(
            prepared,
            method=method,
            expected_regions=expected_regions,
            expected_years=expected_years,
            expected_indicator_ids=indicator_ids,
        )

        if method == "pooled":
            valid = standardized["indicator_id"].eq(FAMILY_FRIENDLY_INDICATOR_ID) & ~standardized[
                "year"
            ].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS)
            values = standardized.loc[valid, "value"].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            vmin = float(values.min())
            vmax = float(values.max())
            standardized.loc[valid, "z_score"] = (values - mean) / std
            standardized.loc[valid, "score_0_100"] = 100.0 * (values - vmin) / (vmax - vmin)
            standardized.loc[valid, "directional_score"] = standardized.loc[valid, "score_0_100"]
            standardized.loc[valid, "group_mean"] = mean
            standardized.loc[valid, "group_std"] = std
            standardized.loc[valid, "group_min_z"] = standardized.loc[valid, "z_score"].min()
            standardized.loc[valid, "group_max_z"] = standardized.loc[valid, "z_score"].max()

        target_scores = standardized["indicator_id"].eq(
            FAMILY_FRIENDLY_INDICATOR_ID
        ) & standardized["year"].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS)
        source = standardized.loc[
            standardized["indicator_id"].eq(PARENTAL_LEAVE_INDICATOR_ID)
            & standardized["year"].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS),
            ["region", "year", "score_0_100", "directional_score"],
        ].rename(
            columns={
                "score_0_100": "transferred_score_0_100",
                "directional_score": "transferred_directional_score",
            }
        )
        target_index = standardized.loc[target_scores, ["region", "year"]]
        transferred = target_index.merge(
            source, on=["region", "year"], how="left", validate="one_to_one"
        )
        if transferred["transferred_directional_score"].isna().any():
            raise ValueError("2016·2017 육아휴직 점수를 가족친화 가중치에 연결하지 못했습니다.")
        standardized.loc[target_scores, "value"] = pd.NA
        standardized.loc[target_scores, "z_score"] = pd.NA
        standardized.loc[target_scores, "score_0_100"] = transferred[
            "transferred_score_0_100"
        ].to_numpy()
        standardized.loc[target_scores, "directional_score"] = transferred[
            "transferred_directional_score"
        ].to_numpy()
        standardized["weight_transfer_applied"] = False
        standardized.loc[target_scores, "weight_transfer_applied"] = True
        standardized["effective_score_source"] = standardized["indicator_id"]
        standardized.loc[target_scores, "effective_score_source"] = PARENTAL_LEAVE_INDICATOR_ID
        results[method] = compute_structural_index(standardized, weights)
    return results


def compare_family_friendly_decision(main_results: dict, raking_results: dict) -> pd.DataFrame:
    """2016·2017 가중치 이전 본계열과 raking 값을 사용한 대안계열을 비교한다."""
    main = (
        main_results["pooled"]
        .final_index[["region", "year", "final_index", "rank"]]
        .rename(columns={"final_index": "final_index_가중치이전", "rank": "rank_가중치이전"})
    )
    alternative = (
        raking_results["pooled"]
        .final_index[["region", "year", "final_index", "rank"]]
        .rename(columns={"final_index": "final_index_raking사용", "rank": "rank_raking사용"})
    )
    comparison = main.merge(alternative, on=["region", "year"], how="inner", validate="one_to_one")
    comparison["score_diff_가중치이전_minus_raking"] = (
        comparison["final_index_가중치이전"] - comparison["final_index_raking사용"]
    )
    comparison["abs_score_diff"] = comparison["score_diff_가중치이전_minus_raking"].abs()
    comparison["rank_diff_가중치이전_minus_raking"] = (
        comparison["rank_가중치이전"] - comparison["rank_raking사용"]
    )
    comparison["abs_rank_diff"] = comparison["rank_diff_가중치이전_minus_raking"].abs()
    return comparison.sort_values(["year", "region"]).reset_index(drop=True)


def build_report(
    results: dict,
    comparison: pd.DataFrame,
    nominal_results: dict,
    family_friendly_comparison: pd.DataFrame,
    input_rows: int,
    cpi_base_year: str,
    cpi_base_value: float,
) -> str:
    yearly_stats = comparison.groupby("year", sort=True).apply(
        lambda group: pd.Series(
            {
                "pearson": group["final_index_pooled"].corr(group["final_index_yearly"]),
                "score_mae": group["abs_score_diff"].mean(),
                "mean_abs_rank_diff": group["abs_rank_diff"].mean(),
                "max_abs_rank_diff": group["abs_rank_diff"].max(),
            }
        ),
        include_groups=False,
    )
    lines = [
        "# 구조환경지수 실제 패널 산출 QA",
        "",
        "## 입력과 범위",
        "",
        f"- #70 결측처리 완료 패널: {input_rows:,}행",
        "- 분석 격자: 17개 시도 × 2016–2024년 × 28개 지표 = 4,284행",
        "- 전국 행은 표준화·지수 산출에서 제외",
        "- 표준화 시나리오: pooled Min-Max, 연도별(yearly) Min-Max",
        f"- 실질화: 주택가격·임차가구 연간 주거비·사교육비·산후조리원 이용요금에 전국 연평균 CPI({cpi_base_year}) 적용",
        "",
        "## 산출 QA",
        "",
        "| 시나리오 | 지표 점수 | 최종지수 | 결측 최종지수 | 지수 범위 |",
        "|---|---:|---:|---:|---|",
    ]
    for method, result in results.items():
        final = result.final_index
        lines.append(
            f"| {method} | {len(result.indicator_scores):,} | {len(final):,} | "
            f"{int(final['final_index'].isna().sum()):,} | "
            f"{final['final_index'].min():.4f}–{final['final_index'].max():.4f} |"
        )
    lines.extend(
        [
            "",
            "모든 시나리오에서 지역·연도별 28개 지표가 완비되고 최종지수 153개가 산출됐다.",
            "전국 근로시간 9개 결측은 전국 행 자체가 분석 대상이 아니므로 결과에 유입되지 않는다.",
            "",
            "## AHP 가중치 계보",
            "",
            "- 원자료: `data/lookup/구조환경지표_AHP가중치_3방식.xlsx`의 `AHP!A7:M34`",
            "- 근거: 제주여성가족연구원 「제주지역 출산환경지수 개발 연구」 p.83 `<표 Ⅳ-3>`의 AHP 열",
            "- 조정: 31개 중 난임시술기관 보급도·학교폭력 발생률·저출생 대응 예산액을 제외하고, 각 가중치를 동일 세부영역의 잔존 지표에 원래 비율대로 재배분",
            "- 정규화: 원문 표시값 합계 0.998을 분모로 사용해 28개 조정 가중치 합을 1로 정규화",
            "- 대체: 원 연구의 분만 가능 산부인과·소아청소년과 보급도 가중치 슬롯을 분만실 병상수·소아청소년과 전문인력 보급도에 승계",
            "- 자동 검증: 원자료 AHP 시트의 28개 최종 조정값과 `configs/structural_index_weights.yaml`을 코드 기준으로 1:1 대조",
            "",
            "## 가족친화인증기업 2016·2017 처리",
            "",
            "- 2016·2017년 가족친화인증기업 비율은 본계열에서 제외하고 표준화 기준에도 포함하지 않음",
            "- 해당 연도의 가족친화 AHP 가중치 0.0434202는 같은 일·가정 양립 여건의 육아휴직 점수에 이전",
            "- 육아휴직의 2016·2017년 유효 가중치: 0.0998664 + 0.0434202 = 0.1432866",
            "- 2019년 가족친화인증기업 비율은 raking 보간값을 원래 AHP 가중치 0.0434202로 사용",
            "- 비교 대안: PR #89에서 복원·추정한 2016·2017 값을 모두 사용하는 28개 고정 AHP 계열",
            "- pooled 표준화에서는 2016·2017 가족친화 값을 기준집단에서도 제외하므로 2018년 이후 가족친화 점수 척도도 대안계열과 달라질 수 있음",
            f"- 영향을 받는 2016·2017년 평균 절대 점수 차이: {family_friendly_comparison.loc[family_friendly_comparison['year'].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS), 'abs_score_diff'].mean():.4f}점",
            f"- 영향을 받는 2016·2017년 최대 절대 점수 차이: {family_friendly_comparison.loc[family_friendly_comparison['year'].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS), 'abs_score_diff'].max():.4f}점",
            f"- 영향을 받는 2016·2017년 평균 절대 순위 차이: {family_friendly_comparison.loc[family_friendly_comparison['year'].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS), 'abs_rank_diff'].mean():.4f}단계",
            f"- 영향을 받는 2016·2017년 최대 절대 순위 차이: {int(family_friendly_comparison.loc[family_friendly_comparison['year'].isin(FAMILY_FRIENDLY_EXCLUDED_YEARS), 'abs_rank_diff'].max())}단계",
            f"- 전체 기간 평균 절대 점수 차이: {family_friendly_comparison['abs_score_diff'].mean():.4f}점",
            f"- 전체 기간 최대 절대 점수 차이: {family_friendly_comparison['abs_score_diff'].max():.4f}점",
            "",
            "## 비용지표 CPI 실질화 영향",
            "",
            f"- 실질화 대상 ID: {', '.join(sorted(REAL_COST_INDICATOR_IDS))}",
            f"- 계산식: 실질금액({cpi_base_year} 가격) = 명목금액 × {cpi_base_value:g} / 해당 연도 CPI",
            f"- pooled 최종지수 평균 절대 변화: "
            f"{(results['pooled'].final_index['final_index'] - nominal_results['pooled'].final_index['final_index']).abs().mean():.4f}점",
            f"- pooled 최종지수 최대 절대 변화: "
            f"{(results['pooled'].final_index['final_index'] - nominal_results['pooled'].final_index['final_index']).abs().max():.4f}점",
            "- 전국 공통 물가상승분만 제거해 지역 간 명목가격 격차를 유지하고 연도 간 화폐가치를 통일",
            "- 시도별 CPI는 각 지역의 물가 변화율 지수로 지역 간 절대 물가수준을 나타내지 않으므로 본계열 디플레이터로 사용하지 않음",
            f"- {cpi_base_year}은 사용한 CPI의 기준연도이며, 기준연도 변경은 실질금액의 표시 단위만 바꾸고 동일한 선형 표준화 결과에는 영향을 주지 않음",
            "",
            "## pooled–yearly 민감도 비교",
            "",
            f"- 연도 내 Pearson 상관계수 범위: {yearly_stats['pearson'].min():.4f}–{yearly_stats['pearson'].max():.4f}",
            f"- 평균 절대 순위 차이: {comparison['abs_rank_diff'].mean():.4f}단계",
            f"- 동일 순위 비율: {comparison['abs_rank_diff'].eq(0).mean() * 100:.1f}%",
            f"- 2단계 이내 순위 차이 비율: {comparison['abs_rank_diff'].le(2).mean() * 100:.1f}%",
            f"- 최대 순위 차이: {int(comparison['abs_rank_diff'].max())}단계",
            "",
            "연도별 Min-Max는 매년 최소·최대를 0·100으로 다시 맞추므로 연도 간 절대 수준 비교에는 적합하지 않다. "
            "따라서 **pooled를 본계열**, yearly를 연도 내 순위의 민감도 분석으로 사용한다.",
            "전체 연도를 한꺼번에 계산한 두 점수의 상관계수는 시간축 재척도화의 영향을 받으므로 판단 근거로 사용하지 않는다.",
            "",
            "### 순위 차이가 큰 지역·연도",
            "",
            "| 지역 | 연도 | pooled 순위 | yearly 순위 | 절대 차이 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.nlargest(5, "abs_rank_diff").itertuples(index=False):
        lines.append(
            f"| {row.region} | {row.year} | {int(row.rank_pooled)} | "
            f"{int(row.rank_yearly)} | {int(row.abs_rank_diff)} |"
        )
    lines.extend(
        [
            "",
            "### 연도별 비교",
            "",
            "| 연도 | Pearson 상관 | 점수 MAE | 평균 절대 순위 차이 | 최대 순위 차이 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for year, row in yearly_stats.iterrows():
        lines.append(
            f"| {year} | {row['pearson']:.4f} | {row['score_mae']:.4f} | "
            f"{row['mean_abs_rank_diff']:.4f} | {int(row['max_abs_rank_diff'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cpi", type=Path, default=DEFAULT_CPI)
    return parser.parse_args()


def main() -> None:
    np.random.seed(42)
    args = parse_args()
    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    panel = prepare_processed_structural_panel(raw)
    cpi = pd.read_csv(args.cpi, encoding="utf-8-sig")
    manifest = load_structural_indicator_manifest(REPO_ROOT)
    weights = load_structural_index_weights(REPO_ROOT)
    validate_structural_index_weights(weights, manifest)
    nominal_results = run_family_friendly_weight_transfer_scenarios(panel, weights)
    real_panel = deflate_structural_cost_indicators(panel, cpi)
    cpi_base_year = str(real_panel.attrs["cpi_base_year"])
    cpi_base_value = float(real_panel.attrs["cpi_base_value"])
    raking_results = run_structural_index_scenarios(real_panel, weights)
    results = run_family_friendly_weight_transfer_scenarios(real_panel, weights)
    comparison = compare_scenarios(results)
    family_friendly_comparison = compare_family_friendly_decision(results, raking_results)

    expected_indicator_rows = (
        len(DEFAULT_STRUCTURAL_REGIONS) * len(DEFAULT_STRUCTURAL_YEARS) * len(weights)
    )
    expected_final_rows = len(DEFAULT_STRUCTURAL_REGIONS) * len(DEFAULT_STRUCTURAL_YEARS)
    for method, result in results.items():
        if len(result.indicator_scores) != expected_indicator_rows:
            raise ValueError(f"{method}: 지표 점수 행 수 불일치")
        if len(result.final_index) != expected_final_rows:
            raise ValueError(f"{method}: 최종지수 행 수 불일치")
        if result.final_index["final_index"].isna().any():
            raise ValueError(f"{method}: 최종지수 결측 발생")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for method, result in results.items():
        for name in ("indicator_scores", "subcategory_scores", "category_scores", "final_index"):
            frame = getattr(result, name)
            frame.to_csv(
                args.output_dir / f"structural_index_{method}_{name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    comparison.to_csv(
        args.output_dir / "structural_index_pooled_yearly_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raking_results["pooled"].final_index.to_csv(
        args.output_dir / "structural_index_family_friendly_raking_pooled_final_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family_friendly_comparison.to_csv(
        args.output_dir / "structural_index_family_friendly_weight_transfer_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    args.report.write_text(
        build_report(
            results,
            comparison,
            nominal_results,
            family_friendly_comparison,
            len(raw),
            cpi_base_year,
            cpi_base_value,
        ),
        encoding="utf-8",
    )
    print(f"구조환경지수 산출 완료: {args.output_dir}")
    print(f"QA 보고서: {args.report}")


if __name__ == "__main__":
    main()
