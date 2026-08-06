"""#70 결측처리 완료 패널로 #82 구조환경지수 두 시나리오를 산출한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.structural_index import (
    DEFAULT_STRUCTURAL_REGIONS,
    DEFAULT_STRUCTURAL_YEARS,
    REAL_COST_INDICATOR_IDS,
    deflate_structural_cost_indicators,
    load_structural_index_weights,
    load_structural_indicator_manifest,
    prepare_processed_structural_panel,
    run_structural_index_scenarios,
    validate_structural_index_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/구조환경지표_28개_결측처리후_본계열패널.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/structural_index"
DEFAULT_REPORT = REPO_ROOT / "reports/methodology/20260806_구조환경지수_실제패널_산출_QA.md"
DEFAULT_CPI = REPO_ROOT / "data/lookup/연도별_소비자물가지수.csv"


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


def build_report(
    results: dict,
    comparison: pd.DataFrame,
    nominal_results: dict,
    input_rows: int,
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
        "- 실질화: 주택가격·임차가구 연간 주거비·사교육비·산후조리원 이용요금에 전국 연평균 CPI(2020=100) 적용",
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
            "## 비용지표 CPI 실질화 영향",
            "",
            f"- 실질화 대상 ID: {', '.join(sorted(REAL_COST_INDICATOR_IDS))}",
            "- 계산식: 실질금액(2020년 가격) = 명목금액 × 100 / 해당 연도 CPI",
            f"- pooled 최종지수 평균 절대 변화: "
            f"{(results['pooled'].final_index['final_index'] - nominal_results['pooled'].final_index['final_index']).abs().mean():.4f}점",
            f"- pooled 최종지수 최대 절대 변화: "
            f"{(results['pooled'].final_index['final_index'] - nominal_results['pooled'].final_index['final_index']).abs().max():.4f}점",
            "- 전국 공통 물가상승분만 제거해 지역 간 명목가격 격차를 유지하고 연도 간 화폐가치를 통일",
            "- 시도별 CPI는 각 지역의 물가 변화율 지수로 지역 간 절대 물가수준을 나타내지 않으므로 본계열 디플레이터로 사용하지 않음",
            "- 2020년은 사용한 CPI의 기준연도이며, 기준연도 변경은 실질금액의 표시 단위만 바꾸고 동일한 선형 표준화 결과에는 영향을 주지 않음",
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
    args = parse_args()
    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    panel = prepare_processed_structural_panel(raw)
    cpi = pd.read_csv(args.cpi, encoding="utf-8-sig")
    manifest = load_structural_indicator_manifest(REPO_ROOT)
    weights = load_structural_index_weights(REPO_ROOT)
    validate_structural_index_weights(weights, manifest)
    nominal_results = run_structural_index_scenarios(panel, weights)
    real_panel = deflate_structural_cost_indicators(panel, cpi)
    results = run_structural_index_scenarios(real_panel, weights)
    comparison = compare_scenarios(results)

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
    args.report.write_text(
        build_report(results, comparison, nominal_results, len(raw)), encoding="utf-8"
    )
    print(f"구조환경지수 산출 완료: {args.output_dir}")
    print(f"QA 보고서: {args.report}")


if __name__ == "__main__":
    main()
