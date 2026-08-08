"""#108 구조환경 변화와 후행 예산비중 변화의 영역별 관련성을 추정한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from scripts.build_structural_budget_share_change_sample import (
    BUDGET_SHARE_CHANGE,
    STRUCTURAL_CHANGE,
)
from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_구조환경변화_후행예산비중변화_표본.csv"
)
DEFAULT_RESULTS = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_구조환경변화_예산비중대응_결과.csv"
)
DEFAULT_REGIONS = (
    REPO_ROOT / "data/processed/analysis/2016-2024_시도별_구조환경변화_예산비중대응_기술통계.csv"
)


def run_subarea_change_models(sample: pd.DataFrame) -> pd.DataFrame:
    """11개 세부영역별 이원 고정효과 변화량 모형과 단순 상관을 산출한다."""
    required = {"지역", "기준연도", "세부영역", STRUCTURAL_CHANGE, BUDGET_SHARE_CHANGE}
    if missing := required - set(sample.columns):
        raise KeyError(f"변화량 회귀 필수 컬럼 누락: {sorted(missing)}")
    summaries: list[dict[str, object]] = []
    for subarea, group in sample.groupby("세부영역", sort=False):
        model, fitted = fit_two_way_fixed_effects(
            group,
            outcome=BUDGET_SHARE_CHANGE,
            predictor=STRUCTURAL_CHANGE,
            year_col="기준연도",
        )
        summary = summarize_fixed_effects(
            model,
            fitted,
            model_name=subarea,
            outcome=BUDGET_SHARE_CHANGE,
            predictor=STRUCTURAL_CHANGE,
            excluded_quality_rows=False,
            year_col="기준연도",
        )
        pearson_r, pearson_p = pearsonr(group[STRUCTURAL_CHANGE], group[BUDGET_SHARE_CHANGE])
        spearman_rho, spearman_p = spearmanr(group[STRUCTURAL_CHANGE], group[BUDGET_SHARE_CHANGE])
        summary.update(
            {
                "단순_Pearson_r": pearson_r,
                "단순_Pearson_p값": pearson_p,
                "단순_Spearman_rho": spearman_rho,
                "단순_Spearman_p값": spearman_p,
                "음의계수_재정대응방향": summary["계수"] < 0,
            }
        )
        summaries.append(summary)
    result = pd.DataFrame(summaries)
    rejected, adjusted, _, _ = multipletests(result["p값"], alpha=0.05, method="fdr_bh")
    result["FDR_q값"] = adjusted
    result["FDR_0.05_유의"] = rejected
    return result.sort_values("계수").reset_index(drop=True)


def build_region_descriptive_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """시도별 변화량 상관과 반대방향 변화 비율을 탐색 지표로 요약한다."""
    required = {"지역", "세부영역", STRUCTURAL_CHANGE, BUDGET_SHARE_CHANGE}
    if missing := required - set(sample.columns):
        raise KeyError(f"시도별 기술통계 필수 컬럼 누락: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for region, group in sample.groupby("지역", sort=True):
        pearson_r, pearson_p = pearsonr(group[STRUCTURAL_CHANGE], group[BUDGET_SHARE_CHANGE])
        spearman_rho, spearman_p = spearmanr(group[STRUCTURAL_CHANGE], group[BUDGET_SHARE_CHANGE])
        nonzero = group.loc[
            group[STRUCTURAL_CHANGE].ne(0) & group[BUDGET_SHARE_CHANGE].ne(0)
        ].copy()
        opposite = (nonzero[STRUCTURAL_CHANGE] * nonzero[BUDGET_SHARE_CHANGE]).lt(0)
        rows.append(
            {
                "지역": region,
                "관측치": len(group),
                "비영변화_관측치": len(nonzero),
                "Pearson_r": pearson_r,
                "Pearson_p값": pearson_p,
                "Spearman_rho": spearman_rho,
                "Spearman_p값": spearman_p,
                "반대방향_변화비율_pct": opposite.mean() * 100,
            }
        )
    result = pd.DataFrame(rows).sort_values("Spearman_rho").reset_index(drop=True)
    result["대응성_기술순위"] = range(1, len(result) + 1)
    result["대응성_탐색집단"] = pd.qcut(
        result["대응성_기술순위"], q=3, labels=["상대적으로 높음", "중간", "상대적으로 낮음"]
    ).astype(str)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    results = run_subarea_change_models(sample)
    regions = build_region_descriptive_summary(sample)
    for path, frame in ((args.results, results), (args.regions, regions)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path} ({len(frame)}행)")
    print(
        results[["모형", "계수", "p값", "FDR_q값", "FDR_0.05_유의"]].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
