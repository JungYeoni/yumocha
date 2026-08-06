"""AHP 구조환경지수 종합지수 시각화와 QA 보고서를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.structural_index import (
    OUTPUT_DIRNAME,
    build_index_summary,
    load_structural_index_artifacts,
    plot_annual_rank_heatmap,
    plot_family_friendly_impact,
    plot_pooled_index_trends,
    plot_pooled_yearly_comparison,
    plot_region_component_comparison,
    plot_region_contribution_heatmap,
    summarize_qa,
    validate_structural_index_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "result" / OUTPUT_DIRNAME
DEFAULT_REPORT = (
    REPO_ROOT / "reports" / "methodology" / "20260806_#91_AHP_구조환경지수_종합지수_시각화_QA.md"
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_visualization_outputs(
    *,
    repo_root: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, int]:
    """입력 산출물 검증 후 종합지수 그래프·요약·QA를 생성한다."""

    artifacts = load_structural_index_artifacts(repo_root)
    validation = validate_structural_index_artifacts(artifacts)
    summary = build_index_summary(artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_figure(
        plot_pooled_index_trends(summary), output_dir / "pooled_종합지수_17개시도_추세.png"
    )
    _save_figure(
        plot_annual_rank_heatmap(summary), output_dir / "pooled_종합지수_연도별_지역순위.png"
    )
    _save_figure(
        plot_pooled_yearly_comparison(summary), output_dir / "pooled_yearly_종합지수_비교.png"
    )
    _save_figure(
        plot_region_component_comparison(artifacts, region="제주"),
        output_dir / "제주_대영역_기여도_구성비교.png",
    )
    _save_figure(
        plot_region_contribution_heatmap(artifacts, region="제주"),
        output_dir / "제주_대영역_기여도_heatmap.png",
    )
    _save_figure(
        plot_family_friendly_impact(artifacts), output_dir / "가족친화인증기업_가중치이전_영향.png"
    )

    summary.to_csv(output_dir / "구조환경지수_지역연도_요약.csv", index=False, encoding="utf-8-sig")
    artifacts["pooled_category"].to_csv(
        output_dir / "구조환경지수_대영역_기여도.csv", index=False, encoding="utf-8-sig"
    )
    artifacts["pooled_subcategory"].to_csv(
        output_dir / "구조환경지수_세부영역_기여도.csv", index=False, encoding="utf-8-sig"
    )

    stats = summarize_qa(artifacts, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# #91 AHP 구조환경지수 종합지수 시각화 QA\n\n"
        "## 입력·검증\n\n"
        "- pooled 본계열: `data/processed/structural_index/structural_index_pooled_final_index.csv`\n"
        "- yearly 민감도: `data/processed/structural_index/structural_index_yearly_final_index.csv`\n"
        "- 구성 점수: pooled 대영역·세부영역·지표 점수 CSV\n"
        "- 민감도: pooled/yearly 비교 및 가족친화인증기업 처리 비교 CSV\n"
        f"- 최종지수 행 수: {validation['final_index_rows']:,}행 (17개 시도 × 2016–2024년)\n"
        f"- 대영역 점수 행 수: {validation['category_rows']:,}행, 세부영역 점수 행 수: {validation['subcategory_rows']:,}행\n"
        f"- 지표 점수 행 수: {validation['indicator_rows']:,}행\n"
        "- 전국 행: 최종지수·구성점수·순위 계산에서 제외됨을 검증\n"
        "- 결측 최종지수·비유한값·지역·연도 중복: 없음\n\n"
        "## 해석 원칙\n\n"
        "- pooled를 본계열로, yearly를 표준화 민감도 분석으로 표시한다.\n"
        "- AHP 가중치는 모든 연도에 고정 적용되며, 지수는 인과효과가 아닌 상대적 구조환경 수준이다.\n"
        "- 가족친화인증기업 2016·2017년은 가중치를 육아휴직 사용률로 이전한 처리와 raking 처리의 차이를 별도 비교한다.\n\n"
        "## 비교 결과\n\n"
        f"- 연도별 pooled/yearly Pearson 상관: {stats['yearly_correlation_min']:.4f}–{stats['yearly_correlation_max']:.4f}\n"
        f"- 평균 절대 순위 차이: {stats['mean_absolute_rank_gap']:.2f}단계\n"
        f"- 순위 차이 2단계 이내: {stats['rank_gap_within_two_pct']:.1f}%\n"
        f"- 최대 절대 순위 차이: {stats['max_absolute_rank_gap']:.0f}단계\n"
        f"- 2016·2017 가중치 이전과 raking의 평균 절대 점수 차이: {stats['family_transfer_2016_2017_mean_score_gap']:.4f}점\n\n"
        "## 산출물\n\n"
        "- `pooled_종합지수_17개시도_추세.png`: 17개 시도별 pooled 추세\n"
        "- `pooled_종합지수_연도별_지역순위.png`: 연도별 순위 heatmap\n"
        "- `pooled_yearly_종합지수_비교.png`: 본계열·민감도 비교\n"
        "- `제주_대영역_기여도_구성비교.png`: 지역 선택형 구성 비교 예시(제주)\n"
        "- `제주_대영역_기여도_heatmap.png`: 대영역별 기여도 크기 비교(제주)\n"
        "- `가족친화인증기업_가중치이전_영향.png`: 2016·2017 처리 영향\n"
        "- `구조환경지수_지역연도_요약.csv`, `구조환경지수_대영역_기여도.csv`, `구조환경지수_세부영역_기여도.csv`\n",
        encoding="utf-8",
    )
    return {"final_index_rows": validation["final_index_rows"], "summary_rows": len(summary)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build_visualization_outputs(output_dir=args.output_dir, report_path=args.report))
