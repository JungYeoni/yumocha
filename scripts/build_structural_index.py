"""#70 결측처리 완료 패널로 #82 구조환경지수 두 시나리오를 산출한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.structural_index import (
    DEFAULT_STRUCTURAL_REGIONS,
    DEFAULT_STRUCTURAL_YEARS,
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


def build_report(results: dict, input_rows: int) -> str:
    lines = [
        "# 구조환경지수 실제 패널 산출 QA",
        "",
        "## 입력과 범위",
        "",
        f"- #70 결측처리 완료 패널: {input_rows:,}행",
        "- 분석 격자: 17개 시도 × 2016–2024년 × 28개 지표 = 4,284행",
        "- 전국 행은 표준화·지수 산출에서 제외",
        "- 표준화 시나리오: pooled Min-Max, 연도별(yearly) Min-Max",
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
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    panel = prepare_processed_structural_panel(raw)
    manifest = load_structural_indicator_manifest(REPO_ROOT)
    weights = load_structural_index_weights(REPO_ROOT)
    validate_structural_index_weights(weights, manifest)
    results = run_structural_index_scenarios(panel, weights)

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
    args.report.write_text(build_report(results, len(raw)), encoding="utf-8")
    print(f"구조환경지수 산출 완료: {args.output_dir}")
    print(f"QA 보고서: {args.report}")


if __name__ == "__main__":
    main()
