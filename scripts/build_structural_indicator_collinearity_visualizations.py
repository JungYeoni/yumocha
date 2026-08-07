"""#62 구조환경지수 다중공선성 진단(pooled/within/고정효과 잔차 상관, VIF)을 시각화한다.

계산 로직은 ``src/modeling/collinearity.py``와 이미 실행된
``scripts/diagnose_structural_indicator_collinearity.py``와 동일하다. 이 스크립트는
같은 계산을 다시 돌려(#82/#96 pooled 구조환경지수를 입력으로) 히트맵·막대그래프
이미지를 만드는 시각화 전용 얇은 레이어다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.modeling.collinearity import (
    compute_pooled_correlation,
    compute_two_way_fe_residuals,
    compute_vif,
    compute_within_region_correlation,
    pivot_scores_to_wide,
)
from src.visualization.plots import PALETTE, plot_precomputed_correlation_heatmap, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURAL_INDEX_DIR = REPO_ROOT / "data" / "processed" / "structural_index"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "구조환경지수_다중공선성_진단"
VIF_WARNING_THRESHOLD = 5.0
VIF_SEVERE_THRESHOLD = 10.0


def _load_level(
    structural_index_dir: Path, *, filename: str, group_col: str, value_col: str
) -> tuple[pd.DataFrame, list[str]]:
    long_scores = pd.read_csv(structural_index_dir / filename).rename(
        columns={"region": "지역", "year": "연도"}
    )
    wide = pivot_scores_to_wide(long_scores, group_col=group_col, value_col=value_col)
    columns = sorted(set(long_scores[group_col]))
    return wide, columns


def plot_vif_bars(vif_table: pd.DataFrame, *, title: str) -> plt.Figure:
    """VIF 막대그래프 — 5·10 기준선 표시."""
    ordered = vif_table.sort_values("VIF", ascending=True)
    colors = [
        PALETTE[3]
        if value >= VIF_SEVERE_THRESHOLD
        else PALETTE[4]
        if value >= VIF_WARNING_THRESHOLD
        else PALETTE[0]
        for value in ordered["VIF"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(3, len(ordered) * 0.45)))
    ax.barh(ordered["변수"], ordered["VIF"], color=colors)
    ax.axvline(VIF_WARNING_THRESHOLD, color="#9ca3af", linestyle="--", linewidth=1, label="주의(5)")
    ax.axvline(VIF_SEVERE_THRESHOLD, color="#c0392b", linestyle="--", linewidth=1, label="심함(10)")
    ax.set_xlabel("VIF")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


def build_level_visualizations(
    wide: pd.DataFrame, *, columns: list[str], level_label: str, output_dir: Path
) -> None:
    pooled_pearson, _ = compute_pooled_correlation(wide, columns=columns, method="pearson")
    within_pearson = compute_within_region_correlation(wide, columns=columns)
    fe_residuals = compute_two_way_fe_residuals(wide, columns=columns)
    fe_residual_pearson = fe_residuals.corr(method="pearson")
    vif_table, condition_number = compute_vif(wide, columns=columns)

    prefix = "대영역" if "대영역" in level_label else "세부영역"

    save_figure(
        plot_precomputed_correlation_heatmap(
            pooled_pearson, title=f"{level_label} pooled Pearson 상관"
        ),
        output_dir / f"{prefix}_pooled_상관",
        formats=["png"],
    )
    save_figure(
        plot_precomputed_correlation_heatmap(
            within_pearson, title=f"{level_label} within(지역 평균 제거) 상관"
        ),
        output_dir / f"{prefix}_within_상관",
        formats=["png"],
    )
    save_figure(
        plot_precomputed_correlation_heatmap(
            fe_residual_pearson,
            title=f"{level_label} 지역+연도 고정효과 잔차 상관",
        ),
        output_dir / f"{prefix}_고정효과잔차_상관",
        formats=["png"],
    )
    save_figure(
        plot_vif_bars(vif_table, title=f"{level_label} VIF (조건수={condition_number:.2f})"),
        output_dir / f"{prefix}_VIF",
        formats=["png"],
    )
    plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-index-dir", type=Path, default=DEFAULT_STRUCTURAL_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    category_wide, category_columns = _load_level(
        args.structural_index_dir,
        filename="structural_index_pooled_category_scores.csv",
        group_col="category",
        value_col="category_score",
    )
    subcategory_wide, subcategory_columns = _load_level(
        args.structural_index_dir,
        filename="structural_index_pooled_subcategory_scores.csv",
        group_col="subcategory",
        value_col="subcategory_score",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_level_visualizations(
        category_wide,
        columns=category_columns,
        level_label="대영역(4개)",
        output_dir=args.output_dir,
    )
    build_level_visualizations(
        subcategory_wide,
        columns=subcategory_columns,
        level_label="세부영역(11개)",
        output_dir=args.output_dir,
    )

    print(f"이미지 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
