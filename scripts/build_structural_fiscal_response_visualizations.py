"""#102 구조환경 → 재정대응 반응성 결과를 시각화한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_구조환경_재정대응_공통반응계수.csv"
)
DEFAULT_COMMON_T1 = DEFAULT_COMMON.with_name(f"{DEFAULT_COMMON.stem}_t+1.csv")
DEFAULT_REGIONAL = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_시도별_구조환경_재정대응_반응계수.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/구조환경_재정대응_반응성"


def plot_lag_comparison(common_t1: pd.DataFrame, common_t2: pd.DataFrame) -> plt.Figure:
    """t+1과 t+2 공통 반응계수를 같은 축에서 비교한다."""
    required = {"모형", "계수", "95%신뢰구간_하한", "95%신뢰구간_상한"}
    for frame, label in ((common_t1, "t+1"), (common_t2, "t+2")):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{label} 시차 비교 그래프 필수 컬럼 누락: {missing}")

    order = common_t2.sort_values("계수")["모형"].tolist()
    t1 = common_t1.set_index("모형").loc[order]
    t2 = common_t2.set_index("모형").loc[order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for frame, offset, label, color in (
        (t1, -0.12, "t+1", "#0E7490"),
        (t2, 0.12, "t+2", YOMOCHA_WEB_COLORS["accent"]),
    ):
        lower = frame["계수"] - frame["95%신뢰구간_하한"]
        upper = frame["95%신뢰구간_상한"] - frame["계수"]
        ax.errorbar(
            frame["계수"],
            y + offset,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=2.5,
            markersize=5,
            alpha=0.9,
            label=label,
        )
    ax.axvline(0, color="#374151", linewidth=1, linestyle="--")
    ax.set_yticks(y, order)
    ax.set_xlabel("구조환경지수 1점 상승에 대한 log1p(실질 1인당 예산) 계수")
    ax.set_ylabel("세부영역")
    ax.legend(frameon=False)
    fig.suptitle("구조환경과 1년·2년 후 재정대응예산 반응계수 비교", y=0.98)
    fig.text(
        0.5,
        0.945,
        "점은 계수, 선은 95% 신뢰구간·동일한 시도/연도 고정효과 모형",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_common_coefficients(common: pd.DataFrame) -> plt.Figure:
    """세부영역별 공통 반응계수와 95% 신뢰구간을 그린다."""
    required = {"모형", "계수", "95%신뢰구간_하한", "95%신뢰구간_상한", "FDR_0.05_유의"}
    missing = sorted(required - set(common.columns))
    if missing:
        raise KeyError(f"공통계수 그래프 필수 컬럼 누락: {missing}")

    plot = common.sort_values("계수").reset_index(drop=True)
    y = np.arange(len(plot))
    lower = plot["계수"] - plot["95%신뢰구간_하한"]
    upper = plot["95%신뢰구간_상한"] - plot["계수"]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.errorbar(
        plot["계수"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=YOMOCHA_WEB_COLORS["accent"],
        ecolor=YOMOCHA_WEB_COLORS["line"],
        elinewidth=1.6,
        capsize=3,
        markersize=6,
    )
    ax.axvline(0, color="#374151", linewidth=1, linestyle="--")
    ax.set_yticks(y, plot["모형"])
    ax.set_xlabel("구조환경지수 1점 상승에 대한 log1p(실질 1인당 예산) 계수")
    ax.set_ylabel("세부영역")
    fig.suptitle("구조환경과 2년 후 재정대응예산의 공통 반응계수", y=0.98)
    fig.text(
        0.5,
        0.945,
        "점은 계수, 선은 95% 신뢰구간·시도/연도 고정효과·시도 군집표준오차",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_regional_coefficient_heatmap(regional: pd.DataFrame) -> plt.Figure:
    """지역×세부영역 탐색적 반응계수 행렬을 그린다."""
    required = {"세부영역", "지역", "지역별_반응계수", "추정가능"}
    missing = sorted(required - set(regional.columns))
    if missing:
        raise KeyError(f"지역별 히트맵 필수 컬럼 누락: {missing}")

    matrix = regional.pivot(index="지역", columns="세부영역", values="지역별_반응계수")
    matrix = matrix.reindex(sorted(matrix.index)).reindex(sorted(matrix.columns), axis=1)
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if len(finite) else 1.0
    limit = max(limit, 1e-6)
    cmap = LinearSegmentedColormap.from_list(
        "yumocha_signed", ["#0E7490", YOMOCHA_WEB_COLORS["surface_neutral"], "#7C3AED"]
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, ax = plt.subplots(figsize=(15, 9.5))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xlabel("세부영역")
    ax.set_ylabel("시도")
    fig.suptitle("시도별 구조환경·재정대응 반응계수(탐색 분석)", y=0.98)
    fig.text(
        0.5,
        0.945,
        "청록=음의 계수, 보라=양의 계수·회색 X=설계행렬 선형종속으로 지역별 계수 식별 불가",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    missing_cells = np.argwhere(~np.isfinite(values))
    for row, column in missing_cells:
        ax.text(column, row, "×", ha="center", va="center", color="#6B7280", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("log1p(실질 1인당 예산) 반응계수")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--common-t1", type=Path, default=DEFAULT_COMMON_T1)
    parser.add_argument("--regional", type=Path, default=DEFAULT_REGIONAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.common.is_file() or not args.common_t1.is_file() or not args.regional.is_file():
        raise FileNotFoundError("공통 또는 시도별 반응계수 산출물이 없습니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = pd.read_csv(args.common)
    common_t1 = pd.read_csv(args.common_t1)
    regional = pd.read_csv(args.regional)
    comparison = common_t1[["모형", "계수", "p값", "FDR_q값", "FDR_0.05_유의"]].merge(
        common[["모형", "계수", "p값", "FDR_q값", "FDR_0.05_유의"]],
        on="모형",
        validate="one_to_one",
        suffixes=("_t+1", "_t+2"),
    )
    comparison["계수_부호일치"] = np.sign(comparison["계수_t+1"]) == np.sign(comparison["계수_t+2"])
    comparison.to_csv(
        args.output_dir / "t+1_t+2_공통_반응계수_비교.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common_figure = plot_common_coefficients(common)
    regional_figure = plot_regional_coefficient_heatmap(regional)
    comparison_figure = plot_lag_comparison(common_t1, common)
    save_figure(common_figure, args.output_dir / "세부영역별_공통_반응계수", formats=["png"])
    save_figure(regional_figure, args.output_dir / "시도별_반응계수_히트맵", formats=["png"])
    save_figure(comparison_figure, args.output_dir / "t+1_t+2_공통_반응계수_비교", formats=["png"])
    plt.close(common_figure)
    plt.close(regional_figure)
    plt.close(comparison_figure)
    print(f"저장: {args.output_dir}")


if __name__ == "__main__":
    main()
