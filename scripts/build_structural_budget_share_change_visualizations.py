"""#108 구조환경 변화–후행 예산비중 변화 대응성 결과를 시각화한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.build_structural_budget_share_change_sample import (
    BUDGET_SHARE_CHANGE,
    STRUCTURAL_CHANGE,
)
from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_SAMPLE = ANALYSIS_DIR / "2016-2024_구조환경변화_후행예산비중변화_표본.csv"
DEFAULT_RESULTS = ANALYSIS_DIR / "2016-2024_세부영역별_구조환경변화_예산비중대응_결과.csv"
DEFAULT_REGIONS = ANALYSIS_DIR / "2016-2024_시도별_구조환경변화_예산비중대응_기술통계.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/figures/change_responsiveness"


def _two_way_residual(data: pd.DataFrame, value: str) -> pd.Series:
    """균형패널 값에서 시도·기준연도 평균을 제거한다."""
    if data.duplicated(["지역", "기준연도"]).any():
        raise ValueError("고정효과 제거 대상 지역×기준연도 키가 중복됩니다.")
    counts = data.groupby("지역")["기준연도"].nunique()
    if counts.nunique() != 1:
        raise ValueError("고정효과 제거에는 균형패널이 필요합니다.")
    return (
        data[value]
        - data.groupby("지역")[value].transform("mean")
        - data.groupby("기준연도")[value].transform("mean")
        + data[value].mean()
    )


def plot_fe_relationships(sample: pd.DataFrame, results: pd.DataFrame) -> plt.Figure:
    """11개 영역의 고정효과 제거 변화량 관계를 소패널로 그린다."""
    order = results.sort_values("계수")["모형"].tolist()
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    for ax, subarea in zip(axes.flat, order, strict=False):
        group = sample.loc[sample["세부영역"].eq(subarea)].copy()
        group["x"] = _two_way_residual(group, STRUCTURAL_CHANGE)
        group["y"] = _two_way_residual(group, BUDGET_SHARE_CHANGE)
        row = results.loc[results["모형"].eq(subarea)].iloc[0]
        ax.scatter(group["x"], group["y"], color="#94A3B8", alpha=0.55, s=18)
        x_line = np.linspace(group["x"].min(), group["x"].max(), 100)
        ax.plot(x_line, row["계수"] * x_line, color=YOMOCHA_WEB_COLORS["accent"], linewidth=1.8)
        ax.axhline(0, color="#D1D5DB", linewidth=0.8)
        ax.axvline(0, color="#D1D5DB", linewidth=0.8)
        ax.set_title(f"{subarea}\nβ={row['계수']:+.3f}, q={row['FDR_q값']:.3f}")
        ax.grid(False)
    for ax in axes.flat[len(order) :]:
        ax.set_visible(False)
    fig.suptitle(
        "구조환경 변화와 후행 계획예산 비중 변화의 고정효과 제거 관계", y=0.995, fontsize=17
    )
    fig.text(
        0.5,
        0.963,
        "구조환경 t→t+1, 예산비중 t+2→t+3·17개 시도×6개 기준연도·음의 기울기가 재정대응 가설 방향",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("구조환경지수 변화: 시도·기준연도 고정효과 제거값")
    fig.supylabel("계획예산 비중 변화(%p): 시도·기준연도 고정효과 제거값")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    return fig


def plot_coefficient_forest(results: pd.DataFrame) -> plt.Figure:
    """영역별 변화량 대응계수와 95% 신뢰구간을 포리스트 플롯으로 그린다."""
    plot = results.sort_values("계수", ascending=False).reset_index(drop=True)
    y = np.arange(len(plot))
    colors = np.where(plot["계수"].lt(0), YOMOCHA_WEB_COLORS["accent"], "#94A3B8")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for position, (_, row), color in zip(y, plot.iterrows(), colors, strict=True):
        ax.errorbar(
            row["계수"],
            position,
            xerr=[
                [row["계수"] - row["95%신뢰구간_하한"]],
                [row["95%신뢰구간_상한"] - row["계수"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
    ax.axvline(0, color="#4B5563", linewidth=1)
    ax.set_yticks(y, plot["모형"])
    ax.invert_yaxis()
    ax.set_xlabel("구조환경지수 1점 변화당 후행 계획예산 비중 변화(%p)")
    ax.set_ylabel("세부영역")
    ax.grid(axis="y", visible=False)
    fig.suptitle("세부영역별 구조환경 변화–후행 예산비중 변화 대응계수", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "점=계수·선=95% 신뢰구간·파란색=재정대응 가설 방향(음수)·BH 보정 후 유의한 영역 없음",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def plot_region_subarea_alignment(sample: pd.DataFrame) -> plt.Figure:
    """시도×영역별 두 변화가 반대 방향인 관측치 비율을 히트맵으로 그린다."""
    usable = sample.loc[sample[STRUCTURAL_CHANGE].ne(0) & sample[BUDGET_SHARE_CHANGE].ne(0)].copy()
    usable["반대방향"] = (usable[STRUCTURAL_CHANGE] * usable[BUDGET_SHARE_CHANGE]).lt(0)
    summary = usable.groupby(["지역", "세부영역"])["반대방향"].mean().mul(100).reset_index()
    matrix = summary.pivot(index="지역", columns="세부영역", values="반대방향")
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(17, 9))
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "반대방향 변화 비율(%)"},
        ax=ax,
    )
    ax.set_xlabel("세부영역")
    ax.set_ylabel("시도(전체 영역 평균 비율이 높은 순)")
    ax.tick_params(axis="x", rotation=45)
    fig.suptitle(
        "시도·세부영역별 구조환경과 후행 예산비중의 반대방향 변화 비율", y=0.99, fontsize=16
    )
    fig.text(
        0.5,
        0.95,
        "각 셀은 최대 6개 시점의 기술통계·100%는 환경과 예산비중이 매번 반대 방향으로 변했음을 뜻함",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def plot_region_summary(regions: pd.DataFrame) -> plt.Figure:
    """시도별 변화량 Spearman 상관을 탐색 순위로 그린다."""
    plot = regions.sort_values("Spearman_rho", ascending=False)
    colors = np.where(plot["Spearman_rho"].lt(0), YOMOCHA_WEB_COLORS["accent"], "#94A3B8")
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot["지역"], plot["Spearman_rho"], color=colors)
    ax.axvline(0, color="#4B5563", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.2f}" for value in plot["Spearman_rho"]], padding=3)
    ax.set_xlabel("구조환경 변화와 후행 예산비중 변화의 Spearman ρ")
    ax.set_ylabel("시도")
    ax.grid(axis="y", visible=False)
    fig.suptitle("시도별 구조환경 변화–후행 예산비중 변화의 기술적 대응성", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.95,
        "시도당 66개 관측치(6개 시점×11개 영역)·음수일수록 재정대응 가설 방향·인과효과 및 성과순위가 아님",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    results = pd.read_csv(args.results)
    regions = pd.read_csv(args.regions)
    figures = {
        "2016-2024_구조환경변화_후행예산비중변화_FE관계": plot_fe_relationships(sample, results),
        "2016-2024_세부영역별_구조환경변화_예산비중대응_계수": plot_coefficient_forest(results),
        "2016-2024_시도세부영역별_반대방향변화비율": plot_region_subarea_alignment(sample),
        "2016-2024_시도별_구조환경변화_예산비중대응_기술상관": plot_region_summary(regions),
    }
    for name, figure in figures.items():
        save_figure(figure, args.output_dir / name)
        plt.close(figure)


if __name__ == "__main__":
    main()
