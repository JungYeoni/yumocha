"""#110 구조환경 군집별 전체예산–후행 TFR 결과를 시각화한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_SAMPLE = ANALYSIS_DIR / "2016-2024_구조환경군집별_전체3개년평균예산_TFR_회귀표본.csv"
DEFAULT_COEFFICIENTS = ANALYSIS_DIR / "2016-2024_구조환경군집_전체예산상호작용_TFR_계수.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/figures/cluster_total_fiscal_tfr"
COLORS = {1: YOMOCHA_WEB_COLORS["accent"], 2: "#7C3AED", 3: "#0E7490"}


def validate_inputs(sample: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    """회귀표본과 계수표의 군집·시차 구조를 검증한다."""
    sample_required = {
        "지역",
        "연도",
        "전체_인구1인당_실질예산_3개년평균",
        "합계출산율_t+1",
        "합계출산율_t+2",
        "군집_2개",
        "군집_3개",
    }
    coefficient_required = {
        "군집수",
        "시차",
        "군집",
        "계수",
        "95%신뢰구간_하한",
        "95%신뢰구간_상한",
    }
    if missing := sample_required - set(sample.columns):
        raise ValueError(f"전체예산 회귀표본 필수 컬럼 누락: {sorted(missing)}")
    if missing := coefficient_required - set(coefficients.columns):
        raise ValueError(f"상호작용 계수표 필수 컬럼 누락: {sorted(missing)}")
    expected = {(2, lag, cluster) for lag in ("t+1", "t+2") for cluster in (1, 2)}
    expected |= {(3, lag, cluster) for lag in ("t+1", "t+2") for cluster in (1, 2, 3)}
    observed = set(coefficients[["군집수", "시차", "군집"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("k=2·k=3의 t+1·t+2 군집별 계수가 모두 필요합니다.")


def build_cluster_trends(sample: pd.DataFrame) -> pd.DataFrame:
    """k=2·k=3 군집별 연도 평균 전체예산과 후행 TFR을 구성한다."""
    usable = sample.dropna(subset=["전체_인구1인당_실질예산_3개년평균"]).copy()
    tables = []
    for cluster_count in (2, 3):
        cluster_column = f"군집_{cluster_count}개"
        table = (
            usable.groupby([cluster_column, "연도"], as_index=False)
            .agg(
                전체_실질1인당_3개년평균예산=(
                    "전체_인구1인당_실질예산_3개년평균",
                    "mean",
                ),
                합계출산율_t1=("합계출산율_t+1", "mean"),
                합계출산율_t2=("합계출산율_t+2", "mean"),
                지역수=("지역", "nunique"),
            )
            .rename(columns={cluster_column: "군집"})
        )
        table.insert(0, "군집수", cluster_count)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_coefficients(coefficients: pd.DataFrame) -> plt.Figure:
    """k=2·k=3 상호작용 모형의 군집별 계수와 신뢰구간을 비교한다."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    for row_index, cluster_count in enumerate((2, 3)):
        for column_index, lag in enumerate(("t+1", "t+2")):
            ax = axes[row_index, column_index]
            plot = coefficients.loc[
                coefficients["군집수"].eq(cluster_count) & coefficients["시차"].eq(lag)
            ].sort_values("군집")
            for _, row in plot.iterrows():
                cluster = int(row["군집"])
                ax.errorbar(
                    row["계수"],
                    cluster,
                    xerr=np.array(
                        [
                            [row["계수"] - row["95%신뢰구간_하한"]],
                            [row["95%신뢰구간_상한"] - row["계수"]],
                        ]
                    ),
                    fmt="o",
                    color=COLORS[cluster],
                    ecolor=COLORS[cluster],
                    capsize=4,
                    markersize=7,
                )
            ax.axvline(0, color=YOMOCHA_WEB_COLORS["line"], linewidth=1)
            ax.set_yticks(
                range(1, cluster_count + 1), [f"군집 {i}" for i in range(1, cluster_count + 1)]
            )
            ax.set_title(f"k={cluster_count} · {lag} TFR")
            ax.set_xlabel("log1p(전체 실질 1인당 3개년 평균예산) 계수")
            ax.grid(axis="y", visible=False)
    fig.suptitle("구조환경 군집별 전체 재정대응예산과 후행 TFR 계수", y=0.995, fontsize=16)
    fig.text(
        0.5,
        0.955,
        "전체 17개 시도 상호작용 모형의 점추정치와 95% 신뢰구간 · 인과효과가 아닌 조건부 관련성",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_trends(trends: pd.DataFrame) -> plt.Figure:
    """군집별 전체예산과 t+1·t+2 TFR 평균 추이를 작은 배수로 제시한다."""
    metrics = [
        ("전체_실질1인당_3개년평균예산", "전체 실질 1인당 3개년 평균예산", "원"),
        ("합계출산율_t1", "1년 후 TFR", "명"),
        ("합계출산율_t2", "2년 후 TFR", "명"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row_index, cluster_count in enumerate((2, 3)):
        subset = trends.loc[trends["군집수"].eq(cluster_count)]
        for column_index, (column, title, unit) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for cluster in range(1, cluster_count + 1):
                group = subset.loc[subset["군집"].eq(cluster)]
                ax.plot(
                    group["연도"],
                    group[column],
                    marker="o",
                    linewidth=2,
                    color=COLORS[cluster],
                    label=f"군집 {cluster}",
                )
            ax.set_title(f"k={cluster_count} · {title}")
            ax.set_ylabel(unit)
            ax.set_xlabel("3개년 예산창 마지막 연도(t)")
            if row_index == 0 and column_index == 0:
                ax.legend(loc="upper left", frameon=False)
            if row_index == 1 and column_index == 0:
                ax.legend(loc="upper left", frameon=False)
    fig.suptitle("구조환경 군집별 전체 재정대응예산과 후행 TFR 평균 추이", y=0.995, fontsize=16)
    fig.text(
        0.5,
        0.955,
        "군집별 기술통계 · 선의 동행만으로 예산 효과 또는 군집 차이를 판단하지 않음",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    coefficients = pd.read_csv(args.coefficients)
    validate_inputs(sample, coefficients)
    trends = build_cluster_trends(sample)
    figures = {
        "2016-2024_구조환경군집별_전체예산_TFR_계수": plot_coefficients(coefficients),
        "2018-2024_구조환경군집별_전체예산_TFR_평균추이": plot_trends(trends),
    }
    for name, figure in figures.items():
        save_figure(figure, args.output_dir / name)
        plt.close(figure)
    trends.to_csv(
        ANALYSIS_DIR / "2018-2024_구조환경군집별_전체예산_TFR_평균추이.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
