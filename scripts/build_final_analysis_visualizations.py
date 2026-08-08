"""#107 구조환경·재정대응·TFR 최종 비교 시각화를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_STRUCTURAL = REPO_ROOT / "result/구조환경지수_시각화/구조환경지수_지역연도_요약.csv"
DEFAULT_FISCAL = ANALYSIS_DIR / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
DEFAULT_TFR = ANALYSIS_DIR / "2016-2024_재정대응지수_패널.csv"
DEFAULT_MOVING_RESULTS = ANALYSIS_DIR / "2016-2024_세부영역별_3개년평균예산_TFR_고정효과_결과.csv"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR
DEFAULT_FIGURE_DIR = REPO_ROOT / "reports/figures/final_analysis"

REAL_PC = "인구1인당_실질예산_원"
YEARS = set(range(2016, 2025))
REGION_COUNT = 17


def _within_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_structural_tfr_panel(structural: pd.DataFrame, tfr: pd.DataFrame) -> pd.DataFrame:
    """구조환경 종합지수와 TFR을 지역×연도로 결합하고 시도 내 표준화한다."""
    required_structural = {"region", "year", "pooled_index"}
    required_tfr = {"지역", "연도", "합계출산율"}
    if missing := required_structural - set(structural.columns):
        raise KeyError(f"구조환경 필수 컬럼 누락: {sorted(missing)}")
    if missing := required_tfr - set(tfr.columns):
        raise KeyError(f"TFR 필수 컬럼 누락: {sorted(missing)}")
    left = structural.rename(
        columns={"region": "지역", "year": "연도", "pooled_index": "구조환경종합지수"}
    )
    left = left[["지역", "연도", "구조환경종합지수"]]
    right = tfr[["지역", "연도", "합계출산율"]].drop_duplicates(["지역", "연도"])
    result = left.merge(right, on=["지역", "연도"], how="inner", validate="one_to_one")
    if len(result) != REGION_COUNT * len(YEARS):
        raise ValueError("구조환경–TFR 결합표는 17개 시도×9개년의 153행이어야 합니다.")
    result["구조환경_연도별순위"] = result.groupby("연도")["구조환경종합지수"].rank(
        ascending=False, method="min"
    )
    result["TFR_연도별순위"] = result.groupby("연도")["합계출산율"].rank(
        ascending=False, method="min"
    )
    result["구조환경_연도별순위점수"] = (
        (REGION_COUNT - result["구조환경_연도별순위"]) / (REGION_COUNT - 1) * 100
    )
    result["TFR_연도별순위점수"] = (
        (REGION_COUNT - result["TFR_연도별순위"]) / (REGION_COUNT - 1) * 100
    )
    result["순위차이_TFR_minus_구조환경"] = result["TFR_연도별순위"] - result["구조환경_연도별순위"]
    result["절대순위차이"] = result["순위차이_TFR_minus_구조환경"].abs()
    annual_spearman = result.groupby("연도").apply(
        lambda group: group["구조환경_연도별순위"].corr(group["TFR_연도별순위"], method="spearman"),
        include_groups=False,
    )
    result["연도별_Spearman_rho"] = result["연도"].map(annual_spearman)
    result["구조환경_시도내_z"] = result.groupby("지역")["구조환경종합지수"].transform(_within_z)
    result["TFR_시도내_z"] = result.groupby("지역")["합계출산율"].transform(_within_z)
    correlations = result.groupby("지역").apply(
        lambda group: group["구조환경_시도내_z"].corr(group["TFR_시도내_z"]),
        include_groups=False,
    )
    result["시도별_동시점_Pearson_r"] = result["지역"].map(correlations)
    return result.sort_values(["지역", "연도"]).reset_index(drop=True)


def build_composite_fiscal_panel(fiscal: pd.DataFrame) -> pd.DataFrame:
    """11개 영역 실질 1인당 예산을 합산해 pooled 종합 재정대응점수를 만든다."""
    required = {"지역", "연도", "세부영역", REAL_PC, "예산결측_사업수"}
    if missing := required - set(fiscal.columns):
        raise KeyError(f"재정대응 필수 컬럼 누락: {sorted(missing)}")
    included = fiscal.loc[fiscal["세부영역"].ne("지표체계 외")].copy()
    counts = included.groupby(["지역", "연도"])["세부영역"].nunique()
    if len(counts) != REGION_COUNT * len(YEARS) or not counts.eq(11).all():
        raise ValueError("모든 지역×연도에 지표체계 내 11개 세부영역이 필요합니다.")
    result = (
        included.groupby(["지역", "연도"], as_index=False)
        .agg(
            지표체계내_실질인구1인당예산_원=(REAL_PC, "sum"),
            예산결측_사업수=("예산결측_사업수", "sum"),
        )
        .sort_values(["지역", "연도"])
        .reset_index(drop=True)
    )
    result["log1p_지표체계내_실질인구1인당예산"] = np.log1p(
        result["지표체계내_실질인구1인당예산_원"]
    )
    logged = result["log1p_지표체계내_실질인구1인당예산"]
    result["종합재정대응지수_z"] = (logged - logged.mean()) / logged.std(ddof=0)
    zscore = result["종합재정대응지수_z"]
    result["종합재정대응점수_0_100"] = (zscore - zscore.min()) / (zscore.max() - zscore.min()) * 100
    result["연도별_종합재정대응순위"] = result.groupby("연도")["종합재정대응점수_0_100"].rank(
        ascending=False, method="min"
    )
    result["예산누락주의"] = result["예산결측_사업수"].gt(0)
    return result


def validate_moving_results(results: pd.DataFrame) -> None:
    """t+1·t+2 각 11개 영역의 계수·구간·q값을 검증한다."""
    required = {
        "모형버전",
        "모형",
        "계수",
        "95%신뢰구간_하한",
        "95%신뢰구간_상한",
        "FDR_q값",
        "FDR_0.05_유의",
        "관측치",
    }
    if missing := required - set(results.columns):
        raise KeyError(f"이동평균 결과 필수 컬럼 누락: {sorted(missing)}")
    expected = {"3개년평균_t+1", "3개년평균_t+2"}
    counts = results.groupby("모형버전")["모형"].nunique()
    if set(counts.index) != expected or not counts.eq(11).all() or len(results) != 22:
        raise ValueError("이동평균 결과는 t+1·t+2 각 11개 영역, 총 22행이어야 합니다.")


def plot_structural_tfr_trends(data: pd.DataFrame) -> plt.Figure:
    """시도별 구조환경 종합지수와 TFR의 표준화 추세를 그린다."""
    regions = sorted(data["지역"].unique())
    fig, axes = plt.subplots(5, 4, figsize=(18, 20), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, regions, strict=False):
        group = data.loc[data["지역"].eq(region)].sort_values("연도")
        ax.plot(
            group["연도"],
            group["구조환경_시도내_z"],
            color=YOMOCHA_WEB_COLORS["accent"],
            marker="o",
            label="구조환경 종합지수",
        )
        ax.plot(
            group["연도"],
            group["TFR_시도내_z"],
            color="#D97706",
            marker="s",
            linestyle="--",
            label="합계출산율",
        )
        correlation = group["시도별_동시점_Pearson_r"].iloc[0]
        ax.set_title(f"{region}  (r={correlation:+.2f})")
        ax.axhline(0, color="#E5E7EB", linewidth=0.8)
        ax.set_xticks([2016, 2018, 2020, 2022, 2024])
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(regions) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=2, frameon=False
    )
    fig.suptitle("시도별 구조환경 종합지수와 합계출산율 추세", y=0.995, fontsize=17)
    fig.text(
        0.5,
        0.948,
        "2016~2024년·각 시도 내 z-score·r은 9개 동시점의 기술적 Pearson 상관이며 인과효과가 아님",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("연도")
    fig.supylabel("시도 내 표준화값(z)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.935))
    return fig


def plot_fiscal_score_trends(data: pd.DataFrame) -> plt.Figure:
    """시도별 종합 재정대응점수와 연도 중앙값의 추세를 그린다."""
    regions = sorted(data["지역"].unique())
    median = data.groupby("연도")["종합재정대응점수_0_100"].median()
    fig, axes = plt.subplots(5, 4, figsize=(18, 20), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, regions, strict=False):
        group = data.loc[data["지역"].eq(region)].sort_values("연도")
        ax.plot(
            group["연도"],
            group["종합재정대응점수_0_100"],
            color=YOMOCHA_WEB_COLORS["accent"],
            marker="o",
            label="해당 시도",
        )
        ax.plot(
            median.index,
            median.values,
            color=YOMOCHA_WEB_COLORS["line"],
            linestyle="--",
            label="17개 시도 중앙값",
        )
        ax.set_title(region)
        ax.set_xticks([2016, 2018, 2020, 2022, 2024])
        ax.set_ylim(-3, 103)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(regions) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=2, frameon=False
    )
    fig.suptitle("시도별 종합 재정대응점수 추세", y=0.995, fontsize=17)
    fig.text(
        0.5,
        0.948,
        "2016~2024년·지표체계 내 11개 영역 실질 1인당 예산 합계·log1p pooled 표준화 후 0~100 표시",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("연도")
    fig.supylabel("종합 재정대응점수(0~100)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.935))
    return fig


def plot_annual_rank_scatter(data: pd.DataFrame) -> plt.Figure:
    """연도별 구조환경과 TFR의 시도 순위를 산점도로 비교한다."""
    years = sorted(data["연도"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)
    for ax, year in zip(axes.flat, years, strict=True):
        group = data.loc[data["연도"].eq(year)]
        ax.scatter(
            group["구조환경_연도별순위점수"],
            group["TFR_연도별순위점수"],
            color=YOMOCHA_WEB_COLORS["accent"],
            s=28,
            alpha=0.8,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                row.지역,
                (row.구조환경_연도별순위점수, row.TFR_연도별순위점수),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.plot([0, 100], [0, 100], color="#9CA3AF", linestyle="--", linewidth=1)
        rho = group["연도별_Spearman_rho"].iloc[0]
        ax.set_title(f"{year}년  (Spearman ρ={rho:+.2f})")
        ax.set_xlim(-5, 110)
        ax.set_ylim(-5, 110)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("연도별 구조환경 종합지수와 합계출산율의 시도 순위 비교", y=0.995, fontsize=17)
    fig.text(
        0.5,
        0.965,
        "1위=100·17위=0·우상단일수록 두 순위 모두 높음·동시점의 기술적 비교이며 인과효과가 아님",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("구조환경 종합지수 연도별 순위점수")
    fig.supylabel("합계출산율 연도별 순위점수")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    return fig


def plot_rank_gap_heatmap(data: pd.DataFrame) -> plt.Figure:
    """시도·연도별 구조환경–TFR 절대 순위차이를 히트맵으로 그린다."""
    order = data.groupby("지역")["절대순위차이"].mean().sort_values().index
    gaps = data.pivot(index="지역", columns="연도", values="절대순위차이").loc[order]
    structural_ranks = data.pivot(index="지역", columns="연도", values="구조환경_연도별순위").loc[
        order
    ]
    tfr_ranks = data.pivot(index="지역", columns="연도", values="TFR_연도별순위").loc[order]
    annotations = structural_ranks.astype(int).astype(str) + "/" + tfr_ranks.astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        gaps,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=REGION_COUNT - 1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "절대 순위차이(0=두 순위 동일)"},
        ax=ax,
    )
    ax.set_xlabel("연도")
    ax.set_ylabel("시도(평균 절대 순위차이가 작은 순)")
    fig.suptitle("시도별 구조환경 종합지수와 합계출산율의 순위차이", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.955,
        "셀 숫자=구조환경 순위/TFR 순위·옅을수록 두 순위가 유사함",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_regional_rank_trajectories(data: pd.DataFrame) -> plt.Figure:
    """각 시도의 구조환경–TFR 순위점수 궤적을 소패널로 그린다."""
    regions = sorted(data["지역"].unique())
    fig, axes = plt.subplots(5, 4, figsize=(18, 21), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, regions, strict=False):
        group = data.loc[data["지역"].eq(region)].sort_values("연도")
        ax.scatter(
            data["구조환경_연도별순위점수"],
            data["TFR_연도별순위점수"],
            color="#D1D5DB",
            s=9,
            alpha=0.25,
        )
        ax.plot(
            group["구조환경_연도별순위점수"],
            group["TFR_연도별순위점수"],
            color=YOMOCHA_WEB_COLORS["accent"],
            marker="o",
            linewidth=1.4,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                str(row.연도)[-2:],
                (row.구조환경_연도별순위점수, row.TFR_연도별순위점수),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
            )
        ax.plot([0, 100], [0, 100], color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.set_title(region)
        ax.set_xlim(-5, 108)
        ax.set_ylim(-5, 108)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[len(regions) :]:
        ax.set_visible(False)
    fig.suptitle("시도별 구조환경 종합지수–합계출산율 순위 궤적", y=0.995, fontsize=17)
    fig.text(
        0.5,
        0.965,
        "점 표기=연도 끝 두 자리·1위=100·17위=0·대각선에 가까울수록 두 순위가 유사함",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("구조환경 종합지수 연도별 순위점수")
    fig.supylabel("합계출산율 연도별 순위점수")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    return fig


def plot_fiscal_rank_heatmap(data: pd.DataFrame) -> plt.Figure:
    """종합 재정대응점수의 연도별 시도 순위를 히트맵으로 그린다."""
    latest_order = (
        data.loc[data["연도"].eq(2024)].sort_values("연도별_종합재정대응순위")["지역"].tolist()
    )
    matrix = data.pivot(index="지역", columns="연도", values="연도별_종합재정대응순위").loc[
        latest_order
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues_r",
        vmin=1,
        vmax=17,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "연도별 순위(1위=실질 1인당 예산 상대적 최대)"},
        ax=ax,
    )
    ax.set_xlabel("연도")
    ax.set_ylabel("시도(2024년 순위 순)")
    fig.suptitle("종합 재정대응점수의 연도별 시도 순위", y=0.99, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_moving_average_forest(results: pd.DataFrame) -> plt.Figure:
    """3개년 평균예산–TFR t+1·t+2 계수와 95% 신뢰구간을 비교한다."""
    validate_moving_results(results)
    t2 = results.loc[results["모형버전"].eq("3개년평균_t+2")].sort_values("계수", ascending=False)
    order = t2["모형"].tolist()
    all_low = results["95%신뢰구간_하한"].min()
    all_high = results["95%신뢰구간_상한"].max()
    padding = (all_high - all_low) * 0.08
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, version, label in zip(
        axes,
        ["3개년평균_t+1", "3개년평균_t+2"],
        ["모형 3-1: 평균창 종료 1년 후 TFR", "모형 3-2: 평균창 종료 2년 후 TFR"],
        strict=True,
    ):
        panel = (
            results.loc[results["모형버전"].eq(version)].set_index("모형").loc[order].reset_index()
        )
        y = np.arange(len(panel))
        colors = np.where(
            panel["FDR_0.05_유의"], YOMOCHA_WEB_COLORS["accent"], YOMOCHA_WEB_COLORS["muted_bar"]
        )
        for position, (_, row), color in zip(y, panel.iterrows(), colors, strict=True):
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
        ax.axvline(0, color="#6B7280", linewidth=1)
        ax.set_xlim(all_low - padding, all_high + padding)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("계수와 95% 신뢰구간")
        ax.grid(axis="y", visible=False)
    fig.suptitle("3개년 평균 재정대응예산과 후행 TFR의 세부영역별 관련성 계수", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "시도·연도 고정효과·시도 군집표준오차·파란색은 BH q<0.05·계수는 영향력이 아닌 조건부 관련성",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--fiscal", type=Path, default=DEFAULT_FISCAL)
    parser.add_argument("--tfr", type=Path, default=DEFAULT_TFR)
    parser.add_argument("--moving-results", type=Path, default=DEFAULT_MOVING_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structural_tfr = build_structural_tfr_panel(pd.read_csv(args.structural), pd.read_csv(args.tfr))
    fiscal = build_composite_fiscal_panel(pd.read_csv(args.fiscal))
    moving_results = pd.read_csv(args.moving_results)
    validate_moving_results(moving_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structural_tfr.to_csv(
        args.output_dir / "2016-2024_시도별_구조환경종합지수_TFR_표준화추세.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fiscal.to_csv(
        args.output_dir / "2016-2024_시도별_종합재정대응점수_추세.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figures = {
        "2016-2024_시도별_구조환경종합지수_TFR_표준화추세": plot_structural_tfr_trends(
            structural_tfr
        ),
        "2016-2024_연도별_구조환경_TFR_순위산점도": plot_annual_rank_scatter(structural_tfr),
        "2016-2024_시도별_구조환경_TFR_순위차이_히트맵": plot_rank_gap_heatmap(structural_tfr),
        "2016-2024_시도별_구조환경_TFR_순위궤적": plot_regional_rank_trajectories(structural_tfr),
        "2016-2024_시도별_종합재정대응점수_추세": plot_fiscal_score_trends(fiscal),
        "2016-2024_종합재정대응점수_연도별_시도순위": plot_fiscal_rank_heatmap(fiscal),
        "2016-2024_3개년평균예산_TFR_세부영역별_계수_포리스트플롯": plot_moving_average_forest(
            moving_results
        ),
    }
    for name, figure in figures.items():
        save_figure(figure, args.figure_dir / name)
        plt.close(figure)


if __name__ == "__main__":
    main()
