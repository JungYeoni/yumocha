"""지역×연도 구조환경지표·계획예산 추세 시각화."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.plots import PALETTE


def shade_basic_plan_periods(ax: plt.Axes) -> None:
    """제3차·제4차 기본계획 기간을 모든 시계열 축에 동일하게 표시한다."""

    ax.axvspan(2015.5, 2020.5, color=PALETTE[0], alpha=0.08, label="제3차")
    ax.axvspan(2020.5, 2024.5, color=PALETTE[2], alpha=0.08, label="제4차")
    ax.axvline(2020.5, color="0.45", linewidth=0.8, linestyle="--")


def plot_structural_indicator_overview(
    indicator_df: pd.DataFrame,
    *,
    indicator: str,
) -> plt.Figure:
    """한 구조환경지표의 기준 추세·지역 분포·결측·급등락을 요약한다."""

    data = indicator_df.loc[indicator_df["세부지표"].eq(indicator)].copy()
    regions = data.loc[~data["지역"].eq("전국")].copy()
    nationwide = data.loc[data["지역"].eq("전국") & data["측정값"].notna()]
    years = sorted(data["연도"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    shade_basic_plan_periods(ax)
    if not nationwide.empty:
        ax.plot(
            nationwide["연도"],
            nationwide["측정값"],
            marker="o",
            color=PALETTE[0],
            linewidth=2,
            label="전국 공표·산출값",
        )
        reference_label = "전국값"
    else:
        summary = regions.groupby("연도")["측정값"].agg(
            중앙값="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        ax.plot(
            summary.index,
            summary["중앙값"],
            marker="o",
            color=PALETTE[0],
            linewidth=2,
            label="17개 시도 중앙값",
        )
        ax.fill_between(
            summary.index,
            summary["q1"],
            summary["q3"],
            color=PALETTE[0],
            alpha=0.18,
            label="시도 간 IQR",
        )
        reference_label = "전국 미공표·시도 중앙값"
    ax.set_title(f"기준 추세 ({reference_label})")
    ax.set_xticks(years)
    ax.legend(fontsize=9)

    sns.boxplot(
        data=regions,
        x="연도",
        y="측정값",
        color=PALETTE[0],
        fliersize=2,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("연도별 17개 시도 분포")
    axes[0, 1].tick_params(axis="x", rotation=45)

    missing = regions.groupby("연도")["실측여부"].apply(lambda values: int((~values).sum()))
    axes[1, 0].bar(missing.index, missing.values, color=PALETTE[1])
    axes[1, 0].set_title("원자료 결측 지역 수")
    axes[1, 0].set_xticks(years)
    axes[1, 0].set_ylim(0, max(1, int(missing.max()) + 1))

    outliers = regions.groupby("연도")["급등락후보"].sum().astype(int)
    axes[1, 1].bar(outliers.index, outliers.values, color=PALETTE[3])
    axes[1, 1].set_title("전년 대비 급등락 후보 수(IQR 기준)")
    axes[1, 1].set_xticks(years)
    axes[1, 1].set_ylim(0, max(1, int(outliers.max()) + 1))

    for current_ax in axes.flat:
        current_ax.set_xlabel("연도")
    fig.suptitle(f"{indicator} — 실측값 기준 지역·연도 EDA", fontsize=16)
    fig.tight_layout()
    return fig


def plot_region_small_multiples(
    indicator_df: pd.DataFrame,
    *,
    indicator: str,
    region_order: list[str],
) -> plt.Figure:
    """한 구조환경지표의 17개 시도 추세를 같은 축 범위의 small multiples로 표시한다."""

    data = indicator_df.loc[
        indicator_df["세부지표"].eq(indicator) & indicator_df["지역"].isin(region_order)
    ].copy()
    n_cols = 4
    n_rows = math.ceil(len(region_order) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 2.8 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    for ax, region in zip(axes, region_order, strict=False):
        region_data = data.loc[data["지역"].eq(region)]
        shade_basic_plan_periods(ax)
        ax.plot(
            region_data["연도"],
            region_data["측정값"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=PALETTE[0],
        )
        flagged = region_data.loc[region_data["급등락후보"].fillna(False)]
        ax.scatter(
            flagged["연도"],
            flagged["측정값"],
            color=PALETTE[3],
            marker="D",
            s=24,
            zorder=3,
        )
        ax.set_title(region)

    for ax in axes[len(region_order) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"{indicator} — 17개 시도 추세 (빈 구간=원자료 결측, ◆=급등락 후보)",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


def plot_budget_overview(budget_df: pd.DataFrame) -> plt.Figure:
    """명목 계획예산의 전국 합계·지역 분포·증감률·계획기간을 요약한다."""

    plot_data = budget_df.assign(당해계획예산_조원=budget_df["당해계획예산_백만원"].div(1_000_000))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    annual_total = plot_data.groupby("연도")["당해계획예산_조원"].sum()
    ax = axes[0, 0]
    shade_basic_plan_periods(ax)
    ax.plot(
        annual_total.index,
        annual_total.values,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
    )
    ax.set_title("17개 시도 명목 계획예산 합계")
    ax.set_ylabel("조원")
    ax.set_xticks(annual_total.index)

    sns.boxplot(
        data=plot_data,
        x="연도",
        y="당해계획예산_조원",
        color=PALETTE[0],
        fliersize=2,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("연도별 시도 계획예산 분포")
    axes[0, 1].tick_params(axis="x", rotation=45)

    sns.boxplot(
        data=plot_data,
        x="연도",
        y="전년대비증감률_pct",
        color=PALETTE[2],
        fliersize=2,
        ax=axes[1, 0],
    )
    flagged = plot_data.loc[plot_data["급등락후보"]]
    year_positions = {year: index for index, year in enumerate(sorted(plot_data["연도"].unique()))}
    axes[1, 0].scatter(
        flagged["연도"].map(year_positions),
        flagged["전년대비증감률_pct"],
        color=PALETTE[3],
        marker="D",
        s=28,
        label="급등락 후보",
        zorder=3,
    )
    axes[1, 0].axhline(0, color="0.4", linewidth=0.8)
    axes[1, 0].set_title("전년 대비 증감률 분포(IQR 기준)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(fontsize=9)

    sns.boxplot(
        data=plot_data,
        x="기본계획기간",
        y="당해계획예산_조원",
        color=PALETTE[1],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("제3차·제4차 기본계획 기간 분포")
    axes[1, 1].tick_params(axis="x", rotation=10)

    fig.suptitle("2016~2024년 시도별 명목 계획예산 EDA", fontsize=16)
    fig.tight_layout()
    return fig


def plot_budget_region_small_multiples(
    budget_df: pd.DataFrame,
    *,
    region_order: list[str],
) -> plt.Figure:
    """17개 시도의 명목 계획예산 추세를 small multiples로 표시한다."""

    n_cols = 4
    n_rows = math.ceil(len(region_order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.8 * n_rows), sharex=True)
    axes = axes.flatten()

    for ax, region in zip(axes, region_order, strict=False):
        region_data = budget_df.loc[budget_df["지역"].eq(region)].assign(
            당해계획예산_조원=lambda frame: frame["당해계획예산_백만원"].div(1_000_000)
        )
        shade_basic_plan_periods(ax)
        ax.plot(
            region_data["연도"],
            region_data["당해계획예산_조원"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=PALETTE[0],
        )
        flagged = region_data.loc[region_data["급등락후보"]]
        ax.scatter(
            flagged["연도"],
            flagged["당해계획예산_조원"],
            color=PALETTE[3],
            marker="D",
            s=24,
            zorder=3,
        )
        quality = region_data.loc[region_data["원자료_누락주의"].notna()]
        ax.scatter(
            quality["연도"],
            quality["당해계획예산_조원"],
            facecolors="none",
            edgecolors="black",
            marker="o",
            s=70,
            linewidth=1.2,
            zorder=4,
        )
        ax.set_title(region)
        ax.set_ylabel("조원")

    for ax in axes[len(region_order) :]:
        ax.set_visible(False)
    fig.suptitle(
        "시도별 명목 계획예산 추세 (◆=급등락 후보, ○=원자료 누락주의)",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


__all__ = [
    "plot_budget_overview",
    "plot_budget_region_small_multiples",
    "plot_region_small_multiples",
    "plot_structural_indicator_overview",
    "shade_basic_plan_periods",
]
