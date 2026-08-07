"""지역×연도 구조환경지표·계획예산 추세 시각화."""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.plots import PALETTE


def _check_log_axis_values(series: pd.Series, *, label: str) -> None:
    """로그축에 그릴 값의 음수·0을 확인한다.

    matplotlib은 로그축에서 0 이하 값을 경고 없이 그냥 빼고 그린다 — 예산이
    0인 실제 관측치가 그래프에서 조용히 사라질 수 있어 명시적으로 알린다.
    음수는 예산 데이터에서 있을 수 없는 값이라 오류로 처리한다.
    """

    if series.lt(0).any():
        raise ValueError(f"{label}에 음수 값이 있어 로그축으로 그릴 수 없습니다.")
    zero_count = int(series.eq(0).sum())
    if zero_count:
        warnings.warn(
            f"{label}에 0인 값이 {zero_count}건 있습니다 — 로그축에서는 표시되지 않습니다.",
            stacklevel=2,
        )


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


def plot_fiscal_response_overview(fiscal_response_df: pd.DataFrame) -> plt.Figure:
    """세부영역별 인구1인당 실질예산액(F_it)의 연도별 추세·분포를 로그축으로 요약한다.

    F_it은 세부영역 간 규모 차이가 1만 배 이상이라(#62 로그변환 판단 근거) 모든
    패널을 로그축으로 그린다.
    """

    data = fiscal_response_df.copy()
    _check_log_axis_values(data["인구1인당_실질예산_원"], label="인구1인당_실질예산_원")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    shade_basic_plan_periods(ax)
    summary = data.groupby("연도")["인구1인당_실질예산_원"].agg(
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
        label="전체 세부영역 중앙값",
    )
    ax.fill_between(
        summary.index, summary["q1"], summary["q3"], color=PALETTE[0], alpha=0.18, label="IQR"
    )
    ax.set_yscale("log")
    ax.set_title("연도별 전국 추세(로그축)")
    ax.set_ylabel("인구1인당 실질예산(원, 로그축)")
    ax.set_xticks(sorted(data["연도"].unique()))
    ax.legend(fontsize=9)

    sns.boxplot(
        data=data, x="연도", y="인구1인당_실질예산_원", color=PALETTE[0], fliersize=2, ax=axes[0, 1]
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("연도별 분포(전체 세부영역·지역, 로그축)")
    axes[0, 1].set_ylabel("인구1인당 실질예산(원, 로그축)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    subarea_order = (
        data.groupby("세부영역")["인구1인당_실질예산_원"].median().sort_values().index.tolist()
    )
    sns.boxplot(
        data=data,
        x="인구1인당_실질예산_원",
        y="세부영역",
        order=subarea_order,
        color=PALETTE[0],
        fliersize=2,
        ax=axes[1, 0],
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("세부영역별 분포(로그축) — 규모 차이가 로그변환 근거")
    axes[1, 0].set_xlabel("인구1인당 실질예산(원, 로그축)")

    region_order = (
        data.groupby("지역")["인구1인당_실질예산_원"].median().sort_values().index.tolist()
    )
    sns.boxplot(
        data=data,
        x="인구1인당_실질예산_원",
        y="지역",
        order=region_order,
        color=PALETTE[1],
        fliersize=2,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("지역별 분포(전체 세부영역·연도, 로그축)")
    axes[1, 1].set_xlabel("인구1인당 실질예산(원, 로그축)")

    fig.suptitle("2016~2024년 세부영역별 인구1인당 실질예산액(F_it) EDA", fontsize=16)
    fig.tight_layout(h_pad=4.0, w_pad=3.0)
    return fig


def plot_fiscal_response_subarea_small_multiples(
    fiscal_response_df: pd.DataFrame,
    *,
    subarea_order: list[str],
) -> plt.Figure:
    """세부영역별로 연도 추세(17개 시도 중앙값+IQR)를 로그축 small multiples로 표시한다.

    세부영역마다 규모(y축 범위)가 크게 달라 축은 공유하지 않는다(sharey=False).
    """

    _check_log_axis_values(
        fiscal_response_df["인구1인당_실질예산_원"], label="인구1인당_실질예산_원"
    )

    n_cols = 3
    n_rows = math.ceil(len(subarea_order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows), sharex=True)
    axes = axes.flatten()

    for ax, subarea in zip(axes, subarea_order, strict=False):
        subarea_data = fiscal_response_df.loc[fiscal_response_df["세부영역"].eq(subarea)]
        summary = subarea_data.groupby("연도")["인구1인당_실질예산_원"].agg(
            중앙값="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        shade_basic_plan_periods(ax)
        ax.plot(summary.index, summary["중앙값"], marker="o", markersize=3, color=PALETTE[0])
        ax.fill_between(summary.index, summary["q1"], summary["q3"], color=PALETTE[0], alpha=0.18)
        ax.set_yscale("log")
        ax.set_title(subarea, fontsize=10)

    for ax in axes[len(subarea_order) :]:
        ax.set_visible(False)
    fig.suptitle("세부영역별 인구1인당 실질예산액 추세(17개 시도 중앙값·IQR, 로그축)", fontsize=16)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_budget_overview",
    "plot_budget_region_small_multiples",
    "plot_fiscal_response_overview",
    "plot_fiscal_response_subarea_small_multiples",
    "plot_region_small_multiples",
    "plot_structural_indicator_overview",
    "shade_basic_plan_periods",
]
