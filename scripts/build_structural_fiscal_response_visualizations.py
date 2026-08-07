"""#102 구조환경 → 재정대응 반응성 결과를 시각화한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_구조환경_재정대응_공통반응계수.csv"
)
DEFAULT_COMMON_T1 = DEFAULT_COMMON.with_name(f"{DEFAULT_COMMON.stem}_t+1.csv")
DEFAULT_REGIONAL = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_시도별_구조환경_재정대응_반응계수.csv"
)
DEFAULT_FISCAL = REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_인구1인당_실질예산액.csv"
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_구조환경_재정대응_반응성_표본.csv"
)
DEFAULT_SAMPLE_T1 = DEFAULT_SAMPLE.with_name(f"{DEFAULT_SAMPLE.stem}_t+1.csv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/구조환경_재정대응_반응성"

STRUCTURAL_COL = "구조환경지수_t"


def _budget_columns(lag_years: int) -> tuple[str, str]:
    if lag_years not in (1, 2):
        raise ValueError("시각화 예산 시차는 1년 또는 2년이어야 합니다.")
    return (
        f"인구1인당_실질예산_t+{lag_years}_원",
        f"log1p_인구1인당_실질예산_t+{lag_years}",
    )


def _standardize(values: pd.Series, *, label: str) -> pd.Series:
    """비교용 연도계열을 평균 0·표준편차 1로 변환한다."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{label}에 비수치 또는 결측값이 있습니다.")
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}의 연도별 변동이 없어 표준화할 수 없습니다.")
    return (numeric - float(numeric.mean())) / scale


def plot_aligned_structure_budget_trends(
    sample: pd.DataFrame,
    *,
    lag_years: int = 2,
) -> plt.Figure:
    """t년 구조환경과 t+lag년 예산 중앙값을 표준화해 시간축에 정렬한다."""
    budget_col, _ = _budget_columns(lag_years)
    required = {
        "세부영역",
        "구조환경연도",
        "예산연도",
        STRUCTURAL_COL,
        budget_col,
    }
    missing = sorted(required - set(sample.columns))
    if missing:
        raise KeyError(f"구조환경·예산 정렬 추이 필수 컬럼 누락: {missing}")

    summary = (
        sample.groupby(["세부영역", "구조환경연도", "예산연도"], as_index=False)
        .agg(
            구조환경_중앙값=(STRUCTURAL_COL, "median"),
            예산_중앙값=(budget_col, "median"),
            예산_하위25=(budget_col, lambda values: values.quantile(0.25)),
            예산_상위75=(budget_col, lambda values: values.quantile(0.75)),
        )
        .sort_values(["세부영역", "구조환경연도"])
    )
    standardized = []
    for subarea, group in summary.groupby("세부영역", sort=True):
        group = group.copy()
        group["구조환경_z"] = _standardize(
            group["구조환경_중앙값"], label=f"{subarea} 구조환경 중앙값"
        )
        budget_mean = float(group["예산_중앙값"].mean())
        budget_scale = float(group["예산_중앙값"].std(ddof=0))
        if not np.isfinite(budget_scale) or budget_scale <= 0:
            raise ValueError(f"{subarea} 예산 중앙값의 연도별 변동이 없습니다.")
        for source, target in (
            ("예산_중앙값", "예산_z"),
            ("예산_하위25", "예산_하위25_z"),
            ("예산_상위75", "예산_상위75_z"),
        ):
            group[target] = (group[source] - budget_mean) / budget_scale
        standardized.append(group)
    plot = pd.concat(standardized, ignore_index=True)

    subareas = sorted(plot["세부영역"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    axes_flat = axes.ravel()
    structure_color = "#7C3AED"
    budget_color = "#0E7490"
    budget_fill = "#CFFAFE"
    for ax, subarea in zip(axes_flat, subareas, strict=False):
        group = plot.loc[plot["세부영역"].eq(subarea)]
        years = group["구조환경연도"].to_numpy(dtype=int)
        ax.fill_between(
            years,
            group["예산_하위25_z"].to_numpy(dtype=float),
            group["예산_상위75_z"].to_numpy(dtype=float),
            color=budget_fill,
            alpha=0.75,
            linewidth=0,
        )
        ax.plot(
            years,
            group["구조환경_z"],
            color=structure_color,
            marker="o",
            linewidth=2,
            markersize=3.5,
            label="구조환경 t",
        )
        ax.plot(
            years,
            group["예산_z"],
            color=budget_color,
            marker="s",
            linestyle="--",
            linewidth=2,
            markersize=3.5,
            label=f"예산 t+{lag_years}",
        )
        ax.axhline(0, color="#D1D5DB", linewidth=0.8)
        ax.set_title(subarea, loc="left", fontsize=11)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes_flat[len(subareas) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("구조환경 측정연도(t)")
    axes_flat[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        f"구조환경과 {lag_years}년 후 재정대응예산의 정렬 추이",
        y=0.985,
        fontsize=17,
    )
    fig.text(
        0.5,
        0.957,
        "각 선은 영역 내 연도계열 표준화값·예산 음영은 17개 시도 25~75% 범위(예산계열 기준 표준화)",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.text(0.015, 0.5, "영역 내 표준화값(z)", rotation=90, va="center", fontsize=11)
    fig.tight_layout(rect=[0.025, 0.025, 1, 0.93])
    return fig


def _two_way_residuals(
    group: pd.DataFrame,
    column: str,
    *,
    region_col: str = "지역",
    year_col: str = "예산연도",
) -> pd.Series:
    """균형패널에서 시도·연도 평균을 제거한 값을 계산한다."""
    values = pd.to_numeric(group[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column}에 비수치 또는 결측값이 있습니다.")
    if group.duplicated([region_col, year_col]).any():
        raise ValueError(f"{column}의 {region_col}·{year_col} 키가 중복됩니다.")
    regions = group[region_col].nunique()
    years = group[year_col].nunique()
    if len(group) != regions * years:
        raise ValueError(f"{column}의 이원 고정효과 제거에는 균형패널이 필요합니다.")
    return (
        values
        - values.groupby(group[region_col]).transform("mean")
        - values.groupby(group[year_col]).transform("mean")
        + float(values.mean())
    )


def plot_fixed_effects_response_scatter(
    sample: pd.DataFrame,
    common: pd.DataFrame,
    *,
    lag_years: int = 2,
) -> plt.Figure:
    """고정효과 제거 후 구조환경과 t+lag 예산의 관계를 영역별로 그린다."""
    _, log_budget_col = _budget_columns(lag_years)
    sample_required = {"세부영역", "지역", "예산연도", STRUCTURAL_COL, log_budget_col}
    common_required = {"모형", "계수", "95%신뢰구간_하한", "95%신뢰구간_상한", "p값", "FDR_q값"}
    for frame, required, label in (
        (sample, sample_required, "고정효과 산점도 표본"),
        (common, common_required, "고정효과 산점도 계수"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{label} 필수 컬럼 누락: {missing}")

    coefficient_lookup = common.set_index("모형")
    if not coefficient_lookup.index.is_unique:
        raise ValueError("공통계수의 모형 값이 중복됩니다.")
    subareas = sorted(sample["세부영역"].unique())
    if set(subareas) != set(coefficient_lookup.index):
        raise ValueError("산점도 표본과 공통계수의 세부영역 구성이 다릅니다.")

    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    axes_flat = axes.ravel()
    point_color = "#94A3B8"
    line_color = "#0E7490"
    fill_color = "#CFFAFE"
    for ax, subarea in zip(axes_flat, subareas, strict=False):
        group = sample.loc[sample["세부영역"].eq(subarea)].copy()
        x = _two_way_residuals(group, STRUCTURAL_COL).to_numpy(dtype=float)
        y = _two_way_residuals(group, log_budget_col).to_numpy(dtype=float)
        coefficient = coefficient_lookup.loc[subarea]
        beta = float(coefficient["계수"])
        lower = float(coefficient["95%신뢰구간_하한"])
        upper = float(coefficient["95%신뢰구간_상한"])
        grid = np.linspace(float(x.min()), float(x.max()), 100)

        ax.scatter(x, y, s=13, color=point_color, alpha=0.45, edgecolors="none")
        ax.fill_between(
            grid,
            np.minimum(lower * grid, upper * grid),
            np.maximum(lower * grid, upper * grid),
            color=fill_color,
            alpha=0.8,
            linewidth=0,
        )
        ax.plot(grid, beta * grid, color=line_color, linewidth=2)
        ax.axhline(0, color="#D1D5DB", linewidth=0.7)
        ax.axvline(0, color="#D1D5DB", linewidth=0.7)
        ax.set_title(subarea, loc="left", fontsize=11)
        ax.text(
            0.02,
            0.96,
            f"β={beta:.3f} · p={coefficient['p값']:.3f} · q={coefficient['FDR_q값']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#4B5563",
        )
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes_flat[len(subareas) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("구조환경지수(t): 시도·연도 고정효과 제거값")
    fig.suptitle(
        f"구조환경과 {lag_years}년 후 재정대응예산의 고정효과 관계",
        y=0.985,
        fontsize=17,
    )
    fig.text(
        0.5,
        0.957,
        "점=시도×연도, 선=공통 반응계수, 음영=계수의 95% 신뢰구간 범위·음의 기울기는 수요반응 방향",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.text(
        0.012,
        0.5,
        f"log1p(실질 1인당 예산 t+{lag_years}): 시도·연도 고정효과 제거값",
        rotation=90,
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.025, 0.025, 1, 0.93])
    return fig


def _format_won(value: float) -> str:
    """축 눈금을 읽기 쉬운 원 단위로 표시한다."""

    def compact(number: float) -> str:
        return f"{number:.1f}".rstrip("0").rstrip(".")

    absolute = abs(value)
    if absolute >= 100_000_000:
        return f"{compact(value / 100_000_000)}억"
    if absolute >= 10_000:
        return f"{compact(value / 10_000)}만"
    if absolute >= 1_000:
        return f"{compact(value / 1_000)}천"
    return f"{value:.0f}"


def plot_fiscal_budget_trends(fiscal: pd.DataFrame) -> plt.Figure:
    """세부영역별 실질 1인당 예산의 연도별 지역 중앙값과 IQR을 그린다."""
    required = {"연도", "세부영역", "지역", "인구1인당_실질예산_원"}
    missing = sorted(required - set(fiscal.columns))
    if missing:
        raise KeyError(f"예산 추이 그래프 필수 컬럼 누락: {missing}")

    plot = fiscal.loc[fiscal["세부영역"].ne("지표체계 외")].copy()
    plot["인구1인당_실질예산_원"] = pd.to_numeric(plot["인구1인당_실질예산_원"], errors="coerce")
    if plot["인구1인당_실질예산_원"].isna().any():
        raise ValueError("예산 추이 자료에 비수치 또는 결측 예산이 있습니다.")
    if plot["인구1인당_실질예산_원"].lt(0).any():
        raise ValueError("예산 추이 자료에 음수 예산이 있습니다.")

    summary = (
        plot.groupby(["세부영역", "연도"], as_index=False)["인구1인당_실질예산_원"]
        .agg(
            중앙값="median",
            하위25=lambda values: values.quantile(0.25),
            상위75=lambda values: values.quantile(0.75),
        )
        .sort_values(["세부영역", "연도"])
    )
    subareas = sorted(summary["세부영역"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    axes_flat = axes.ravel()
    line_color = "#0E7490"
    fill_color = "#CFFAFE"

    for ax, subarea in zip(axes_flat, subareas, strict=False):
        group = summary.loc[summary["세부영역"].eq(subarea)]
        years = group["연도"].to_numpy(dtype=int)
        median = group["중앙값"].to_numpy(dtype=float)
        lower = group["하위25"].to_numpy(dtype=float)
        upper = group["상위75"].to_numpy(dtype=float)
        ax.fill_between(years, lower, upper, color=fill_color, alpha=0.8, linewidth=0)
        ax.plot(
            years,
            median,
            color=line_color,
            linewidth=2,
            marker="o",
            markersize=3.5,
        )
        ax.set_title(subarea, loc="left", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_won(value)))
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes_flat[len(subareas) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("예산연도")

    fig.suptitle("세부영역별 실질 1인당 재정대응예산 추이", y=0.985, fontsize=17)
    fig.text(
        0.5,
        0.957,
        "선=17개 시도 중앙값, 음영=지역 간 25~75% 범위·2024년 화폐가치·전체 주민등록인구 1인당",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.text(0.015, 0.5, "실질 1인당 예산(원)", rotation=90, va="center", fontsize=11)
    fig.tight_layout(rect=[0.025, 0.025, 1, 0.93])
    return fig


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
    parser.add_argument("--fiscal", type=Path, default=DEFAULT_FISCAL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-t1", type=Path, default=DEFAULT_SAMPLE_T1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not args.common.is_file()
        or not args.common_t1.is_file()
        or not args.regional.is_file()
        or not args.fiscal.is_file()
        or not args.sample.is_file()
        or not args.sample_t1.is_file()
    ):
        raise FileNotFoundError("공통 또는 시도별 반응계수 산출물이 없습니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = pd.read_csv(args.common)
    common_t1 = pd.read_csv(args.common_t1)
    regional = pd.read_csv(args.regional)
    fiscal = pd.read_csv(args.fiscal)
    sample = pd.read_csv(args.sample)
    sample_t1 = pd.read_csv(args.sample_t1)
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
    budget_trend_figure = plot_fiscal_budget_trends(fiscal)
    aligned_trend_figure = plot_aligned_structure_budget_trends(sample)
    response_scatter_figure = plot_fixed_effects_response_scatter(sample, common)
    aligned_trend_t1_figure = plot_aligned_structure_budget_trends(sample_t1, lag_years=1)
    response_scatter_t1_figure = plot_fixed_effects_response_scatter(
        sample_t1,
        common_t1,
        lag_years=1,
    )
    save_figure(common_figure, args.output_dir / "세부영역별_공통_반응계수", formats=["png"])
    save_figure(regional_figure, args.output_dir / "시도별_반응계수_히트맵", formats=["png"])
    save_figure(comparison_figure, args.output_dir / "t+1_t+2_공통_반응계수_비교", formats=["png"])
    save_figure(
        budget_trend_figure,
        args.output_dir / "세부영역별_실질1인당_예산_추이",
        formats=["png"],
    )
    save_figure(
        aligned_trend_figure,
        args.output_dir / "세부영역별_구조환경_t_예산_t+2_정렬추이",
        formats=["png"],
    )
    save_figure(
        response_scatter_figure,
        args.output_dir / "세부영역별_구조환경_t_예산_t+2_FE관계",
        formats=["png"],
    )
    save_figure(
        aligned_trend_t1_figure,
        args.output_dir / "세부영역별_구조환경_t_예산_t+1_정렬추이",
        formats=["png"],
    )
    save_figure(
        response_scatter_t1_figure,
        args.output_dir / "세부영역별_구조환경_t_예산_t+1_FE관계",
        formats=["png"],
    )
    plt.close(common_figure)
    plt.close(regional_figure)
    plt.close(comparison_figure)
    plt.close(budget_trend_figure)
    plt.close(aligned_trend_figure)
    plt.close(response_scatter_figure)
    plt.close(aligned_trend_t1_figure)
    plt.close(response_scatter_t1_figure)
    print(f"저장: {args.output_dir}")


if __name__ == "__main__":
    main()
