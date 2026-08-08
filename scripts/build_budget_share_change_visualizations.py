"""2016~2024년 지역·세부영역별 저출생 대응 계획예산 비중 변화를 산출·시각화한다."""

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
DEFAULT_INPUT = ANALYSIS_DIR / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR
DEFAULT_FIGURE_DIR = REPO_ROOT / "reports/figures/budget_share_change"

BUDGET = "당해계획예산_백만원_provisional"
SUBAREA = "세부영역"
YEARS = set(range(2016, 2025))


def validate_budget_panel(panel: pd.DataFrame) -> None:
    """비중 산출 입력이 17개 시도×9개 연도×12개 분류의 완전 패널인지 확인한다."""
    required = {"지역", "연도", SUBAREA, BUDGET, "사업수", "예산결측_사업수"}
    if missing := required - set(panel.columns):
        raise KeyError(f"예산비중 입력 필수 컬럼 누락: {sorted(missing)}")
    if panel.duplicated(["지역", "연도", SUBAREA]).any():
        raise ValueError("지역×연도×세부영역 키가 중복됩니다.")
    if panel["지역"].nunique() != 17 or set(panel["연도"].unique()) != YEARS:
        raise ValueError("예산비중 입력은 17개 시도의 2016~2024년 패널이어야 합니다.")
    category_counts = panel.groupby(["지역", "연도"])[SUBAREA].nunique()
    if category_counts.nunique() != 1 or category_counts.iloc[0] != 12:
        raise ValueError("모든 지역×연도에 12개 세부영역(지표체계 외 포함)이 필요합니다.")
    if panel[BUDGET].isna().any() or (panel[BUDGET] < 0).any():
        raise ValueError("계획예산에 결측 또는 음수가 있습니다.")


def build_budget_share_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """지역×연도 전체 예산 대비 세부영역별 비중을 계산한다."""
    validate_budget_panel(panel)
    result = panel[["지역", "연도", SUBAREA, BUDGET, "사업수", "예산결측_사업수"]].copy()
    result["지역연도_전체계획예산_백만원"] = result.groupby(["지역", "연도"])[BUDGET].transform(
        "sum"
    )
    if (result["지역연도_전체계획예산_백만원"] <= 0).any():
        raise ValueError("비중 분모인 지역×연도 전체 계획예산은 0보다 커야 합니다.")
    result["계획예산비중_pct"] = result[BUDGET] / result["지역연도_전체계획예산_백만원"] * 100
    result["예산누락주의"] = result["예산결측_사업수"].gt(0)
    return result


def _period_share(data: pd.DataFrame, years: tuple[int, ...], name: str) -> pd.DataFrame:
    period = data.loc[data["연도"].isin(years)].groupby(SUBAREA, as_index=False)[BUDGET].sum()
    period[name] = period[BUDGET] / period[BUDGET].sum() * 100
    return period[[SUBAREA, name]]


def build_national_change_summary(shares: pd.DataFrame) -> pd.DataFrame:
    """전국 시도 합계 기준의 2016→2024년 및 3개년 구간 비중 변화를 요약한다."""
    annual = shares.groupby(["연도", SUBAREA], as_index=False)[BUDGET].sum()
    annual["전국합계기준_비중_pct"] = (
        annual[BUDGET] / annual.groupby("연도")[BUDGET].transform("sum") * 100
    )
    start = annual.loc[annual["연도"].eq(2016), [SUBAREA, "전국합계기준_비중_pct"]].rename(
        columns={"전국합계기준_비중_pct": "2016_비중_pct"}
    )
    end = annual.loc[annual["연도"].eq(2024), [SUBAREA, "전국합계기준_비중_pct"]].rename(
        columns={"전국합계기준_비중_pct": "2024_비중_pct"}
    )
    result = start.merge(end, on=SUBAREA, validate="one_to_one")
    result = result.merge(
        _period_share(shares, (2016, 2017, 2018), "2016-2018_합계기준_비중_pct"),
        on=SUBAREA,
        validate="one_to_one",
    ).merge(
        _period_share(shares, (2022, 2023, 2024), "2022-2024_합계기준_비중_pct"),
        on=SUBAREA,
        validate="one_to_one",
    )
    result["2016→2024_비중변화_pp"] = result["2024_비중_pct"] - result["2016_비중_pct"]
    result["3개년구간_비중변화_pp"] = (
        result["2022-2024_합계기준_비중_pct"] - result["2016-2018_합계기준_비중_pct"]
    )
    result["증감폭절대값_순위"] = (
        result["2016→2024_비중변화_pp"].abs().rank(method="min", ascending=False).astype(int)
    )
    return result.sort_values("2016→2024_비중변화_pp", ascending=False).reset_index(drop=True)


def _regional_period_share(data: pd.DataFrame, years: tuple[int, ...], name: str) -> pd.DataFrame:
    period = (
        data.loc[data["연도"].isin(years)].groupby(["지역", SUBAREA], as_index=False)[BUDGET].sum()
    )
    period[name] = period[BUDGET] / period.groupby("지역")[BUDGET].transform("sum") * 100
    return period[["지역", SUBAREA, name]]


def build_regional_change_table(shares: pd.DataFrame) -> pd.DataFrame:
    """시도별 2016→2024년과 초기→최근 3개년 구간의 비중 변화를 산출한다."""
    annual = shares[["지역", "연도", SUBAREA, "계획예산비중_pct"]]
    start = annual.loc[annual["연도"].eq(2016)].rename(
        columns={"계획예산비중_pct": "2016_비중_pct"}
    )[["지역", SUBAREA, "2016_비중_pct"]]
    end = annual.loc[annual["연도"].eq(2024)].rename(columns={"계획예산비중_pct": "2024_비중_pct"})[
        ["지역", SUBAREA, "2024_비중_pct"]
    ]
    result = start.merge(end, on=["지역", SUBAREA], validate="one_to_one")
    result = result.merge(
        _regional_period_share(shares, (2016, 2017, 2018), "2016-2018_합계기준_비중_pct"),
        on=["지역", SUBAREA],
        validate="one_to_one",
    ).merge(
        _regional_period_share(shares, (2022, 2023, 2024), "2022-2024_합계기준_비중_pct"),
        on=["지역", SUBAREA],
        validate="one_to_one",
    )
    result["2016→2024_비중변화_pp"] = result["2024_비중_pct"] - result["2016_비중_pct"]
    result["3개년구간_비중변화_pp"] = (
        result["2022-2024_합계기준_비중_pct"] - result["2016-2018_합계기준_비중_pct"]
    )
    quality = shares.groupby(["지역", SUBAREA], as_index=False)["예산누락주의"].any()
    return result.merge(quality, on=["지역", SUBAREA], validate="one_to_one")


def _national_annual_shares(shares: pd.DataFrame) -> pd.DataFrame:
    annual = shares.groupby(["연도", SUBAREA], as_index=False)[BUDGET].sum()
    annual["비중_pct"] = annual[BUDGET] / annual.groupby("연도")[BUDGET].transform("sum") * 100
    return annual


def plot_national_share_change(summary: pd.DataFrame) -> plt.Figure:
    """전국 합계 기준 2016→2024년 세부영역별 예산비중 증감을 그린다."""
    plot = summary.sort_values("2016→2024_비중변화_pp")
    values = plot["2016→2024_비중변화_pp"]
    colors = np.where(values.ge(0), YOMOCHA_WEB_COLORS["accent"], "#D97706")
    fig, ax = plt.subplots(figsize=(11, 7.2))
    bars = ax.barh(plot[SUBAREA], values, color=colors)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.1f}pp" for value in values], padding=4, fontsize=9)
    limit = float(values.abs().max() * 1.18)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("2024년 비중 - 2016년 비중(%p)")
    ax.set_ylabel("세부영역")
    ax.grid(axis="y", visible=False)
    fig.suptitle("2016~2024년 저출생 대응 계획예산의 세부영역별 비중 변화", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "17개 시도 계획예산 합계 기준·지표체계 외 예산 포함·파란색은 비중 증가, 주황색은 감소",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_regional_change_heatmap(regional: pd.DataFrame) -> plt.Figure:
    """시도×세부영역의 2016→2024년 예산비중 변화를 히트맵으로 그린다."""
    matrix = regional.pivot(index="지역", columns=SUBAREA, values="2016→2024_비중변화_pp")
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    cmap = sns.diverging_palette(30, 240, s=85, l=55, as_cmap=True)
    fig, ax = plt.subplots(figsize=(17, 9))
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "2016→2024년 비중 변화(%p)"},
        ax=ax,
    )
    ax.set_xlabel("세부영역")
    ax.set_ylabel("시도")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.suptitle("17개 시도별 저출생 대응 계획예산 비중 변화", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.95,
        "각 시도의 2024년 세부영역 비중 - 2016년 비중·지표체계 외 예산 포함",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_national_share_trends(shares: pd.DataFrame) -> plt.Figure:
    """세부영역별 전국 합계 예산비중의 9개년 추세를 소다중 그림으로 그린다."""
    annual = _national_annual_shares(shares)
    subareas = list(dict.fromkeys(annual[SUBAREA]))
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    for ax, subarea in zip(axes.flat, subareas, strict=True):
        rows = annual.loc[annual[SUBAREA].eq(subarea)]
        ax.plot(
            rows["연도"],
            rows["비중_pct"],
            color=YOMOCHA_WEB_COLORS["accent"],
            marker="o",
            linewidth=1.8,
            markersize=4,
        )
        ax.axhline(rows.iloc[0]["비중_pct"], color=YOMOCHA_WEB_COLORS["line"], linestyle="--")
        ax.set_title(subarea)
        ax.set_ylabel("비중(%)")
        ax.set_xticks([2016, 2018, 2020, 2022, 2024])
        ax.grid(axis="x", visible=False)
    fig.suptitle("2016~2024년 세부영역별 저출생 대응 계획예산 비중 추세", y=0.995, fontsize=16)
    fig.text(
        0.5,
        0.968,
        "17개 시도 계획예산 합계 기준·점선은 2016년 비중·단년도 급변은 사업 통합·분리·기록 변경의 영향 가능",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shares = build_budget_share_panel(pd.read_csv(args.input))
    national = build_national_change_summary(shares)
    regional = build_regional_change_table(shares)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shares.to_csv(
        args.output_dir / "2016-2024_지역연도_세부영역별_계획예산비중.csv",
        index=False,
        encoding="utf-8-sig",
    )
    national.to_csv(
        args.output_dir / "2016-2024_세부영역별_계획예산비중_변화요약.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regional.to_csv(
        args.output_dir / "2016-2024_시도별_세부영역_계획예산비중_변화.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figures = {
        "2016-2024_세부영역별_계획예산비중_변화": plot_national_share_change(national),
        "2016-2024_시도별_세부영역_계획예산비중_변화": plot_regional_change_heatmap(regional),
        "2016-2024_세부영역별_계획예산비중_추세": plot_national_share_trends(shares),
    }
    for name, figure in figures.items():
        save_figure(figure, args.figure_dir / name)
        plt.close(figure)
    print(national.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
