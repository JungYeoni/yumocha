"""#114 구조환경 하락 사례의 후행 예산비중 증가 현황을 시각화한다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.plots import YOMOCHA_WEB_COLORS, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_SAMPLE = ANALYSIS_DIR / "2016-2024_구조환경변화_후행예산비중변화_표본.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/figures/decline_budget_share_response"
STRUCTURAL_CHANGE = "구조환경지수_변화_t_t1"
BUDGET_SHARE_CHANGE = "계획예산비중_변화_t2_t3_pp"

np.random.seed(42)


def subarea_sort_key(label: object) -> tuple[int, int, str]:
    """`대영역-세부영역` 번호 순으로 세부영역을 정렬한다."""
    text = str(label)
    match = re.match(r"(\d+)-(\d+)\.", text)
    return (int(match.group(1)), int(match.group(2)), text) if match else (99, 99, text)


def validate_sample(sample: pd.DataFrame) -> None:
    """공식 17개 시도×6개 기준연도×11개 세부영역 표본을 검증한다."""
    required = {
        "지역",
        "기준연도",
        "세부영역",
        STRUCTURAL_CHANGE,
        BUDGET_SHARE_CHANGE,
        "예산누락주의_두연도",
    }
    if missing := required - set(sample.columns):
        raise ValueError(f"구조환경 하락 시각화 필수 컬럼 누락: {sorted(missing)}")
    if sample.duplicated(["지역", "기준연도", "세부영역"]).any():
        raise ValueError("지역×기준연도×세부영역 중복이 있습니다.")
    if sample["지역"].nunique() != 17 or sample["세부영역"].nunique() != 11:
        raise ValueError("공식 표본은 17개 시도×11개 세부영역이어야 합니다.")
    if set(sample["기준연도"].unique()) != set(range(2016, 2022)) or len(sample) != 1122:
        raise ValueError("공식 표본은 2016~2021년 기준 1,122행이어야 합니다.")
    numeric = sample[[STRUCTURAL_CHANGE, BUDGET_SHARE_CHANGE]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("변화량에 결측 또는 비유한 값이 있습니다.")


def build_decline_events(sample: pd.DataFrame) -> pd.DataFrame:
    """구조환경이 하락한 관측치만 남기고 후행 예산비중 증가 여부를 표시한다."""
    # #108의 전체 변화량 표본은 재사용하되, `src/features/`·`notebooks/`·
    # `reports/`에는 하락 사건만의 후행 증가 여부를 분류한 기존 구현이 없어 여기서 파생한다.
    decline = sample.loc[sample[STRUCTURAL_CHANGE].lt(0)].copy()
    decline["지역"] = pd.Categorical(
        decline["지역"],
        categories=list(dict.fromkeys(sample["지역"])),
        ordered=True,
    )
    decline["세부영역"] = pd.Categorical(
        decline["세부영역"],
        categories=list(dict.fromkeys(sample["세부영역"])),
        ordered=True,
    )
    decline["후행예산비중_증가"] = decline[BUDGET_SHARE_CHANGE].gt(0)
    decline["대응방향"] = np.where(
        decline["후행예산비중_증가"], "하락 후 비중 증가", "하락 후 비중 비증가"
    )
    return decline


def summarize_by_subarea(decline: pd.DataFrame) -> pd.DataFrame:
    """세부영역별 하락 사례와 후행 예산비중 증가 건수·비율을 요약한다."""
    summary = (
        decline.groupby("세부영역", as_index=False, sort=False, observed=False)
        .agg(
            구조환경_하락사례수=(STRUCTURAL_CHANGE, "size"),
            후행예산비중_증가건수=("후행예산비중_증가", "sum"),
            예산누락주의_사례수=("예산누락주의_두연도", "sum"),
        )
        .assign(
            후행예산비중_증가비율_pct=lambda x: (
                100 * x["후행예산비중_증가건수"] / x["구조환경_하락사례수"]
            )
        )
    )
    clean = decline.loc[~decline["예산누락주의_두연도"].astype(bool)]
    clean_summary = clean.groupby("세부영역", observed=False).agg(
        누락주의제외_하락사례수=(STRUCTURAL_CHANGE, "size"),
        누락주의제외_증가건수=("후행예산비중_증가", "sum"),
    )
    summary = summary.join(clean_summary, on="세부영역")
    summary["누락주의제외_증가비율_pct"] = (
        100
        * summary["누락주의제외_증가건수"]
        / summary["누락주의제외_하락사례수"].replace(0, np.nan)
    )
    return summary


def summarize_region_subarea(decline: pd.DataFrame) -> pd.DataFrame:
    """시도×세부영역별 하락 사례의 후행 예산비중 증가 비율을 요약한다."""
    return (
        decline.groupby(["지역", "세부영역"], as_index=False, sort=False, observed=False)
        .agg(
            구조환경_하락사례수=(STRUCTURAL_CHANGE, "size"),
            후행예산비중_증가건수=("후행예산비중_증가", "sum"),
            예산누락주의_사례수=("예산누락주의_두연도", "sum"),
        )
        .assign(
            후행예산비중_증가비율_pct=lambda x: (
                100 * x["후행예산비중_증가건수"] / x["구조환경_하락사례수"]
            )
        )
    )


def plot_subarea_response(summary: pd.DataFrame) -> plt.Figure:
    """세부영역별 구조환경 하락 후 예산비중 증가 비율을 비교한다."""
    plot = summary.sort_values("후행예산비중_증가비율_pct", ascending=True).reset_index(drop=True)
    positions = np.arange(len(plot))
    valid = plot["구조환경_하락사례수"].gt(0)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    bars = ax.barh(
        positions[valid],
        plot.loc[valid, "후행예산비중_증가비율_pct"],
        color=YOMOCHA_WEB_COLORS["accent"],
    )
    for bar, (_, row) in zip(bars, plot.loc[valid].iterrows(), strict=True):
        ax.text(
            min(bar.get_width() + 1.2, 98),
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['후행예산비중_증가건수'])}/{int(row['구조환경_하락사례수'])}",
            va="center",
            fontsize=9,
        )
    for position in positions[~valid]:
        ax.text(1.2, position, "— (0/0)", va="center", fontsize=9, color="#6B7280")
    ax.set_yticks(positions, labels=plot["세부영역"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("구조환경 하락 사례 중 후행 예산비중 증가 비율(%)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    fig.suptitle("세부영역별 구조환경 하락 후 예산비중 증가 비율", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "구조환경 t→t+1 하락 사례만 집계 · 예산비중 t+2→t+3 증가 건수/하락 사례 수",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_region_subarea_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """시도×세부영역별 증가 비율과 증가/하락 사례 수를 히트맵으로 표시한다."""
    regions = sorted(summary["지역"].unique())
    subareas = sorted(summary["세부영역"].unique(), key=subarea_sort_key)
    rate = summary.pivot(index="지역", columns="세부영역", values="후행예산비중_증가비율_pct")
    rate = rate.reindex(index=regions, columns=subareas)
    numerator = summary.pivot(
        index="지역", columns="세부영역", values="후행예산비중_증가건수"
    ).reindex(index=regions, columns=subareas)
    denominator = summary.pivot(
        index="지역", columns="세부영역", values="구조환경_하락사례수"
    ).reindex(index=regions, columns=subareas)
    annotations = np.empty(rate.shape, dtype=object)
    for row in range(rate.shape[0]):
        for column in range(rate.shape[1]):
            if pd.isna(rate.iloc[row, column]):
                annotations[row, column] = "—"
            else:
                annotations[row, column] = (
                    f"{int(numerator.iloc[row, column])}/{int(denominator.iloc[row, column])}\n"
                    f"{rate.iloc[row, column]:.0f}%"
                )
    fig, ax = plt.subplots(figsize=(18, 11.5))
    sns.heatmap(
        rate,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette(YOMOCHA_WEB_COLORS["accent"], as_cmap=True),
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "후행 예산비중 증가 비율(%)"},
        ax=ax,
    )
    ax.set_xlabel("세부영역")
    ax.set_ylabel("시도")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.suptitle("시도·세부영역별 구조환경 하락 후 예산비중 증가 현황", y=0.995, fontsize=16)
    fig.text(
        0.5,
        0.962,
        "셀: 증가 건수/구조환경 하락 사례 수와 비율 · 사례가 없는 셀은 —",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    validate_sample(sample)
    decline = build_decline_events(sample)
    subarea = summarize_by_subarea(decline)
    region_subarea = summarize_region_subarea(decline)
    figures = {
        "2016-2024_세부영역별_구조환경하락_후행예산비중증가비율": plot_subarea_response(subarea),
        "2016-2024_시도세부영역별_구조환경하락_후행예산비중증가_히트맵": (
            plot_region_subarea_heatmap(region_subarea)
        ),
    }
    for name, figure in figures.items():
        save_figure(figure, args.output_dir / name)
        plt.close(figure)
    decline.to_csv(
        ANALYSIS_DIR / "2016-2024_구조환경하락_후행예산비중증가_사례.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subarea.to_csv(
        ANALYSIS_DIR / "2016-2024_세부영역별_구조환경하락_후행예산비중증가_요약.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_subarea.to_csv(
        ANALYSIS_DIR / "2016-2024_시도세부영역별_구조환경하락_후행예산비중증가_요약.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
