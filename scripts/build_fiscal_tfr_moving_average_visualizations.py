"""#98 3개년 평균 돌봄예산과 t+2 TFR 핵심 결과의 검증 시각화를 만든다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects
from src.visualization.plots import YOMOCHA_WEB_COLORS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = (
    REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_3개년평균예산_TFR_회귀표본.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/재정대응_TFR_3개년평균"
SUBAREA = "2-1. 돌봄 여건"
OUTCOME = "합계출산율_t+2"
BUDGET = "인구1인당_실질예산_3개년평균"
PREDICTOR = "log1p_3개년평균_돌봄예산"


def prepare_care_sample(data: pd.DataFrame) -> pd.DataFrame:
    required = {"지역", "연도", "세부영역", OUTCOME, BUDGET}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"시각화 필수 컬럼 누락: {missing}")
    sample = data.loc[data["세부영역"].eq(SUBAREA)].dropna(subset=[OUTCOME, BUDGET]).copy()
    if sample.duplicated(["지역", "연도"]).any():
        raise ValueError("돌봄 시각화 표본의 지역·연도 키 중복")
    sample[PREDICTOR] = np.log1p(sample[BUDGET])
    return sample


def residualize_two_way(sample: pd.DataFrame, column: str) -> pd.Series:
    """값에서 시도·연도 고정효과를 제거한 잔차를 반환한다."""
    fixed = pd.get_dummies(
        sample[["지역", "연도"]].astype({"연도": "string"}),
        columns=["지역", "연도"],
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(fixed, has_constant="add")
    return pd.Series(sm.OLS(sample[column].astype(float), design).fit().resid, index=sample.index)


def leave_one_region_out(sample: pd.DataFrame) -> pd.DataFrame:
    """전체 표본과 시도별 1개 제외 모형의 돌봄 계수·신뢰구간을 산출한다."""
    rows: list[dict[str, object]] = []
    for omitted in ["제외 없음", *sorted(sample["지역"].unique())]:
        subset = sample if omitted == "제외 없음" else sample.loc[sample["지역"].ne(omitted)]
        model, fitted = fit_two_way_fixed_effects(subset, outcome=OUTCOME, predictor=PREDICTOR)
        row = summarize_fixed_effects(
            model,
            fitted,
            model_name=str(omitted),
            outcome=OUTCOME,
            predictor=PREDICTOR,
            excluded_quality_rows=False,
        )
        row["제외시도"] = omitted
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fe_scatter(sample: pd.DataFrame, output: Path) -> None:
    x = residualize_two_way(sample, PREDICTOR)
    y = residualize_two_way(sample, OUTCOME)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x, y, color=YOMOCHA_WEB_COLORS["accent"], alpha=0.68, edgecolor="white")
    ax.plot(grid, intercept + slope * grid, color=YOMOCHA_WEB_COLORS["accent_hover"], lw=2)
    ax.axhline(0, color=YOMOCHA_WEB_COLORS["line"], lw=1)
    ax.axvline(0, color=YOMOCHA_WEB_COLORS["line"], lw=1)
    ax.set_title("3개년 평균 돌봄예산과 2년 후 TFR의 고정효과 제거 관계")
    ax.set_xlabel("log1p(실질 1인당 돌봄예산 3개년 평균): 시도·연도 효과 제거값")
    ax.set_ylabel("2년 후 TFR: 시도·연도 효과 제거값")
    ax.text(0.02, 0.98, f"β={slope:.4f} · N={len(sample)}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_leave_one_out(results: pd.DataFrame, output: Path) -> None:
    ordered = results.sort_values("계수").reset_index(drop=True)
    colors = [
        YOMOCHA_WEB_COLORS["accent"] if name == "제외 없음" else YOMOCHA_WEB_COLORS["muted_bar"]
        for name in ordered["제외시도"]
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(ordered))
    for position, (_, row), color in zip(y, ordered.iterrows(), colors, strict=True):
        ax.errorbar(
            row["계수"],
            position,
            xerr=[[row["계수"] - row["95%신뢰구간_하한"]], [row["95%신뢰구간_상한"] - row["계수"]]],
            fmt="none",
            ecolor=color,
            capsize=3,
        )
    ax.scatter(ordered["계수"], y, c=colors, zorder=3)
    ax.axvline(0, color=YOMOCHA_WEB_COLORS["line"], lw=1)
    ax.set_yticks(y, ordered["제외시도"])
    ax.set_xlabel("돌봄예산 계수와 95% 신뢰구간")
    ax.set_title("시도 하나씩 제외한 3개년 평균 돌봄예산–t+2 TFR 계수")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_trends(sample: pd.DataFrame, output: Path) -> None:
    regions = sorted(sample["지역"].unique())
    fig, axes = plt.subplots(5, 4, figsize=(18, 20), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, regions, strict=False):
        group = sample.loc[sample["지역"].eq(region)].sort_values("연도").copy()
        for source, target in ((PREDICTOR, "예산_z"), (OUTCOME, "TFR_z")):
            std = group[source].std(ddof=0)
            group[target] = (group[source] - group[source].mean()) / std if std else 0.0
        ax.plot(
            group["연도"],
            group["예산_z"],
            marker="o",
            color=YOMOCHA_WEB_COLORS["accent"],
            label="3개년 평균예산",
        )
        ax.plot(
            group["연도"],
            group["TFR_z"],
            marker="s",
            linestyle="--",
            color=YOMOCHA_WEB_COLORS["line"],
            label="t+2 TFR",
        )
        ax.axhline(0, color="#e5e7eb", lw=0.8)
        ax.set_title(region)
    for ax in axes.flat[len(regions) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=2)
    fig.suptitle("시도별 3개년 평균 돌봄예산과 2년 후 TFR의 표준화 추이", y=0.995)
    fig.supxlabel("예산 이동창 마지막 연도(t)")
    fig.supylabel("시도 내 표준화값(z)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.945))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = prepare_care_sample(pd.read_csv(args.sample))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loo = leave_one_region_out(sample)
    loo.to_csv(args.output_dir / "돌봄_t+2_시도제외_민감도.csv", index=False, encoding="utf-8-sig")
    plot_fe_scatter(sample, args.output_dir / "돌봄_t+2_FE제거_산점도.png")
    plot_leave_one_out(loo, args.output_dir / "돌봄_t+2_시도제외_민감도.png")
    plot_region_trends(sample, args.output_dir / "돌봄_t+2_시도별_표준화추이.png")
    print(
        loo[["제외시도", "계수", "p값", "95%신뢰구간_하한", "95%신뢰구간_상한"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
