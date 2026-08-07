"""#103 — 구조환경 유형별 재정대응예산·TFR 분석 결과를 시각화한다."""

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
DEFAULT_ASSIGNMENTS = ANALYSIS_DIR / "2016-2024_시도별_구조환경_군집.csv"
DEFAULT_CENTERS = ANALYSIS_DIR / "2016-2024_구조환경_군집중심.csv"
DEFAULT_STABILITY = ANALYSIS_DIR / "2016-2024_구조환경_군집_안정성.csv"
DEFAULT_DIAGNOSTICS = ANALYSIS_DIR / "2016-2024_구조환경_군집수_진단.csv"
DEFAULT_SAMPLE = ANALYSIS_DIR / "2016-2024_세부영역별_3개년평균예산_TFR_회귀표본.csv"
DEFAULT_SUBGROUP = ANALYSIS_DIR / "2016-2024_구조환경_2개군집별_3개년평균예산_TFR_결과.csv"
DEFAULT_INTERACTION = ANALYSIS_DIR / "2016-2024_구조환경군집_예산상호작용_TFR_결과.csv"
DEFAULT_SUBGROUP_THREE = ANALYSIS_DIR / "2016-2024_구조환경_3개군집별_3개년평균예산_TFR_결과.csv"
DEFAULT_INTERACTION_THREE = ANALYSIS_DIR / "2016-2024_구조환경_3개군집_예산상호작용_TFR_계수.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/figures/cluster_fiscal_tfr"

CLUSTER_COLORS = {1: YOMOCHA_WEB_COLORS["accent"], 2: "#0E7490"}
CLUSTER_LABELS = {
    1: "군집 1: 가족·생활·사회·문화 낮은 유형",
    2: "군집 2: 가족·생활·사회·문화 높은 유형",
}


def validate_inputs(assignments: pd.DataFrame, centers: pd.DataFrame) -> None:
    """군집 배정과 중심값의 필수 구조를 검증한다."""
    required_assignments = {"region", "군집_2개", "군집_3개"}
    required_centers = {"군집수", "군집", "가족·생활", "경제·고용·주거", "보건·안전", "사회·문화"}
    if missing := required_assignments - set(assignments.columns):
        raise KeyError(f"군집 배정 필수 컬럼 누락: {sorted(missing)}")
    if missing := required_centers - set(centers.columns):
        raise KeyError(f"군집 중심 필수 컬럼 누락: {sorted(missing)}")
    if len(assignments) != 17 or assignments["region"].nunique() != 17:
        raise ValueError("군집 배정은 중복 없는 17개 시도여야 합니다.")


def plot_cluster_profile(centers: pd.DataFrame) -> plt.Figure:
    """2개 군집의 4개 대영역 구조환경 중심값을 비교한다."""
    fields = ["가족·생활", "경제·고용·주거", "보건·안전", "사회·문화"]
    plot = centers.loc[centers["군집수"].eq(2)].set_index("군집")
    x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for cluster in (1, 2):
        ax.plot(
            x,
            plot.loc[cluster, fields],
            marker="o",
            linewidth=2.2,
            color=CLUSTER_COLORS[cluster],
            label=CLUSTER_LABELS[cluster],
        )
    ax.set_xticks(x, fields)
    ax.set_ylabel("2016~2024년 평균 구조환경지수(0~100)")
    ax.set_ylim(20, 65)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)
    fig.suptitle("구조환경 2개 군집의 대영역별 프로파일", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.935,
        "17개 시도의 2016~2024년 대영역별 평균을 표준화해 K-means로 분류",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def plot_cluster_membership(assignments: pd.DataFrame) -> plt.Figure:
    """2개·3개 군집 배정을 시도별 타일로 비교한다."""
    plot = assignments.sort_values(["군집_2개", "region"]).reset_index(drop=True)
    values = plot[["군집_2개", "군집_3개"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    cmap = sns.color_palette([YOMOCHA_WEB_COLORS["accent"], "#0E7490", "#7C3AED"], as_cmap=True)
    sns.heatmap(
        values,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        xticklabels=["2개 군집", "3개 군집(민감도)"],
        yticklabels=plot["region"],
        ax=ax,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("시도")
    fig.suptitle("군집 수에 따른 17개 시도 유형 배정", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.955,
        "3개 군집에서 대전·세종이 별도 유형으로 분리되지만 표본이 2개에 그쳐 기술통계로만 활용",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def build_cluster_trends(sample: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """군집별 연도 평균 3개년 예산과 TFR을 구성한다."""
    mapping = assignments[["region", "군집_2개"]].rename(columns={"region": "지역"})
    merged = sample.merge(mapping, on="지역", how="left", validate="many_to_one")
    if merged["군집_2개"].isna().any():
        raise ValueError("회귀표본에 군집 배정이 없는 지역이 있습니다.")
    expected_subareas = sample["세부영역"].nunique()
    observed_subareas = merged.groupby(["지역", "연도"])["인구1인당_실질예산_3개년평균"].transform(
        "count"
    )
    valid_budget = merged.loc[observed_subareas.eq(expected_subareas)].copy()
    region_year = valid_budget.groupby(["지역", "연도", "군집_2개"], as_index=False).agg(
        실질_1인당_3개년평균예산=("인구1인당_실질예산_3개년평균", "sum"),
        합계출산율=("합계출산율", "first"),
    )
    return region_year.groupby(["군집_2개", "연도"], as_index=False).agg(
        실질_1인당_3개년평균예산=("실질_1인당_3개년평균예산", "mean"),
        합계출산율=("합계출산율", "mean"),
    )


def build_three_cluster_lagged_trends(
    sample: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """3개 군집별 완전한 예산 합계와 t+1·t+2 TFR 평균을 구성한다."""
    mapping = assignments[["region", "군집_3개"]].rename(columns={"region": "지역"})
    merged = sample.merge(mapping, on="지역", how="left", validate="many_to_one")
    if merged["군집_3개"].isna().any():
        raise ValueError("회귀표본에 3개 군집 배정이 없는 지역이 있습니다.")
    expected_subareas = sample["세부영역"].nunique()
    observed_subareas = merged.groupby(["지역", "연도"])["인구1인당_실질예산_3개년평균"].transform(
        "count"
    )
    valid = merged.loc[observed_subareas.eq(expected_subareas)].copy()
    region_year = valid.groupby(["지역", "연도", "군집_3개"], as_index=False).agg(
        실질_1인당_3개년평균예산=("인구1인당_실질예산_3개년평균", "sum"),
        합계출산율_t1=("합계출산율_t+1", "first"),
        합계출산율_t2=("합계출산율_t+2", "first"),
    )
    return region_year.groupby(["군집_3개", "연도"], as_index=False).agg(
        실질_1인당_3개년평균예산=("실질_1인당_3개년평균예산", "mean"),
        합계출산율_t1=("합계출산율_t1", "mean"),
        합계출산율_t2=("합계출산율_t2", "mean"),
        지역수=("지역", "nunique"),
    )


def plot_cluster_trends(trends: pd.DataFrame) -> plt.Figure:
    """군집별 3개년 예산과 TFR 추이를 나란히 비교한다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    specs = [
        ("실질_1인당_3개년평균예산", "실질 1인당 예산(3개년 평균, 원)", "재정대응예산"),
        ("합계출산율", "합계출산율", "합계출산율"),
    ]
    for ax, (column, ylabel, title) in zip(axes, specs, strict=True):
        for cluster in (1, 2):
            rows = trends.loc[trends["군집_2개"].eq(cluster)]
            ax.plot(
                rows["연도"],
                rows[column],
                marker="o",
                linewidth=2,
                color=CLUSTER_COLORS[cluster],
                label=CLUSTER_LABELS[cluster],
            )
        ax.set_title(title)
        ax.set_xlabel("연도")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False
    )
    fig.suptitle("구조환경 유형별 재정대응예산과 TFR 추이", y=0.995, fontsize=16)
    fig.text(
        0.5,
        0.947,
        "예산은 11개 세부영역 합계의 시도별 값을 군집 내 평균·TFR은 동일 연도 기술통계",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def plot_interaction_coefficients(interaction: pd.DataFrame) -> plt.Figure:
    """t+1·t+2 군집별 예산 계수와 신뢰구간을 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, lag in zip(axes, ("t+1", "t+2"), strict=True):
        rows = interaction.loc[interaction["시차"].eq(lag)].reset_index(drop=True)
        labels = rows["세부영역"].tolist()
        y = np.arange(len(rows))
        for cluster, offset in ((1, -0.12), (2, 0.12)):
            coefficient = rows[f"군집{cluster}_계수"]
            lower = rows[f"군집{cluster}_95%신뢰구간_하한"]
            upper = rows[f"군집{cluster}_95%신뢰구간_상한"]
            ax.errorbar(
                coefficient,
                y + offset,
                xerr=np.vstack([coefficient - lower, upper - coefficient]),
                fmt="o",
                color=CLUSTER_COLORS[cluster],
                ecolor=CLUSTER_COLORS[cluster],
                capsize=2.5,
                markersize=5,
                label=CLUSTER_LABELS[cluster],
            )
        ax.axvline(0, color="#374151", linestyle="--", linewidth=1)
        ax.set_title(f"{lag} TFR")
        ax.set_xlabel("log1p(실질 1인당 예산 3개년 평균) 계수")
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.suptitle("구조환경 유형별 3개년 평균예산과 후행 TFR의 관계", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.947,
        "점은 군집별 계수, 선은 95% 신뢰구간·시도·연도 고정효과 모형",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_cluster_stability(stability: pd.DataFrame) -> plt.Figure:
    """2개·3개 군집의 분리도와 안정성을 비교한다."""
    metrics = ["실루엣점수", "계층군집_ARI", "시도제외_ARI_최소", "시도제외_ARI_평균"]
    labels = ["실루엣", "계층군집 일치", "1개 시도 제외 최소", "1개 시도 제외 평균"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, cluster_count, color in (
        (-width / 2, 2, CLUSTER_COLORS[1]),
        (width / 2, 3, "#7C3AED"),
    ):
        row = stability.loc[stability["군집수"].eq(cluster_count)].iloc[0]
        ax.bar(x + offset, row[metrics], width, color=color, label=f"{cluster_count}개 군집")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("지표값(0~1)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.suptitle("2개·3개 구조환경 군집의 안정성 비교", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.94,
        "3개 군집은 분리도가 높지만 최소 군집이 2개 시도에 그쳐 회귀분석에서 제외",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_elbow_diagnostics(diagnostics: pd.DataFrame) -> plt.Figure:
    """k=1~6 WCSS 엘보우와 k=2~6 실루엣 점수를 함께 표시한다."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(
        diagnostics["군집수"],
        diagnostics["WCSS"],
        marker="o",
        linewidth=2.2,
        color=YOMOCHA_WEB_COLORS["accent"],
        label="WCSS(왼쪽 축)",
    )
    ax.set_xlabel("군집 수(k)")
    ax.set_ylabel("군집 내 제곱합(WCSS)", color=YOMOCHA_WEB_COLORS["accent"])
    ax.set_xticks(diagnostics["군집수"])
    ax.tick_params(axis="y", labelcolor=YOMOCHA_WEB_COLORS["accent"])
    ax.grid(axis="x", visible=False)

    second = ax.twinx()
    silhouette = diagnostics.dropna(subset=["실루엣점수"])
    second.plot(
        silhouette["군집수"],
        silhouette["실루엣점수"],
        marker="s",
        linestyle="--",
        linewidth=2,
        color="#0E7490",
        label="실루엣(오른쪽 축)",
    )
    second.set_ylabel("실루엣 점수", color="#0E7490")
    second.tick_params(axis="y", labelcolor="#0E7490")
    second.set_ylim(0, max(0.5, silhouette["실루엣점수"].max() + 0.05))

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    ax.axvspan(1.85, 3.15, color=YOMOCHA_WEB_COLORS["surface_alt"], alpha=0.45, zorder=0)
    fig.suptitle("구조환경 군집 수 선택 진단", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "WCSS 감소는 k=2~3에서 완만해지고, 실루엣은 k=3에서 최고·k≥4는 1개 시도 군집 발생",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def plot_three_cluster_coefficients(coefficients: pd.DataFrame) -> plt.Figure:
    """3개 군집 상호작용 모형의 군집별 계수를 표시한다."""
    colors = {1: YOMOCHA_WEB_COLORS["accent"], 2: "#7C3AED", 3: "#0E7490"}
    labels = {
        1: "군집 1(7개 시도)",
        2: "군집 2(대전·세종, 추론 불안정)",
        3: "군집 3(8개 시도)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    for ax, lag in zip(axes, ("t+1", "t+2"), strict=True):
        lag_rows = coefficients.loc[coefficients["시차"].eq(lag)]
        subareas = list(dict.fromkeys(lag_rows["세부영역"]))
        y = np.arange(len(subareas))
        for cluster, offset in ((1, -0.18), (2, 0.0), (3, 0.18)):
            rows = lag_rows.loc[lag_rows["군집"].eq(cluster)].set_index("세부영역").loc[subareas]
            coefficient = rows["계수"]
            lower = rows["95%신뢰구간_하한"]
            upper = rows["95%신뢰구간_상한"]
            ax.errorbar(
                coefficient,
                y + offset,
                xerr=np.vstack([coefficient - lower, upper - coefficient]),
                fmt="o" if cluster != 2 else "o",
                markerfacecolor=colors[cluster] if cluster != 2 else "white",
                markeredgecolor=colors[cluster],
                color=colors[cluster],
                ecolor=colors[cluster],
                capsize=2,
                markersize=5,
                linestyle="none",
                label=labels[cluster],
            )
        ax.axvline(0, color="#374151", linestyle="--", linewidth=1)
        ax.set_title(f"{lag} TFR")
        ax.set_xlabel("log1p(실질 1인당 예산 3개년 평균) 계수")
        ax.set_yticks(y, subareas)
        ax.grid(axis="y", visible=False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("3개 구조환경 군집별 3개년 평균예산과 후행 TFR", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.947,
        "전체 17개 시도 상호작용 모형·군집 2는 2개 시도뿐이라 큰 계수와 유의성을 확정적으로 해석하지 않음",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    return fig


def plot_daejeon_sejong_exploratory_coefficients(subgroup_three: pd.DataFrame) -> plt.Figure:
    """대전·세종 군집의 t+1·t+2 고정효과 계수 방향을 비교한다."""
    plot = subgroup_three.loc[subgroup_three["군집"].eq(2)].copy()
    expected_lags = {"t+1", "t+2"}
    if set(plot["시차"].unique()) != expected_lags or len(plot) != 22:
        raise ValueError("대전·세종 탐색 시각화는 t+1·t+2별 11개 세부영역이 필요합니다.")
    if plot["p값"].notna().any() or plot["계수"].isna().any():
        raise ValueError("대전·세종 탐색결과는 계수만 있고 p값은 비어 있어야 합니다.")

    order = list(dict.fromkeys(plot["모형"]))
    wide = plot.pivot(index="모형", columns="시차", values="계수").loc[order]
    y = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    for position, (lag1, lag2) in enumerate(wide[["t+1", "t+2"]].to_numpy()):
        ax.plot(
            [lag1, lag2],
            [position, position],
            color=YOMOCHA_WEB_COLORS["line"],
            linewidth=1.4,
            zorder=1,
        )
    ax.scatter(
        wide["t+1"],
        y,
        s=52,
        color=YOMOCHA_WEB_COLORS["accent"],
        label="t+1 TFR 계수(N=12)",
        zorder=3,
    )
    ax.scatter(
        wide["t+2"],
        y,
        s=58,
        facecolor="white",
        edgecolor="#0E7490",
        linewidth=1.8,
        label="t+2 TFR 계수(N=10)",
        zorder=3,
    )
    ax.axvline(0, color="#374151", linestyle="--", linewidth=1.1)
    ax.set_yticks(y, wide.index)
    ax.invert_yaxis()
    ax.set_xlabel("log1p(실질 1인당 예산 3개년 평균) 계수")
    ax.set_ylabel("세부영역")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False)
    fig.suptitle("대전·세종의 3개년 평균예산과 후행 TFR 탐색계수", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.945,
        "2개 시도 지역·연도 고정효과 점추정치·선은 동일 영역의 t+1·t+2를 연결·p값과 신뢰구간은 해석하지 않음",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def plot_three_cluster_subgroup_coefficients(subgroup_three: pd.DataFrame) -> plt.Figure:
    """3개 군집의 부분표본 계수를 추론 가능 여부와 함께 비교한다."""
    required = {"시차", "군집", "모형", "계수", "95%신뢰구간_하한", "95%신뢰구간_상한"}
    if missing := required - set(subgroup_three.columns):
        raise KeyError(f"3개 군집 부분표본 시각화 필수 컬럼 누락: {sorted(missing)}")
    expected = {(lag, cluster) for lag in ("t+1", "t+2") for cluster in (1, 2, 3)}
    observed = set(zip(subgroup_three["시차"], subgroup_three["군집"], strict=False))
    if not expected.issubset(observed):
        raise ValueError("t+1·t+2와 3개 군집의 부분표본 결과가 모두 필요합니다.")

    colors = {1: YOMOCHA_WEB_COLORS["accent"], 2: "#7C3AED", 3: "#0E7490"}
    markers = {1: "o", 2: "D", 3: "s"}
    labels = {
        1: "군집 1(7개 시도·95% CI)",
        2: "군집 2(대전·세종·계수만)",
        3: "군집 3(8개 시도·95% CI)",
    }
    offsets = {1: -0.19, 2: 0.0, 3: 0.19}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.7), sharex=True, sharey=True)
    for ax, lag in zip(axes, ("t+1", "t+2"), strict=True):
        lag_rows = subgroup_three.loc[subgroup_three["시차"].eq(lag)]
        subareas = list(dict.fromkeys(lag_rows["모형"]))
        y = np.arange(len(subareas))
        for cluster in (1, 2, 3):
            rows = lag_rows.loc[lag_rows["군집"].eq(cluster)].set_index("모형").loc[subareas]
            coefficient = rows["계수"]
            if cluster == 2:
                ax.scatter(
                    coefficient,
                    y + offsets[cluster],
                    marker=markers[cluster],
                    s=50,
                    facecolor="white",
                    edgecolor=colors[cluster],
                    linewidth=1.7,
                    label=labels[cluster],
                    zorder=3,
                )
            else:
                lower = rows["95%신뢰구간_하한"]
                upper = rows["95%신뢰구간_상한"]
                ax.errorbar(
                    coefficient,
                    y + offsets[cluster],
                    xerr=np.vstack([coefficient - lower, upper - coefficient]),
                    fmt=markers[cluster],
                    color=colors[cluster],
                    ecolor=colors[cluster],
                    capsize=2.5,
                    markersize=5,
                    linestyle="none",
                    label=labels[cluster],
                )
        ax.axvline(0, color="#374151", linestyle="--", linewidth=1.1)
        ax.set_title(f"{lag} TFR")
        ax.set_xlabel("log1p(실질 1인당 예산 3개년 평균) 계수")
        ax.set_yticks(y, subareas)
        ax.grid(axis="y", visible=False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("3개 구조환경 군집별 부분표본 예산·TFR 계수", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.947,
        "군집 1·3은 점과 95% 신뢰구간·군집 2(대전·세종)는 독립 지역 2개로 계수점만 탐색적 표시",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    return fig


def plot_three_cluster_lagged_trends(trends: pd.DataFrame) -> plt.Figure:
    """3개 군집별 3개년 평균예산과 후행 TFR 추세를 비교한다."""
    colors = {1: YOMOCHA_WEB_COLORS["accent"], 2: "#7C3AED", 3: "#0E7490"}
    markers = {1: "o", 2: "D", 3: "s"}
    linestyles = {1: "-", 2: "--", 3: "-."}
    labels = {1: "군집 1(7개 시도)", 2: "군집 2(대전·세종)", 3: "군집 3(8개 시도)"}
    specs = [
        ("실질_1인당_3개년평균예산", "실질 1인당 예산(원)", "3개년 평균예산"),
        ("합계출산율_t1", "합계출산율", "t+1 TFR"),
        ("합계출산율_t2", "합계출산율", "t+2 TFR"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, (column, ylabel, title) in zip(axes, specs, strict=True):
        for cluster in (1, 2, 3):
            rows = trends.loc[trends["군집_3개"].eq(cluster)].dropna(subset=[column])
            ax.plot(
                rows["연도"],
                rows[column],
                color=colors[cluster],
                marker=markers[cluster],
                linestyle=linestyles[cluster],
                linewidth=2,
                markersize=5,
                label=labels[cluster],
            )
        ax.set_title(title)
        ax.set_xlabel("3개년 예산창 마지막 연도(t)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(trends.loc[trends[column].notna(), "연도"].unique()))
        ax.grid(axis="x", visible=False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("3개 구조환경 군집별 재정대응예산과 후행 TFR 추세", y=0.99, fontsize=16)
    fig.text(
        0.5,
        0.947,
        "예산은 11개 세부영역 합계의 군집별 평균·TFR은 예산창 마지막 연도보다 1년·2년 후의 군집별 평균·기술통계로만 해석",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--centers", type=Path, default=DEFAULT_CENTERS)
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--subgroup", type=Path, default=DEFAULT_SUBGROUP)
    parser.add_argument("--interaction", type=Path, default=DEFAULT_INTERACTION)
    parser.add_argument("--subgroup-three", type=Path, default=DEFAULT_SUBGROUP_THREE)
    parser.add_argument("--interaction-three", type=Path, default=DEFAULT_INTERACTION_THREE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments = pd.read_csv(args.assignments)
    centers = pd.read_csv(args.centers)
    stability = pd.read_csv(args.stability)
    diagnostics = pd.read_csv(args.diagnostics)
    sample = pd.read_csv(args.sample)
    interaction = pd.read_csv(args.interaction)
    subgroup_three = pd.read_csv(args.subgroup_three)
    interaction_three = pd.read_csv(args.interaction_three)
    # 군집별 세부 회귀표는 보고서 정확값 표의 입력으로도 사용하므로 생성 여부를 확인한다.
    pd.read_csv(args.subgroup)
    validate_inputs(assignments, centers)
    trends = build_cluster_trends(sample, assignments)
    trends_three = build_three_cluster_lagged_trends(sample, assignments)

    figures = {
        "2016-2024_구조환경_2개군집_프로파일": plot_cluster_profile(centers),
        "2016-2024_구조환경_2개_3개군집_시도배정": plot_cluster_membership(assignments),
        "2016-2024_구조환경_유형별_예산_TFR_추이": plot_cluster_trends(trends),
        "2016-2024_구조환경_유형별_예산_TFR_계수": plot_interaction_coefficients(interaction),
        "2016-2024_구조환경_2개_3개군집_안정성": plot_cluster_stability(stability),
        "2016-2024_구조환경_군집수_엘보우_실루엣": plot_elbow_diagnostics(diagnostics),
        "2016-2024_구조환경_3개군집_예산_TFR_계수": plot_three_cluster_coefficients(
            interaction_three
        ),
        "2016-2024_대전_세종_3개년평균예산_TFR_탐색계수": (
            plot_daejeon_sejong_exploratory_coefficients(subgroup_three)
        ),
        "2016-2024_구조환경_3개군집별_부분표본_예산_TFR_계수": (
            plot_three_cluster_subgroup_coefficients(subgroup_three)
        ),
        "2016-2024_구조환경_3개군집별_재정대응예산_후행_TFR_추세": (
            plot_three_cluster_lagged_trends(trends_three)
        ),
    }
    for name, figure in figures.items():
        save_figure(figure, args.output_dir / name)
        plt.close(figure)
    trends.to_csv(
        ANALYSIS_DIR / "2018-2024_구조환경_유형별_예산_TFR_추이.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trends_three.to_csv(
        ANALYSIS_DIR / "2018-2024_구조환경_3개군집별_재정대응예산_후행_TFR_추세.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
