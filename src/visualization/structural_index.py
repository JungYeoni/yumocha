"""AHP 구조환경지수 결과 검증·요약·시각화 유틸."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.evaluation.structural_index import DEFAULT_STRUCTURAL_REGIONS, DEFAULT_STRUCTURAL_YEARS
from src.visualization.plots import PALETTE, YOMOCHA_WEB_COLORS

INDEX_DIRNAME = "structural_index"
OUTPUT_DIRNAME = "구조환경지수_시각화"

ARTIFACT_FILES = {
    "pooled_final": "structural_index_pooled_final_index.csv",
    "yearly_final": "structural_index_yearly_final_index.csv",
    "pooled_indicator": "structural_index_pooled_indicator_scores.csv",
    "pooled_subcategory": "structural_index_pooled_subcategory_scores.csv",
    "pooled_category": "structural_index_pooled_category_scores.csv",
    "yearly_comparison": "structural_index_pooled_yearly_comparison.csv",
    "family_sensitivity": "structural_index_family_friendly_sensitivity_comparison.csv",
    "family_weight_transfer": "structural_index_family_friendly_weight_transfer_comparison.csv",
}

FINAL_COLUMNS = {
    "region",
    "year",
    "final_index",
    "missing_indicator_count",
    "indicator_count",
    "has_missing_indicators",
    "rank",
}
CATEGORY_COLUMNS = {
    "region",
    "year",
    "category",
    "category_contribution",
    "category_weight_total",
    "category_score",
}
SUBCATEGORY_COLUMNS = {
    "region",
    "year",
    "category",
    "subcategory",
    "subcategory_contribution",
    "subcategory_weight_total",
    "subcategory_score",
}


def load_structural_index_artifacts(repo_root: Path) -> dict[str, pd.DataFrame]:
    """저장된 AHP 지수 CSV를 고정된 이름으로 읽는다."""

    artifact_dir = repo_root / "data" / "processed" / INDEX_DIRNAME
    artifacts: dict[str, pd.DataFrame] = {}
    missing = []
    for key, filename in ARTIFACT_FILES.items():
        path = artifact_dir / filename
        if not path.exists():
            missing.append(str(path.relative_to(repo_root)))
            continue
        artifacts[key] = pd.read_csv(path, encoding="utf-8-sig")
    if missing:
        raise FileNotFoundError(f"구조환경지수 산출물 누락: {missing}")
    return artifacts


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 필수 컬럼 누락: {missing}")


def _validate_key_grid(frame: pd.DataFrame, *, name: str, key_columns: list[str]) -> None:
    if frame.duplicated(key_columns).any():
        raise ValueError(f"{name}에 중복 키가 있습니다: {key_columns}")
    if "전국" in set(frame["region"].dropna()):
        raise ValueError(f"{name}에 전국 행이 포함되어 있습니다.")
    expected = len(DEFAULT_STRUCTURAL_REGIONS) * len(DEFAULT_STRUCTURAL_YEARS)
    if len(frame) != expected:
        raise ValueError(f"{name} 행 수가 17개 시도×2016–2024와 다릅니다: {len(frame)}")
    if set(frame["region"]) != set(DEFAULT_STRUCTURAL_REGIONS):
        raise ValueError(f"{name}의 지역 구성이 17개 시도와 다릅니다.")
    if set(frame["year"]) != set(DEFAULT_STRUCTURAL_YEARS):
        raise ValueError(f"{name}의 연도 구성이 2016–2024와 다릅니다.")


def _validate_finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{name}.{column}에 결측 또는 비유한값이 있습니다.")


def validate_structural_index_artifacts(artifacts: dict[str, pd.DataFrame]) -> dict[str, int]:
    """최종지수·구성점수·민감도 결과의 행, 키, 유한값을 검증한다."""

    for key in ("pooled_final", "yearly_final"):
        frame = artifacts[key]
        _require_columns(frame, FINAL_COLUMNS, key)
        _validate_key_grid(frame, name=key, key_columns=["region", "year"])
        _validate_finite(frame, ["final_index", "rank"], key)
        if frame["has_missing_indicators"].fillna(True).astype(bool).any():
            raise ValueError(f"{key}에 결측 지표를 포함한 최종지수가 있습니다.")

    for key in ("pooled_category",):
        frame = artifacts[key]
        _require_columns(frame, CATEGORY_COLUMNS, key)
        _validate_finite(frame, ["category_contribution", "category_score"], key)
        if frame.duplicated(["region", "year", "category"]).any():
            raise ValueError(f"{key}에 지역·연도·대영역 중복이 있습니다.")
        if "전국" in set(frame["region"]):
            raise ValueError(f"{key}에 전국 행이 포함되어 있습니다.")

    frame = artifacts["pooled_subcategory"]
    _require_columns(frame, SUBCATEGORY_COLUMNS, "pooled_subcategory")
    _validate_finite(frame, ["subcategory_contribution", "subcategory_score"], "pooled_subcategory")
    if frame.duplicated(["region", "year", "subcategory"]).any():
        raise ValueError("pooled_subcategory에 지역·연도·세부영역 중복이 있습니다.")
    if "전국" in set(frame["region"]):
        raise ValueError("pooled_subcategory에 전국 행이 포함되어 있습니다.")

    comparison = artifacts["yearly_comparison"]
    _require_columns(
        comparison,
        {
            "region",
            "year",
            "final_index_pooled",
            "final_index_yearly",
            "rank_pooled",
            "rank_yearly",
        },
        "yearly_comparison",
    )
    _validate_key_grid(comparison, name="yearly_comparison", key_columns=["region", "year"])
    _validate_finite(
        comparison,
        ["final_index_pooled", "final_index_yearly", "rank_pooled", "rank_yearly"],
        "yearly_comparison",
    )

    indicator = artifacts["pooled_indicator"]
    _require_columns(
        indicator,
        {"region", "year", "indicator_id", "indicator_weighted_contribution"},
        "pooled_indicator",
    )
    if indicator.duplicated(["region", "year", "indicator_id"]).any():
        raise ValueError("pooled_indicator에 지역·연도·지표 중복이 있습니다.")
    if "전국" in set(indicator["region"]):
        raise ValueError("pooled_indicator에 전국 행이 포함되어 있습니다.")
    if indicator["indicator_weighted_contribution"].notna().any():
        values = pd.to_numeric(
            indicator["indicator_weighted_contribution"].dropna(), errors="coerce"
        )
        if not np.isfinite(values).all():
            raise ValueError(
                "pooled_indicator.indicator_weighted_contribution에 비유한값이 있습니다."
            )

    for key in ("family_sensitivity", "family_weight_transfer"):
        frame = artifacts[key]
        _require_columns(frame, {"region", "year", "abs_score_diff", "abs_rank_diff"}, key)
        _validate_key_grid(frame, name=key, key_columns=["region", "year"])
        _validate_finite(frame, ["abs_score_diff", "abs_rank_diff"], key)

    return {
        "final_index_rows": len(artifacts["pooled_final"]),
        "category_rows": len(artifacts["pooled_category"]),
        "subcategory_rows": len(artifacts["pooled_subcategory"]),
        "indicator_rows": len(artifacts["pooled_indicator"]),
    }


def build_index_summary(artifacts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """pooled 본계열과 yearly 민감도, 순위·변화량을 하나로 결합한다."""

    pooled = artifacts["pooled_final"].rename(
        columns={"final_index": "pooled_index", "rank": "pooled_rank"}
    )
    yearly = artifacts["yearly_final"].rename(
        columns={"final_index": "yearly_index", "rank": "yearly_rank"}
    )
    comparison = artifacts["yearly_comparison"]
    result = pooled.merge(
        yearly[["region", "year", "yearly_index", "yearly_rank"]],
        on=["region", "year"],
        validate="one_to_one",
    )
    return result.merge(
        comparison[
            [
                "region",
                "year",
                "score_diff_yearly_minus_pooled",
                "abs_score_diff",
                "rank_diff_yearly_minus_pooled",
                "abs_rank_diff",
            ]
        ],
        on=["region", "year"],
        validate="one_to_one",
    )


def plot_pooled_index_trends(summary: pd.DataFrame) -> plt.Figure:
    """17개 시도의 pooled 종합지수 연도 추세를 small multiples로 그린다."""

    ncols = 4
    nrows = int(np.ceil(len(DEFAULT_STRUCTURAL_REGIONS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.5 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, region in zip(axes, DEFAULT_STRUCTURAL_REGIONS, strict=False):
        data = summary.loc[summary["region"].eq(region)].sort_values("year")
        ax.plot(
            data["year"],
            data["pooled_index"],
            color=PALETTE[0],
            marker="o",
            linewidth=1.6,
            markersize=3,
        )
        ax.set_title(region)
        ax.set_xticks(DEFAULT_STRUCTURAL_YEARS[::2])
        ax.grid(alpha=0.25)
    for ax in axes[len(DEFAULT_STRUCTURAL_REGIONS) :]:
        ax.set_visible(False)
    fig.suptitle("pooled 종합 구조환경지수 연도 추세", fontsize=16)
    fig.supxlabel("연도")
    fig.supylabel("지수(0–100)")
    fig.tight_layout()
    return fig


def plot_annual_rank_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """연도별 pooled 지역 순위를 heatmap으로 표시한다."""

    rank = summary.pivot(index="region", columns="year", values="pooled_rank").reindex(
        DEFAULT_STRUCTURAL_REGIONS
    )
    cmap = LinearSegmentedColormap.from_list(
        "yumocha_rank", [YOMOCHA_WEB_COLORS["surface_alt"], YOMOCHA_WEB_COLORS["accent"]]
    )
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        rank,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar_kws={"label": "순위(1=상위)"},
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("pooled 종합지수 연도별 지역 순위")
    ax.set_xlabel("연도")
    ax.set_ylabel("지역")
    fig.tight_layout()
    return fig


def plot_pooled_yearly_comparison(summary: pd.DataFrame) -> plt.Figure:
    """pooled 본계열과 yearly 민감도 점수의 지역·연도별 차이를 표시한다."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(
        summary["pooled_index"], summary["yearly_index"], color=PALETTE[0], alpha=0.55, s=28
    )
    limits = [
        summary[["pooled_index", "yearly_index"]].min().min(),
        summary[["pooled_index", "yearly_index"]].max().max(),
    ]
    axes[0].plot(limits, limits, color=PALETTE[4], linestyle="--", linewidth=1)
    axes[0].set_xlabel("pooled 지수")
    axes[0].set_ylabel("yearly 지수")
    axes[0].set_title("표준화 방식별 종합지수 비교")
    annual = summary.groupby("year", as_index=False)["abs_score_diff"].mean()
    axes[1].plot(annual["year"], annual["abs_score_diff"], color=PALETTE[0], marker="o")
    axes[1].set_title("연도별 평균 절대 점수 차이")
    axes[1].set_xlabel("연도")
    axes[1].set_ylabel("|yearly - pooled|")
    axes[1].set_xticks(DEFAULT_STRUCTURAL_YEARS)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("pooled 본계열·yearly 민감도 분석")
    fig.tight_layout()
    return fig


def plot_region_component_comparison(
    artifacts: dict[str, pd.DataFrame], *, region: str = "제주"
) -> plt.Figure:
    """선택 지역의 대영역 점수와 최종지수 기여도를 비교한다."""

    category = artifacts["pooled_category"].loc[lambda frame: frame["region"].eq(region)].copy()
    if category.empty:
        raise ValueError(f"구성점수에 지역이 없습니다: {region}")
    contribution = category.pivot(
        index="year", columns="category", values="category_contribution"
    ).reindex(DEFAULT_STRUCTURAL_YEARS)
    score = category.pivot(index="year", columns="category", values="category_score").reindex(
        DEFAULT_STRUCTURAL_YEARS
    )
    categories = list(contribution.columns)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(categories))]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].stackplot(
        contribution.index,
        [contribution[col] for col in categories],
        labels=categories,
        colors=colors,
        alpha=0.85,
    )
    axes[0].set_title(f"{region} pooled 종합지수 기여도")
    axes[0].set_ylabel("기여도(점수)")
    axes[0].legend(ncol=2, loc="upper left", fontsize=9)
    for category_name, color in zip(categories, colors, strict=True):
        axes[1].plot(
            score.index,
            score[category_name],
            marker="o",
            linewidth=1.5,
            label=category_name,
            color=color,
        )
    axes[1].set_title(f"{region} 대영역 점수")
    axes[1].set_ylabel("대영역 점수(0–100)")
    axes[1].set_xlabel("연도")
    axes[1].set_xticks(DEFAULT_STRUCTURAL_YEARS)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_family_friendly_impact(artifacts: dict[str, pd.DataFrame]) -> plt.Figure:
    """2016·2017 가족친화인증기업 가중치 이전의 점수·순위 영향을 표시한다."""

    impact = artifacts["family_weight_transfer"].copy()
    annual = impact.groupby("year", as_index=False).agg(
        abs_score_diff=("abs_score_diff", "mean"), abs_rank_diff=("abs_rank_diff", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(annual["year"].astype(str), annual["abs_score_diff"], color=PALETTE[0])
    axes[0].set_title("가중치 이전의 평균 절대 점수 차이")
    axes[0].set_ylabel("|가중치 이전 - raking|")
    axes[1].bar(annual["year"].astype(str), annual["abs_rank_diff"], color=PALETTE[1])
    axes[1].set_title("가중치 이전의 평균 절대 순위 차이")
    axes[1].set_ylabel("순위 단계")
    for ax in axes:
        ax.set_xlabel("연도")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("2016·2017 가족친화인증기업 가중치 이전 영향")
    fig.tight_layout()
    return fig


def summarize_qa(artifacts: dict[str, pd.DataFrame], summary: pd.DataFrame) -> dict[str, float]:
    """QA 보고서에 사용할 핵심 비교 통계를 계산한다."""

    correlations = summary.groupby("year").apply(
        lambda frame: frame["pooled_index"].corr(frame["yearly_index"]), include_groups=False
    )
    rank_gap = (summary["pooled_rank"] - summary["yearly_rank"]).abs()
    return {
        "yearly_correlation_min": float(correlations.min()),
        "yearly_correlation_max": float(correlations.max()),
        "mean_absolute_rank_gap": float(rank_gap.mean()),
        "rank_gap_within_two_pct": float((rank_gap <= 2).mean() * 100),
        "max_absolute_rank_gap": float(rank_gap.max()),
        "family_transfer_2016_2017_mean_score_gap": float(
            artifacts["family_weight_transfer"]
            .loc[lambda frame: frame["year"].isin([2016, 2017]), "abs_score_diff"]
            .mean()
        ),
    }
