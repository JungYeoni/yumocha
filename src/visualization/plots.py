"""
재사용 가능한 시각화 함수 모음.

기준:
- 정적 이미지: dpi=300, bbox_inches='tight'
- 색약 친화 팔레트 기본 사용
- 한국어 폰트 설정 포함 (Mac: AppleGothic)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 전역 스타일 설정
plt.rcParams.update(
    {
        "font.family": "AppleGothic",  # macOS 한국어; Linux는 'NanumGothic' 등으로 변경
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)
PALETTE = sns.color_palette("colorblind")
sns.set_style("whitegrid")


def save_figure(fig: plt.Figure, path: str | Path, formats: list[str] | None = None) -> None:
    """그림 저장 — PNG + PDF 병행 저장 기본."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or ["png", "pdf"]
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")


def plot_distribution(df: pd.DataFrame, cols: list[str], title: str = "분포") -> plt.Figure:
    """수치형 컬럼 분포 히스토그램 + KDE."""
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=PALETTE[0])
        axes[i].set_title(col)
        axes[i].set_xlabel("값")
        axes[i].set_ylabel("빈도")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "상관관계 히트맵") -> plt.Figure:
    """상관관계 히트맵 — 하삼각만 표시."""
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str = "잔차 진단") -> plt.Figure:
    """잔차 진단 3종 패널 — 회귀 모델 평가용."""
    from scipy import stats

    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.4, color=PALETTE[0])
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("잔차 vs 예측값")
    axes[0].set_xlabel("예측값")
    axes[0].set_ylabel("잔차")

    axes[1].hist(residuals, bins=30, edgecolor="white", color=PALETTE[0])
    axes[1].set_title("잔차 분포")
    axes[1].set_xlabel("잔차")

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
