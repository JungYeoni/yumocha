"""2016~2024년 저출생 대응 실질 계획예산을 대영역·세부영역별 누적 막대로 그린다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plots import save_figure

np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_INPUT = ANALYSIS_DIR / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
DEFAULT_FIGURE_DIR = REPO_ROOT / "reports/figures/budget_composition"
REAL_BUDGET = "당해계획예산_실질_백만원_provisional"
YEARS = list(range(2016, 2025))

SUBAREA_ORDER = [
    "1-1. 고용여건",
    "1-2. 주거안정성",
    "1-3. 경제적 여건",
    "2-1. 돌봄 여건",
    "2-2. 여가 인프라",
    "2-3. 가사수행 격차",
    "3-1. 의료서비스 여건",
    "3-2. 산후조리 여건",
    "3-3. 아동안전 수준",
    "4-1. 일·가정 양립 여건",
    "4-2. 사회적 가치관",
]
INPUT_SUBAREAS = [*SUBAREA_ORDER, "지표체계 외"]
MAJOR_ORDER = ["경제·고용·주거", "가족·생활", "보건·안전", "사회·문화"]

SUBAREA_TO_MAJOR = {
    subarea: (
        "경제·고용·주거"
        if subarea.startswith("1-")
        else "가족·생활"
        if subarea.startswith("2-")
        else "보건·안전"
        if subarea.startswith("3-")
        else "사회·문화"
        if subarea.startswith("4-")
        else "사회·문화"
    )
    for subarea in SUBAREA_ORDER
}

MAJOR_COLORS = {
    "경제·고용·주거": "#246BEB",
    "가족·생활": "#F59E0B",
    "보건·안전": "#718355",
    "사회·문화": "#C060A1",
}
SUBAREA_COLORS = {
    "1-1. 고용여건": "#0B50D0",
    "1-2. 주거안정성": "#5B8FF9",
    "1-3. 경제적 여건": "#9FC0FF",
    "2-1. 돌봄 여건": "#D97706",
    "2-2. 여가 인프라": "#F59E0B",
    "2-3. 가사수행 격차": "#F6C453",
    "3-1. 의료서비스 여건": "#556B2F",
    "3-2. 산후조리 여건": "#7E9461",
    "3-3. 아동안전 수준": "#A3B18A",
    "4-1. 일·가정 양립 여건": "#9B4D96",
    "4-2. 사회적 가치관": "#D989B5",
}


def validate_input(data: pd.DataFrame) -> None:
    """영역별 누적 막대에 필요한 완전 패널과 분류값을 검증한다."""
    required = {"지역", "연도", "세부영역", REAL_BUDGET, "CPI_기준연도", "예산결측_사업수"}
    if missing := required - set(data.columns):
        raise KeyError(f"영역별 예산 구성 입력 필수 컬럼 누락: {sorted(missing)}")
    if data["지역"].nunique() != 17 or sorted(data["연도"].unique()) != YEARS:
        raise ValueError("입력은 17개 시도의 2016~2024년 패널이어야 합니다.")
    if set(data["세부영역"].unique()) != set(INPUT_SUBAREAS):
        raise ValueError("입력에 11개 세부영역과 지표체계 외 분류가 모두 필요합니다.")
    keys = ["지역", "연도", "세부영역"]
    if data.duplicated(keys).any():
        raise ValueError("지역×연도×세부영역 키가 중복됩니다.")
    group_sets = data.groupby(["지역", "연도"])["세부영역"].agg(set)
    if not group_sets.map(lambda values: values == set(INPUT_SUBAREAS)).all():
        raise ValueError("모든 지역×연도에 12개 기준 분류가 정확히 한 번씩 필요합니다.")
    if data[REAL_BUDGET].isna().any() or (data[REAL_BUDGET] < 0).any():
        raise ValueError("실질 계획예산에 결측 또는 음수가 있습니다.")


def build_composition_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """연도별 세부영역·대영역 실질예산 합계와 비중을 산출한다."""
    validate_input(data)
    data = data.loc[data["세부영역"].ne("지표체계 외")].copy()
    detail = (
        data.groupby(["연도", "세부영역"], as_index=False)
        .agg(
            실질계획예산_2024년가격_백만원=(REAL_BUDGET, "sum"),
            예산결측_사업수=("예산결측_사업수", "sum"),
        )
        .sort_values(["연도", "세부영역"])
        .reset_index(drop=True)
    )
    detail["대영역"] = detail["세부영역"].map(SUBAREA_TO_MAJOR)
    totals = detail.groupby("연도")["실질계획예산_2024년가격_백만원"].transform("sum")
    detail["연도내_세부영역비중_pct"] = detail["실질계획예산_2024년가격_백만원"] / totals * 100
    detail["예산누락주의"] = detail["예산결측_사업수"].gt(0)

    major = (
        detail.groupby(["연도", "대영역"], as_index=False)
        .agg(
            실질계획예산_2024년가격_백만원=("실질계획예산_2024년가격_백만원", "sum"),
            예산결측_사업수=("예산결측_사업수", "sum"),
        )
        .sort_values(["연도", "대영역"])
        .reset_index(drop=True)
    )
    major_totals = major.groupby("연도")["실질계획예산_2024년가격_백만원"].transform("sum")
    major["연도내_대영역비중_pct"] = major["실질계획예산_2024년가격_백만원"] / major_totals * 100
    major["예산누락주의"] = major["예산결측_사업수"].gt(0)
    return detail, major


def plot_budget_composition(
    table: pd.DataFrame,
    category: str,
    order: list[str],
    colors: dict[str, str],
    title: str,
) -> plt.Figure:
    """연도별 실질예산 총액을 영역별 누적 막대로 그린다."""
    pivot = (
        table.pivot(index="연도", columns=category, values="실질계획예산_2024년가격_백만원")
        .reindex(index=YEARS, columns=order)
        .fillna(0)
        / 1_000_000
    )
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for name in order:
        values = pivot[name].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            label=name,
            color=colors[name],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values
    for xpos, total in zip(x, bottom, strict=True):
        ax.text(
            xpos, total + bottom.max() * 0.012, f"{total:.1f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_xticks(x, YEARS)
    ax.set_xlabel("연도")
    ax.set_ylabel("실질 계획예산(조 원, 2024년 가격)")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(
        title=category,
        ncol=3 if len(order) > 6 else len(order),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        columnspacing=1.35,
        handlelength=1.2,
    )
    fig.suptitle(title, y=0.99, fontsize=17)
    fig.text(
        0.5,
        0.947,
        "17개 시도 합계·지표체계 내 예산만 포함·2024년 가격 실질예산·막대 위 숫자는 영역 합계(조 원)",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.02,
        "주: '지표체계 외' 예산은 제외했으며 계획예산은 집행액이 아님. 시행계획 기록방식 변화의 영향을 받을 수 있음.",
        ha="center",
        fontsize=9,
        color="#6B7280",
    )
    fig.tight_layout(rect=[0.04, 0.09, 0.98, 0.91])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail, major = build_composition_tables(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "2016-2024_연도별_세부영역별_저출생대응_실질계획예산_구성.csv",
        index=False,
        encoding="utf-8-sig",
    )
    major.to_csv(
        args.output_dir / "2016-2024_연도별_대영역별_저출생대응_실질계획예산_구성.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figures = {
        "2016-2024_대영역별_저출생대응_실질계획예산_누적막대": plot_budget_composition(
            major,
            "대영역",
            MAJOR_ORDER,
            MAJOR_COLORS,
            "2016~2024년 대영역별 저출생 대응 계획예산 구성과 총규모",
        ),
        "2016-2024_세부영역별_저출생대응_실질계획예산_누적막대": plot_budget_composition(
            detail,
            "세부영역",
            SUBAREA_ORDER,
            SUBAREA_COLORS,
            "2016~2024년 세부영역별 저출생 대응 계획예산 구성과 총규모",
        ),
    }
    for name, figure in figures.items():
        save_figure(figure, args.figure_dir / name)
        plt.close(figure)


if __name__ == "__main__":
    main()
