"""#62 세부영역별 재정대응(F_it) 분포·추세 시각화를 생성한다.

계산 로직 없이 순수 시각화 전용 스크립트다. 입력은
build_subarea_fiscal_response_variable.py가 만든
2016-2024_세부영역별_인구1인당_실질예산액.csv이며, "지표체계 외"는 회귀분석과
동일하게 제외한다(정책 영역이 아니라 잔여 범주).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
)
from src.visualization.plots import save_figure
from src.visualization.trends import (
    plot_fiscal_response_overview,
    plot_fiscal_response_subarea_small_multiples,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISCAL_RESPONSE = (
    REPO_ROOT / "data" / "processed" / "analysis" / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "세부영역_재정대응_분포추세_EDA"
SUBAREA_ORDER = list(FISCAL_TO_STRUCTURAL_SUBCATEGORY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiscal-response", type=Path, default=DEFAULT_FISCAL_RESPONSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fiscal_response.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {args.fiscal_response}")

    fiscal_response = pd.read_csv(args.fiscal_response)
    indexable = fiscal_response.loc[fiscal_response["세부영역"].ne("지표체계 외")].copy()

    observed = set(indexable["세부영역"])
    if observed != set(SUBAREA_ORDER):
        raise ValueError(
            f"세부영역 표기가 예상과 다릅니다: 데이터에만={sorted(observed - set(SUBAREA_ORDER))}, "
            f"예상에만={sorted(set(SUBAREA_ORDER) - observed)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_fiscal_response_overview(indexable),
        args.output_dir / "세부영역_재정대응_개요",
        formats=["png"],
    )
    save_figure(
        plot_fiscal_response_subarea_small_multiples(indexable, subarea_order=SUBAREA_ORDER),
        args.output_dir / "세부영역별_재정대응_추세_small_multiples",
        formats=["png"],
    )
    plt.close("all")

    print(f"이미지 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
