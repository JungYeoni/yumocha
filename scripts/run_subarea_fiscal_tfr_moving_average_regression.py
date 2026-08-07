"""#98 — 3개년 평균 실질 1인당 예산과 t+1·t+2 TFR의 관계를 추정한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from scripts.run_subarea_fiscal_tfr_regression import run_subarea_models
from src.visualization.plots import YOMOCHA_WEB_COLORS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = REPO_ROOT / "data/processed/analysis/2016-2024_세부영역별_재정반응성_회귀표본.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"
DEFAULT_SINGLE = DEFAULT_OUTPUT_DIR / "2016-2024_세부영역별_재정_TFR_모형C_고정효과_결과.csv"
KEYS = ["지역", "세부영역"]
BUDGET = "인구1인당_실질예산_원"
MA3 = "인구1인당_실질예산_3개년평균"


def build_moving_average_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """각 지역·세부영역의 t-2~t 평균과 t+1/t+2 TFR을 만든다."""
    panel = sample.sort_values([*KEYS, "연도"]).copy()
    if panel.duplicated([*KEYS, "연도"]).any():
        raise ValueError("지역·세부영역·연도 키 중복")
    year_gaps = panel.groupby(KEYS, sort=False)["연도"].diff().dropna().ne(1)
    if year_gaps.any():
        raise ValueError("3개년 이동평균과 후행 TFR 생성에는 그룹별 연속 연도가 필요합니다.")
    grouped = panel.groupby(KEYS, sort=False)
    panel[MA3] = grouped[BUDGET].transform(lambda x: x.rolling(3, min_periods=3).mean())
    panel["합계출산율_t+1"] = grouped["합계출산율"].shift(-1)
    panel["합계출산율_t+2"] = grouped["합계출산율"].shift(-2)
    return panel


def add_bh_correction(table: pd.DataFrame) -> pd.DataFrame:
    """동일 시차의 11개 세부영역 검정에 BH 보정을 적용한다."""
    result = table.copy()
    rejected, adjusted, _, _ = multipletests(result["p값"], alpha=0.05, method="fdr_bh")
    result["FDR_q값"] = adjusted
    result["FDR_0.05_유의"] = rejected
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-year-results", type=Path, default=DEFAULT_SINGLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = build_moving_average_sample(pd.read_csv(args.sample))
    tables = []
    for lag in (1, 2):
        model_sample = sample.copy()
        # 기존 동년 TFR 대신 후행 TFR을 종속변수로 사용한다.
        model_sample["합계출산율"] = sample[f"합계출산율_t+{lag}"]
        model_sample = model_sample.dropna(subset=["합계출산율", MA3])
        raw_result = run_subarea_models(model_sample, lag_column=MA3)
        if len(raw_result) != 11 or raw_result["모형"].nunique() != 11:
            raise ValueError("BH 보정에는 중복 없는 11개 세부영역 회귀 결과가 필요합니다.")
        result = add_bh_correction(raw_result)
        result.insert(0, "모형버전", f"3개년평균_t+{lag}")
        tables.append(result)
    combined = pd.concat(tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(
        args.output_dir / "2016-2024_세부영역별_3개년평균예산_TFR_회귀표본.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined.to_csv(
        args.output_dir / "2016-2024_세부영역별_3개년평균예산_TFR_고정효과_결과.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if args.single_year_results.is_file():
        single = pd.read_csv(args.single_year_results)
        wanted = {
            "기본모형(F_t-1)": "단년도_t+1",
            "강건성체크_2년시차(F_t-2)": "단년도_t+2",
        }
        single = single.loc[single["모형버전"].isin(wanted)].copy()
        single["모형버전"] = single["모형버전"].map(wanted)
        comparison = pd.concat([single, combined], ignore_index=True, sort=False)
        comparison.to_csv(
            args.output_dir / "2016-2024_단년도_3개년평균예산_TFR_결과비교.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
        for ax, lag in zip(axes, (1, 2), strict=True):
            subset = comparison.loc[
                comparison["모형버전"].isin([f"단년도_t+{lag}", f"3개년평균_t+{lag}"])
            ]
            labels = list(dict.fromkeys(subset["모형"]))
            for offset, version, color in zip(
                (-0.12, 0.12),
                (f"단년도_t+{lag}", f"3개년평균_t+{lag}"),
                (YOMOCHA_WEB_COLORS["muted_bar"], YOMOCHA_WEB_COLORS["accent"]),
                strict=True,
            ):
                rows = subset.loc[subset["모형버전"].eq(version)].set_index("모형").loc[labels]
                y = [i + offset for i in range(len(labels))]
                ax.errorbar(
                    rows["계수"],
                    y,
                    xerr=[
                        rows["계수"] - rows["95%신뢰구간_하한"],
                        rows["95%신뢰구간_상한"] - rows["계수"],
                    ],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=version,
                )
            ax.axvline(0, color="gray", lw=1)
            ax.set_title(f"t+{lag} TFR")
            ax.set_xlabel("log1p(실질 1인당 예산) 계수와 95% 신뢰구간")
            ax.set_yticks(range(len(labels)), labels)
            ax.legend()
        fig.suptitle("단년도와 3개년 평균 재정대응예산의 TFR 관련성 비교")
        fig.tight_layout()
        fig.savefig(args.output_dir / "2016-2024_단년도_3개년평균예산_TFR_계수비교.png", dpi=180)
        plt.close(fig)
    print(combined[["모형버전", "모형", "계수", "p값", "관측치"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
