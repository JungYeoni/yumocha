"""28개 구조환경지표의 검증 원자료 기준 추세 시각화와 상세 보고서를 생성한다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.features.trend_eda import (
    build_structural_region_summary,
    render_structural_region_report,
    reshape_structural_indicators,
)
from src.visualization.trends import (
    plot_region_small_multiples,
    plot_structural_indicator_overview,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/구조환경지표_28개_보간전_기준패널.csv"
DEFAULT_OUTPUT = REPO_ROOT / "result/구조환경지표_시각화"
DEFAULT_REPORT = REPO_ROOT / "reports/methodology/20260806_#63_구조환경지표_추세_시각화_QA.md"
EXPECTED_REGIONS = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
)
EXPECTED_YEARS = tuple(range(2016, 2025))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value).strip("_")


def load_indicator_names(repo_root: Path = REPO_ROOT) -> list[str]:
    """검증 manifest의 지표명을 순서대로 읽는다."""

    config = yaml.safe_load(
        (repo_root / "configs/structural_indicators_verification.yaml").read_text(encoding="utf-8")
    )
    names = [item["name"] for item in config["indicators"]]
    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError(f"구조환경지표 manifest 지표명 수·중복이 올바르지 않습니다: {len(names)}")
    return names


def load_manifest_metadata(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """매니페스트 버전과 지표 메타데이터를 읽는다."""

    config = yaml.safe_load(
        (repo_root / "configs/structural_indicators_verification.yaml").read_text(encoding="utf-8")
    )
    return config


def prepare_visualization_input(
    panel: pd.DataFrame,
    *,
    expected_regions: tuple[str, ...] = EXPECTED_REGIONS,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
) -> pd.DataFrame:
    """28개 long 패널을 기존 추세 EDA가 사용하는 wide 입력으로 변환한다."""

    required = {
        "지역",
        "지표_id",
        "지표명",
        "연도",
        "측정값",
        "단위",
        "출처",
        "대분류",
        "세부영역",
        "방향",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"시각화 입력 필수 컬럼 누락: {missing}")
    data = panel.copy()
    data["연도"] = pd.to_numeric(data["연도"], errors="raise").astype(int)
    if set(data["연도"]) != set(expected_years):
        raise ValueError("시각화 입력의 연도 범위가 2016–2024와 다릅니다.")
    if data.duplicated(["지역", "연도", "지표_id"]).any():
        raise ValueError("시각화 입력에 지역·연도·지표 ID 중복이 있습니다.")
    expected_keys = {
        (indicator_id, region, year)
        for indicator_id in data["지표_id"].unique()
        for region in expected_regions
        for year in expected_years
    }
    actual_keys = set(
        data.loc[data["지역"].ne("전국"), ["지표_id", "지역", "연도"]].itertuples(
            index=False, name=None
        )
    )
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        raise ValueError(
            f"시각화 입력의 지역·연도·지표 원본 행이 누락되었습니다: {missing_keys[:10]}"
        )
    metadata_counts = data.groupby("지표_id")[["지표명", "단위", "출처"]].nunique(dropna=False)
    if (metadata_counts > 1).any(axis=None):
        raise ValueError("지표별 지표명·단위·출처 메타데이터가 일관되지 않습니다.")

    wide = (
        data.pivot(
            index=["지역", "대분류", "세부영역", "지표명"],
            columns="연도",
            values="측정값",
        )
        .reset_index()
        .rename(
            columns={
                "대분류": "대영역",
                "지표명": "세부지표",
            }
        )
    )
    wide["검증상태"] = "검증 원자료 기준"
    wide.columns = [str(column) if isinstance(column, int) else column for column in wide.columns]
    return wide[["지역", "대영역", "세부영역", "세부지표", "검증상태", *map(str, expected_years)]]


def build_visualization_outputs(
    panel: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, int]:
    """검증 원자료 기준 28개 지표의 그림·요약표·보고서를 생성한다."""

    manifest = load_manifest_metadata()
    names = [item["name"] for item in manifest["indicators"]]
    wide = prepare_visualization_input(panel)
    long = reshape_structural_indicators(
        wide,
        expected_regions=EXPECTED_REGIONS,
        years=EXPECTED_YEARS,
    )
    actual_names = set(long["세부지표"].unique())
    if actual_names != set(names):
        raise ValueError(
            f"manifest와 입력 지표 목록이 다릅니다: {sorted(set(names) ^ actual_names)}"
        )
    summary = build_structural_region_summary(long, region_order=EXPECTED_REGIONS)
    indicator_metadata = (
        panel.groupby("지표명", as_index=False)
        .agg(단위=("단위", "first"), 출처=("출처", "first"))
        .rename(columns={"지표명": "세부지표"})
    )
    indicator_metadata["관측기간"] = f"{min(EXPECTED_YEARS)}–{max(EXPECTED_YEARS)}"
    indicator_metadata["자료기준"] = "검증 원자료 기준"
    summary = summary.merge(indicator_metadata, on="세부지표", how="left", validate="many_to_one")

    overview_dir = output_dir / "overview"
    region_dir = output_dir / "17개시도_추세"
    for directory in (overview_dir, region_dir, report_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    for indicator in names:
        stem = _safe_filename(indicator)
        overview = plot_structural_indicator_overview(long, indicator=indicator)
        overview.savefig(overview_dir / f"{stem}_overview.png", dpi=160, bbox_inches="tight")
        plt.close(overview)
        multiples = plot_region_small_multiples(
            long,
            indicator=indicator,
            region_order=list(EXPECTED_REGIONS),
        )
        multiples.savefig(region_dir / f"{stem}_17개시도.png", dpi=160, bbox_inches="tight")
        plt.close(multiples)

    summary.to_csv(
        output_dir / "구조환경지표별_17개시도_상세결과.csv", index=False, encoding="utf-8-sig"
    )
    report = render_structural_region_report(summary).replace(
        "이 부록은 검증된 28개 구조환경지표의 지역별 결과를 비교한다.",
        "이 부록은 검증 원자료 기준 28개 구조환경지표의 지역별 결과를 비교한다.",
    )
    report_path.write_text(
        "# #63 구조환경지표 추세 시각화 QA\n\n"
        "- 입력: `data/processed/구조환경지표_28개_보간전_기준패널.csv`\n"
        "- 대상: 28개 manifest 지표 × 17개 시도 × 2016–2024년\n"
        f"- 데이터셋 버전: `structural_indicators_verification.yaml` manifest_version={manifest['manifest_version']}; 입력 파일명 고정\n"
        "- 분석 기간: 2016–2024년, 17개 시도(전국값은 참고용)\n"
        "- 분할 전략: 해당 없음(예측·학습이 아닌 기술통계·시각화)\n"
        "- 평가 지표·교차검증 fold: 해당 없음(모델 평가 없음)\n"
        "- 핵심 가정: 지표별 방향성은 manifest를 따르고, 전국값은 지역 순위·변화량에서 제외\n"
        "- 방법론적 제약: 조사주기·관측 시작연도가 지표별로 다르며 원자료 결측은 보간하지 않음\n"
        "- 잠재 편향: 조사 표본·자기보고 응답·공표 기준 변경에 따른 측정 및 비교 가능성 편향\n"
        "- 보간·AHP 가중치·최종 지수는 적용하지 않고 검증 원자료의 실측·결측 상태를 유지\n"
        f"- 입력 행 수: {len(panel):,}행, long 변환 행 수: {len(long):,}행\n"
        f"- 생성 overview: {len(names):,}개, 17개 시도 추세: {len(names):,}개\n\n" + report,
        encoding="utf-8",
    )
    return {"indicators": len(names), "long_rows": len(long), "summary_rows": len(summary)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.input, encoding="utf-8-sig")
    counts = build_visualization_outputs(panel, output_dir=args.output_dir, report_path=args.report)
    print(f"구조환경지표 시각화 완료: {counts}")


if __name__ == "__main__":
    main()
