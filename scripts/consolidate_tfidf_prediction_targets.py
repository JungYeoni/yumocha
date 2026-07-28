"""2021년을 제외한 8개 연도의 TF-IDF 예측 대상 데이터를 취합한다.

지역 순서와 텍스트 정규화는 2021년 학습 데이터 취합 스크립트의 정의를
재사용하여 학습·예측 데이터의 전처리 차이를 방지한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.consolidate_2021_area_labels import REGION_ORDER, normalize_text

np.random.seed(42)

PREDICTION_YEARS = [2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024]

SOURCE_COLUMNS = [
    "연도",
    "지역",
    "대분류",
    "중분류",
    "사업분류재정구분",
    "세부사업명",
    "주요내용",
    "주요내용_정제",
    "당해예산",
    "전년도예산",
    "증감액",
    "증감율",
    "원본행",
    "지원대상",
    "지원내용_상세",
]

PREDICTION_COLUMNS = [
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "주요내용_정제",
    "분류텍스트",
    "주요내용_정제_결측",
    "원본파일",
]


def read_prediction_source(path: Path, *, year: int, region: str) -> pd.DataFrame:
    """지역·연도 정제 CSV 하나를 읽고 예측 입력 스키마를 검증한다."""
    frame = pd.read_csv(path)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"필수 열 누락: {path}={missing_columns}")

    key_columns = ["연도", "지역", "원본행"]
    invalid_key = frame[key_columns].isna().any(axis=1)
    invalid_key |= frame["원본행"].map(normalize_text).eq("")
    if invalid_key.any():
        missing = frame.loc[invalid_key, key_columns]
        raise ValueError(f"연도·지역·원본행 결측: {path}={missing.to_dict('records')}")

    observed_years = pd.to_numeric(frame["연도"], errors="coerce").unique().tolist()
    if observed_years != [year]:
        raise ValueError(f"파일명 연도와 데이터 연도가 다릅니다: {path}={observed_years}")

    normalized_regions = frame["지역"].map(normalize_text)
    observed_regions = normalized_regions.unique().tolist()
    if observed_regions != [region]:
        raise ValueError(f"경로 지역과 데이터 지역이 다릅니다: {path}={observed_regions}")

    output = frame[SOURCE_COLUMNS].copy()
    output["연도"] = pd.to_numeric(output["연도"], errors="raise").astype("int64")
    output["지역"] = normalized_regions
    output["원본파일"] = f"{region}/{path.name}"
    output["주요내용_정제_결측"] = output["주요내용_정제"].isna()
    normalized_project_names = output["세부사업명"].map(normalize_text)
    output["분류텍스트"] = (
        normalized_project_names + " " + output["주요내용_정제"].map(normalize_text)
    ).str.strip()

    if normalized_project_names.eq("").any() or output["분류텍스트"].eq("").any():
        raise ValueError(f"학습 텍스트를 만들 수 없는 행이 있습니다: {path}")
    if output.duplicated(key_columns).any():
        duplicate = output.loc[
            output.duplicated(key_columns, keep=False), key_columns
        ].drop_duplicates()
        raise ValueError(f"파일 내부 키 중복: {path}={duplicate.to_dict('records')}")
    return output


def consolidate_prediction_targets(
    source_dir: Path,
    *,
    years: Sequence[int] = PREDICTION_YEARS,
    regions: Sequence[str] = REGION_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """예측 대상 지역·연도 파일을 통합하고 1:1 입력 QA를 생성한다."""
    if 2021 in years:
        raise ValueError("정답 라벨 연도인 2021년은 예측 대상에 포함할 수 없습니다.")
    if len(set(years)) != len(years):
        raise ValueError("예측 대상 연도가 중복되었습니다.")
    if len(set(regions)) != len(regions):
        raise ValueError("예측 대상 지역이 중복되었습니다.")

    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for year in years:
        for region in regions:
            path = source_dir / region / f"{year}_{region}_세부사업_정제.csv"
            if not path.exists():
                missing_files.append(str(path))
                continue
            frames.append(read_prediction_source(path, year=year, region=region))

    if missing_files:
        raise FileNotFoundError(f"예측 대상 파일 누락: {missing_files}")
    if not frames:
        raise ValueError("취합할 예측 대상 데이터가 없습니다.")

    combined = pd.concat(frames, ignore_index=True)
    key_columns = ["연도", "지역", "원본행"]
    if combined[key_columns].isna().any().any():
        raise ValueError("통합 후 연도·지역·원본행 결측이 남아 있습니다.")
    if combined.duplicated(key_columns).any():
        duplicate = combined.loc[
            combined.duplicated(key_columns, keep=False), key_columns
        ].drop_duplicates()
        raise ValueError(f"통합 키 중복: {duplicate.to_dict('records')}")
    if combined["연도"].eq(2021).any():
        raise ValueError("통합 데이터에 정답 라벨 연도인 2021년이 포함되었습니다.")

    expected_pairs = {(year, region) for year in years for region in regions}
    observed_pairs = set(combined[["연도", "지역"]].itertuples(index=False, name=None))
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(f"연도·지역 조합 불완전: 누락={missing}, 예상 밖={extra}")

    year_order = {year: index for index, year in enumerate(years)}
    region_order = {region: index for index, region in enumerate(regions)}
    combined["_연도순서"] = combined["연도"].map(year_order)
    combined["_지역순서"] = combined["지역"].map(region_order)
    combined = (
        combined.sort_values(["_연도순서", "_지역순서", "원본행"])
        .drop(columns=["_연도순서", "_지역순서"])
        .reset_index(drop=True)
    )

    qa = (
        combined.groupby(["연도", "지역"], sort=False, observed=True)
        .agg(
            행수=("원본행", "size"),
            사업명_결측=("세부사업명", lambda values: int(values.isna().sum())),
            주요내용_정제_결측=("주요내용_정제_결측", "sum"),
            원본행_중복=("원본행", lambda values: int(values.duplicated().sum())),
            분류텍스트_결측=("분류텍스트", lambda values: int(values.eq("").sum())),
        )
        .reset_index()
    )
    return combined, qa


def save_outputs(
    combined: pd.DataFrame,
    qa: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """통합본·분류 핵심본·QA를 UTF-8-SIG CSV로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "통합본": output_dir / "2016_2020_2022_2024_TFIDF_분류대상_통합.csv",
        "분류본": output_dir / "2016_2020_2022_2024_TFIDF_분류대상.csv",
        "QA": output_dir / "2016_2020_2022_2024_TFIDF_분류대상_QA.csv",
    }
    combined.to_csv(paths["통합본"], index=False, encoding="utf-8-sig")
    combined[PREDICTION_COLUMNS].to_csv(paths["분류본"], index=False, encoding="utf-8-sig")
    qa.to_csv(paths["QA"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/interim"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/TFIDF_분류대상"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined, qa = consolidate_prediction_targets(args.source_dir)
    paths = save_outputs(combined, qa, args.output_dir)

    print(f"예측 대상 연도: {combined['연도'].nunique()}개")
    print(f"지역: {combined['지역'].nunique()}개")
    print(f"연도·지역 조합: {len(qa)}개")
    print(f"통합 행: {len(combined):,}개")
    print(f"주요내용_정제 결측: {int(combined['주요내용_정제_결측'].sum()):,}개")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
