"""2021년 17개 시도 영역분류 라벨 파일을 TF-IDF 학습용으로 취합한다.

기존 ``src/features``와 연도별 정제 노트북은 시행계획 원문을 세부사업
데이터로 변환하는 로직이다. 이번 입력은 정제가 끝난 지역별 라벨 XLSX이며,
원본 CSV 키 대조와 taxonomy 정규화가 필요해 별도 취합 단계로 구현했다.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

REGION_ORDER = [
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
]

BASE_COLUMNS = [
    "대영역",
    "세부영역",
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

MAJOR_BY_SUBCATEGORY = {
    "1-1. 고용여건": "1. 경제·고용·주거",
    "1-2. 주거안정성": "1. 경제·고용·주거",
    "1-3. 경제적 여건": "1. 경제·고용·주거",
    "2-1. 돌봄 여건": "2. 가족·생활",
    "2-2. 여가 인프라": "2. 가족·생활",
    "2-3. 가사수행 격차": "2. 가족·생활",
    "3-1. 의료서비스 여건": "3. 보건·안전",
    "3-2. 산후조리 여건": "3. 보건·안전",
    "3-3. 아동안전 수준": "3. 보건·안전",
    "4-1. 일·가정 양립 여건": "4. 사회·문화",
    "4-2. 사회적 가치관": "4. 사회·문화",
    "지표체계 외": "지표체계 외",
}

TRAIN_COLUMNS = [
    "지역",
    "원본행",
    "세부사업명",
    "주요내용_정제",
    "분류텍스트",
    "대영역",
    "세부영역",
]
LABEL_COLUMNS = ["대영역", "세부영역", "라벨원본파일"]
SOURCE_COLUMNS = [column for column in BASE_COLUMNS if column not in {"대영역", "세부영역"}]


def normalize_text(value: object) -> str:
    """문자열을 NFC로 통일하고 연속 공백을 한 칸으로 줄인다."""
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def region_from_filename(path: Path) -> str:
    """파일명에서 17개 시도 중 정확히 하나를 찾는다."""
    normalized_name = unicodedata.normalize("NFC", path.name)
    matches = [region for region in REGION_ORDER if f"_{region}_" in normalized_name]
    if len(matches) != 1:
        raise ValueError(f"파일명에서 지역을 하나로 판별할 수 없습니다: {path.name}")
    return matches[0]


def read_label_file(path: Path) -> pd.DataFrame:
    """지역 라벨 XLSX 하나를 읽고 공통 스키마를 검증한다."""
    excel = pd.ExcelFile(path)
    if len(excel.sheet_names) != 1:
        raise ValueError(f"시트가 정확히 하나가 아닙니다: {path.name}={excel.sheet_names}")

    frame = pd.read_excel(path, sheet_name=excel.sheet_names[0])
    missing_columns = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"필수 열 누락: {path.name}={missing_columns}")

    if frame["지역"].isna().any():
        raise ValueError(f"지역 결측이 있습니다: {path.name}")

    region = region_from_filename(path)
    observed_regions = frame["지역"].dropna().map(normalize_text).unique().tolist()
    if observed_regions != [region]:
        raise ValueError(f"파일명 지역과 데이터 지역이 다릅니다: {path.name}={observed_regions}")

    output = frame[BASE_COLUMNS].copy()
    output["지역"] = output["지역"].map(normalize_text)
    output["라벨원본파일"] = unicodedata.normalize("NFC", path.name)
    return output


def _key_records(frame: pd.DataFrame) -> set[tuple[str, object]]:
    """지역·원본행 키를 비교 가능한 튜플 집합으로 반환한다."""
    return set(frame[["지역", "원본행"]].itertuples(index=False, name=None))


def validate_source_keys(combined: pd.DataFrame, source_dir: Path) -> None:
    """통합 라벨 키를 지역별 정제 원본 CSV와 양방향으로 대조한다."""
    key_columns = ["지역", "원본행"]
    if combined[key_columns].isna().any().any():
        missing = combined.loc[combined[key_columns].isna().any(axis=1), key_columns]
        raise ValueError(f"지역·원본행 결측: {missing.to_dict('records')}")

    failures: list[dict[str, object]] = []
    for region in REGION_ORDER:
        source_path = source_dir / region / f"2021_{region}_세부사업_정제.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"지역 정제 원본 CSV가 없습니다: {source_path}")

        source = pd.read_csv(source_path, usecols=key_columns)
        if source[key_columns].isna().any().any():
            missing = source.loc[source[key_columns].isna().any(axis=1), key_columns]
            raise ValueError(f"원자료 지역·원본행 결측: {source_path}={missing.to_dict('records')}")
        source["지역"] = source["지역"].map(normalize_text)
        observed_regions = source["지역"].unique().tolist()
        if observed_regions != [region]:
            raise ValueError(
                f"원자료 경로 지역과 데이터 지역이 다릅니다: {source_path}={observed_regions}"
            )
        if source.duplicated(key_columns).any():
            duplicate = source.loc[
                source.duplicated(key_columns, keep=False), key_columns
            ].drop_duplicates()
            raise ValueError(
                f"원자료 지역·원본행 중복: {source_path}={duplicate.to_dict('records')}"
            )

        label_keys = _key_records(combined.loc[combined["지역"].eq(region)])
        source_keys = _key_records(source)
        missing_label = sorted(source_keys - label_keys)
        unexpected_label = sorted(label_keys - source_keys)
        if missing_label or unexpected_label:
            failures.append(
                {
                    "지역": region,
                    "라벨누락": missing_label,
                    "예상밖라벨": unexpected_label,
                }
            )

    if failures:
        raise ValueError(f"원자료 키 대조 실패: {failures}")


def refresh_from_sources(combined: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """수작업 라벨은 유지하고 비라벨 필드는 최신 지역별 정제 CSV에서 가져온다."""
    frames: list[pd.DataFrame] = []
    for region in REGION_ORDER:
        source_path = source_dir / region / f"2021_{region}_세부사업_정제.csv"
        source = pd.read_csv(source_path, usecols=SOURCE_COLUMNS)
        source["지역"] = source["지역"].map(normalize_text)
        frames.append(source)

    latest = pd.concat(frames, ignore_index=True)
    key_columns = ["지역", "원본행"]
    if latest.duplicated(key_columns).any():
        duplicate = latest.loc[
            latest.duplicated(key_columns, keep=False),
            key_columns,
        ].drop_duplicates()
        raise ValueError(f"최신 원자료 지역·원본행 중복: {duplicate.to_dict('records')}")

    labels = combined[[*key_columns, *LABEL_COLUMNS]].copy()
    refreshed = latest.merge(
        labels,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    required_labels = ["세부영역", "라벨원본파일"]
    if refreshed[required_labels].isna().any().any():
        missing = refreshed.loc[
            refreshed[required_labels].isna().any(axis=1),
            key_columns,
        ]
        raise ValueError(f"최신 원자료에 대응하는 라벨 누락: {missing.to_dict('records')}")
    return refreshed[[*BASE_COLUMNS, "라벨원본파일"]]


def consolidate_labels(
    input_dir: Path,
    source_dir: Path = Path("data/interim"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """지역별 파일을 결합하고 taxonomy를 세부영역 기준으로 정규화한다."""
    files = sorted(input_dir.glob("*.xlsx"))
    if len(files) != len(REGION_ORDER):
        raise ValueError(f"지역 파일은 17개여야 합니다: {len(files)}개")

    frames = [read_label_file(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)

    if combined[["지역", "원본행"]].isna().any().any():
        missing_key = combined.loc[
            combined[["지역", "원본행"]].isna().any(axis=1), ["지역", "원본행"]
        ]
        raise ValueError(f"지역·원본행 결측: {missing_key.to_dict('records')}")
    if set(combined["지역"]) != set(REGION_ORDER):
        missing = sorted(set(REGION_ORDER) - set(combined["지역"]))
        extra = sorted(set(combined["지역"]) - set(REGION_ORDER))
        raise ValueError(f"지역 불완전: 누락={missing}, 예상 밖={extra}")
    if not combined["연도"].eq(2021).all():
        raise ValueError("2021년 이외의 행이 포함되어 있습니다.")
    if combined.duplicated(["지역", "원본행"]).any():
        duplicate = combined.loc[
            combined.duplicated(["지역", "원본행"], keep=False), ["지역", "원본행"]
        ]
        raise ValueError(f"지역·원본행 중복: {duplicate.drop_duplicates().to_dict('records')}")
    validate_source_keys(combined, source_dir)
    combined = refresh_from_sources(combined, source_dir)

    combined["대영역_원본"] = combined["대영역"]
    canonical_major = combined["세부영역"].map(MAJOR_BY_SUBCATEGORY)
    unknown_subcategory = canonical_major.isna()
    if unknown_subcategory.any():
        values = sorted(combined.loc[unknown_subcategory, "세부영역"].dropna().unique())
        raise ValueError(f"정의되지 않은 세부영역: {values}")

    combined["taxonomy_정규화여부"] = combined["대영역"].ne(canonical_major)
    combined["대영역"] = canonical_major
    combined["주요내용_정제_결측"] = combined["주요내용_정제"].isna()
    combined["분류텍스트"] = (
        combined["세부사업명"].map(normalize_text)
        + " "
        + combined["주요내용_정제"].map(normalize_text)
    ).str.strip()

    if combined["세부사업명"].isna().any() or combined["분류텍스트"].eq("").any():
        raise ValueError("학습 텍스트를 만들 수 없는 행이 있습니다.")
    if combined[["대영역", "세부영역"]].isna().any().any():
        raise ValueError("정규화 후 라벨 결측이 남아 있습니다.")

    combined["지역"] = pd.Categorical(combined["지역"], categories=REGION_ORDER, ordered=True)
    combined = combined.sort_values(["지역", "원본행"]).reset_index(drop=True)
    combined["지역"] = combined["지역"].astype("string")

    qa = (
        combined.groupby("지역", sort=False, observed=True)
        .agg(
            행수=("원본행", "size"),
            taxonomy_정규화=("taxonomy_정규화여부", "sum"),
            주요내용_정제_결측=("주요내용_정제_결측", "sum"),
            대영역_종류=("대영역", "nunique"),
            세부영역_종류=("세부영역", "nunique"),
        )
        .reset_index()
    )
    qa["원본행_중복"] = (
        combined.groupby("지역", sort=False, observed=True)["원본행"]
        .apply(lambda values: int(values.duplicated().sum()))
        .to_numpy()
    )
    return combined, qa


def save_outputs(
    combined: pd.DataFrame,
    qa: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """통합본·학습본·QA·정규화 상세를 UTF-8-SIG CSV로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "통합본": output_dir / "2021_17개시도_영역분류_라벨_통합.csv",
        "학습본": output_dir / "2021_17개시도_TFIDF_학습데이터.csv",
        "QA": output_dir / "2021_17개시도_영역분류_라벨_QA.csv",
        "정규화상세": output_dir / "2021_영역분류_taxonomy_정규화_상세.csv",
    }
    combined.to_csv(paths["통합본"], index=False, encoding="utf-8-sig")
    combined[TRAIN_COLUMNS].to_csv(paths["학습본"], index=False, encoding="utf-8-sig")
    qa.to_csv(paths["QA"], index=False, encoding="utf-8-sig")
    combined.loc[
        combined["taxonomy_정규화여부"],
        ["지역", "원본행", "세부사업명", "대영역_원본", "대영역", "세부영역"],
    ].to_csv(paths["정규화상세"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/2021"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/interim"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/2021/통합"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined, qa = consolidate_labels(args.input_dir, args.source_dir)
    paths = save_outputs(combined, qa, args.output_dir)

    print(f"지역: {combined['지역'].nunique()}개")
    print(f"통합 행: {len(combined):,}개")
    print(f"taxonomy 정규화: {int(combined['taxonomy_정규화여부'].sum()):,}개")
    print(f"주요내용_정제 결측: {int(combined['주요내용_정제_결측'].sum()):,}개")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
