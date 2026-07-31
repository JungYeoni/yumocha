"""2021년 의미보존 전수검토 확정 오류를 관련 산출물에 동시 반영한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from scripts.consolidate_2021_area_labels import consolidate_labels, save_outputs
from src.features.text_match import dedup_label


@dataclass(frozen=True)
class SemanticCorrection:
    region: str
    source_row: str
    name: str
    expected_original: str
    expected_cleaned: str
    corrected_cleaned: str
    reason: str


CORRECTIONS = (
    SemanticCorrection(
        "인천",
        "1747",
        "저소득층 아동,청소년 구강보건사업",
        "저소득,장애 아동 청소년 구강보건서비스 제공",
        "저소득층 아동 청소년 구강보건서비스 제공",
        "저소득, 장애 아동 청소년 구강보건서비스 제공",
        "`장애` 대상이 삭제되고 `저소득`이 `저소득층`으로 바뀐 의미변경 복구",
    ),
    SemanticCorrection(
        "인천",
        "1828",
        "다문화가족 생애주기별 맞춤형 서비스",
        (
            "지원대상 : 유,초,중,고 재(원)학 중인 다문화 학(원)생과 학부모 "
            "지원내용 : 학생맞춤형 교육지원 및 다문화 이해제고"
        ),
        (
            "지원대상 : 유,초,중,고 재학 중인 다문화 학생과 학부모 "
            "지원내용 : 학생맞춤형 교육지원 및 다문화 이해제고"
        ),
        (
            "지원대상 : 유,초,중,고 재(원)학 중인 다문화 학(원)생과 학부모 "
            "지원내용 : 학생맞춤형 교육지원 및 다문화 이해제고"
        ),
        "`재원·원생` 의미가 삭제된 대상 범위 축소 복구",
    ),
    SemanticCorrection(
        "서울",
        "322",
        "다문화가족에 대한 생애주기별 맞춤형 서비스 제공",
        "다+온센터(거점형 다문화교육지원센터)에서 학부모 연수 추진",
        "다온센터(거점형 다문화교육지원센터)에서 학부모 연수 추진",
        "다+온센터(거점형 다문화교육지원센터)에서 학부모 연수 추진",
        "고유 사업명 `다+온센터`의 `+` 표기 복구",
    ),
    SemanticCorrection(
        "전북",
        "7630",
        "노인일자리사업 수행기관 인프라 확충(시니어클럽 운영지원)",
        (
            "내용: 노인 적합 일자리 개발･보급과 수행 전담기관 지원 시 니어클럽 "
            "인건비, 운영비, 사업비 지원"
        ),
        (
            "내용: 노인 적합 일자리 개발･보급과 수행 전담기관 지원 시 시니어클럽 "
            "인건비, 운영비, 사업비 지원"
        ),
        (
            "내용: 노인 적합 일자리 개발･보급과 수행 전담기관 지원 시니어클럽 "
            "인건비, 운영비, 사업비 지원"
        ),
        "`시 니어클럽` 교정 과정에서 추가된 중복 `시` 제거",
    ),
)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        dtype={"원본행": "string"},
        **kwargs,
    )


def _write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def apply_corrections(data_root: Path) -> list[dict[str, object]]:
    """확정 4건을 checkpoint·wide·long에 반영하고 변경 기록을 반환한다."""
    checkpoint_path = data_root / "2021_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    review_rows: list[dict[str, object]] = []

    for correction in CORRECTIONS:
        wide_path = data_root / correction.region / f"2021_{correction.region}_세부사업_정제.csv"
        long_path = (
            data_root / correction.region / f"2021_{correction.region}_세부사업_정제_long.csv"
        )
        wide = _read_csv(wide_path)
        long = _read_csv(long_path)
        wide_mask = (
            wide["지역"].eq(correction.region)
            & wide["원본행"].eq(correction.source_row)
            & wide["세부사업명"].eq(correction.name)
        )
        long_mask = (
            long["지역"].eq(correction.region)
            & long["원본행"].eq(correction.source_row)
            & long["세부사업명"].eq(correction.name)
        )
        if int(wide_mask.sum()) != 1 or int(long_mask.sum()) != 2:
            raise ValueError(
                f"수정 키 건수 불일치: {correction.region}/{correction.source_row} "
                f"wide={int(wide_mask.sum())}, long={int(long_mask.sum())}"
            )
        wide_row = wide.loc[wide_mask].iloc[0]
        if wide_row["주요내용"] != correction.expected_original:
            raise ValueError(f"예상 원문 불일치: {correction.region}/{correction.source_row}")
        corrected_checkpoint_mask = (
            checkpoint["주요내용_정제"].apply(dedup_label).eq(correction.corrected_cleaned)
        )
        already_corrected = (
            wide_row["주요내용_정제"] == correction.corrected_cleaned
            and long.loc[long_mask, "주요내용_정제"].eq(correction.corrected_cleaned).all()
            and int(corrected_checkpoint_mask.sum()) == 1
        )
        if already_corrected:
            review_rows.append(
                {
                    "연도": 2021,
                    "지역": correction.region,
                    "원본행": correction.source_row,
                    "세부사업명": correction.name,
                    "주요내용": correction.expected_original,
                    "수정전_주요내용_정제": correction.expected_cleaned,
                    "수정후_주요내용_정제": correction.corrected_cleaned,
                    "검토결과": "의미보존 오류 수정",
                    "검토근거": correction.reason,
                }
            )
            continue
        if wide_row["주요내용_정제"] != correction.expected_cleaned:
            raise ValueError(f"예상 정제문 불일치: {correction.region}/{correction.source_row}")
        if not long.loc[long_mask, "주요내용_정제"].eq(correction.expected_cleaned).all():
            raise ValueError(
                f"long 예상 정제문 불일치: {correction.region}/{correction.source_row}"
            )

        checkpoint_mask = (
            checkpoint["주요내용_정제"].apply(dedup_label).eq(correction.expected_cleaned)
        )
        if int(checkpoint_mask.sum()) != 1:
            raise ValueError(
                f"체크포인트 정제문 연결 실패: {correction.region}/{correction.source_row} "
                f"matches={int(checkpoint_mask.sum())}"
            )

        wide.loc[wide_mask, "주요내용_정제"] = correction.corrected_cleaned
        long.loc[long_mask, "주요내용_정제"] = correction.corrected_cleaned
        checkpoint.loc[checkpoint_mask, "주요내용_정제"] = correction.corrected_cleaned
        _write_csv(wide, wide_path)
        _write_csv(long, long_path)

        review_rows.append(
            {
                "연도": 2021,
                "지역": correction.region,
                "원본행": correction.source_row,
                "세부사업명": correction.name,
                "주요내용": correction.expected_original,
                "수정전_주요내용_정제": correction.expected_cleaned,
                "수정후_주요내용_정제": correction.corrected_cleaned,
                "검토결과": "의미보존 오류 수정",
                "검토근거": correction.reason,
            }
        )

    _write_csv(checkpoint, checkpoint_path, index=True)
    return review_rows


def build_full_review(data_root: Path, correction_rows: list[dict[str, object]]) -> pd.DataFrame:
    """정제값이 달랐던 116건 전체의 검토 결과를 만든다."""
    frames = [_read_csv(path) for path in sorted(data_root.glob("*/2021_*_세부사업_정제.csv"))]
    combined = pd.concat(frames, ignore_index=True)
    original_compact = combined["주요내용"].astype(str).str.replace(r"\s+", "", regex=True)
    cleaned_compact = combined["주요내용_정제"].astype(str).str.replace(r"\s+", "", regex=True)
    changed = combined.loc[original_compact.ne(cleaned_compact)].copy()
    changed["문자유사도"] = [
        round(SequenceMatcher(None, original, cleaned).ratio(), 6)
        for original, cleaned in zip(
            original_compact.loc[changed.index],
            cleaned_compact.loc[changed.index],
            strict=True,
        )
    ]
    changed["검토결과"] = "의미보존 확인"
    changed["검토근거"] = "오탈자·공백·구두점·중복 표현 정리이며 대상·행위·수량 의미 유지"

    columns = [
        "연도",
        "지역",
        "원본행",
        "세부사업명",
        "주요내용",
        "주요내용_정제",
        "문자유사도",
        "검토결과",
        "검토근거",
    ]
    corrected_review = pd.DataFrame(correction_rows).rename(
        columns={"수정후_주요내용_정제": "주요내용_정제"}
    )
    corrected_review["문자유사도"] = [
        round(
            SequenceMatcher(
                None,
                "".join(str(original).split()),
                "".join(str(cleaned).split()),
            ).ratio(),
            6,
        )
        for original, cleaned in zip(
            corrected_review["주요내용"],
            corrected_review["주요내용_정제"],
            strict=True,
        )
    ]
    reviewed = pd.concat(
        [changed[columns], corrected_review[columns]],
        ignore_index=True,
    )
    if reviewed.duplicated(["지역", "원본행"]).any():
        raise ValueError("2021년 의미보존 검토표에 지역·원본행 중복이 있습니다.")
    return reviewed.sort_values(["검토결과", "문자유사도", "지역", "원본행"])


def refresh_label_outputs(data_root: Path) -> pd.DataFrame:
    """기존 수작업 라벨을 최신 정제문과 결합해 통합 학습 데이터를 갱신한다."""
    existing_output = next(data_root.rglob("2021_17개시도_영역분류_라벨_통합.csv"))
    output_dir = existing_output.parent
    input_dir = output_dir.parent
    combined, qa = consolidate_labels(input_dir, data_root)
    save_outputs(combined, qa, output_dir)
    return combined


def build_label_impact(
    combined_labels: pd.DataFrame,
    correction_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """수정 4건의 기존 라벨과 유지 판정을 기록한다."""
    keys = pd.DataFrame(correction_rows)[["지역", "원본행"]]
    keys["원본행"] = keys["원본행"].astype("string")
    combined_labels = combined_labels.copy()
    combined_labels["원본행"] = combined_labels["원본행"].astype("string")
    impact = keys.merge(
        combined_labels[["지역", "원본행", "세부사업명", "주요내용_정제", "대영역", "세부영역"]],
        on=["지역", "원본행"],
        how="left",
        validate="one_to_one",
    )
    impact["라벨검토결과"] = "기존 라벨 유지"
    impact["판단근거"] = (
        "정제문 오류 수정 후에도 사업 목적과 기존 대영역·세부영역의 분류 관계가 유지됨"
    )
    return impact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("reports/yearly/2021/2021_LLM_의미보존_전수검토.csv"),
    )
    parser.add_argument(
        "--label-impact-output",
        type=Path,
        default=Path("reports/yearly/2021/2021_LLM_정제변경_라벨영향검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    correction_rows = apply_corrections(args.data_root)
    full_review = build_full_review(args.data_root, correction_rows)
    combined_labels = refresh_label_outputs(args.data_root)
    label_impact = build_label_impact(combined_labels, correction_rows)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(full_review, args.review_output)
    _write_csv(label_impact, args.label_impact_output)
    print(f"정제 전후 변경 검토: {len(full_review):,}건")
    print(f"의미보존 오류 수정: {len(correction_rows):,}건")
    print(f"라벨 영향 검토: {len(label_impact):,}건, 기존 라벨 유지")


if __name__ == "__main__":
    main()
