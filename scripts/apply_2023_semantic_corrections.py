"""2023년 잔여 의미보존 후보 전수검토의 확정 오류를 반영한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.apply_2023_blank_input_recovery import resolve_checkpoint_indices
from scripts.apply_2024_semantic_corrections import CorrectionRule, corrected_text
from scripts.audit_llm_semantic_preservation import REGIONS
from scripts.consolidate_tfidf_prediction_targets import (
    consolidate_prediction_targets,
    save_outputs,
)
from src.features.text_match import dedup_label

CORRECTIONS = (
    CorrectionRule(
        "서울",
        "180",
        "strip_source_label",
        "사업명과 `원문:` 메타 라벨이 추가된 정제문에서 라벨·사업명 제거",
    ),
    CorrectionRule(
        "세종",
        "2750",
        "restore_original",
        "원문 앞에 세부사업명을 추가한 의미 외 정보 삽입 복구",
    ),
    CorrectionRule(
        "경기",
        "3663",
        "strip_source_label",
        "사업명과 `원문:` 메타 라벨이 추가된 정제문에서 라벨·사업명 제거",
    ),
    CorrectionRule(
        "경기",
        "3680",
        "restore_original",
        "경로당 운영 목적·활동 정보를 삭제한 과도한 요약 복구",
    ),
    CorrectionRule(
        "충북",
        "5430",
        "restore_original",
        "`온가족`을 `온가가족`으로 바꾼 신규 오탈자 복구",
    ),
    CorrectionRule(
        "전남",
        "6982",
        "restore_original",
        "산전 검진쿠폰 제공 내용을 사업명으로 대체한 의미변경 복구",
    ),
    CorrectionRule(
        "경북",
        "7546",
        "strip_source_label",
        "사업명과 `원문:` 메타 라벨이 추가된 정제문에서 라벨·사업명 제거",
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


def load_2023_wide(data_root: Path) -> pd.DataFrame:
    frames = [
        _read_csv(data_root / region / f"2023_{region}_세부사업_정제.csv") for region in REGIONS
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["지역", "원본행"]).any():
        raise ValueError("2023 wide에 지역·원본행 중복이 있습니다.")
    return combined


def apply_corrections(data_root: Path, audit: pd.DataFrame) -> pd.DataFrame:
    """확정 7건을 checkpoint·wide·long에 반영하고 14건 검토표를 반환한다."""
    wide = load_2023_wide(data_root)
    checkpoint_path = data_root / "2023_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    rules = {(rule.region, rule.source_row): rule for rule in CORRECTIONS}
    targets = wide.loc[wide.apply(lambda row: (row["지역"], row["원본행"]) in rules, axis=1)].copy()
    if len(targets) != len(CORRECTIONS):
        raise ValueError(f"2023 확정 오류 키 누락: {len(targets)}/{len(CORRECTIONS)}")

    normalized_wide = wide.copy()
    normalized_checkpoint = checkpoint.copy()
    normalized_targets = targets.copy()
    for frame in (normalized_wide, normalized_checkpoint, normalized_targets):
        frame["주요내용_정제"] = frame["주요내용_정제"].apply(dedup_label)
    checkpoint_indices = resolve_checkpoint_indices(
        normalized_wide,
        normalized_checkpoint,
        normalized_targets,
    )

    review = audit.copy()
    review["사람검토결과"] = "의미보존 확인"
    review["조치내용"] = "정제값 유지"
    review["수정전_주요내용_정제"] = review["주요내용_정제"]
    review["수정후_주요내용_정제"] = review["주요내용_정제"]
    review["TFIDF_영향"] = "없음"
    region_updates: dict[str, tuple[Path, pd.DataFrame, Path, pd.DataFrame]] = {}

    for wide_index, target in targets.iterrows():
        key = (target["지역"], target["원본행"])
        rule = rules[key]
        new_cleaned = corrected_text(
            rule,
            original=str(target["주요내용"]),
            cleaned=str(target["주요내용_정제"]),
        )
        checkpoint_index = checkpoint_indices[wide_index]
        if dedup_label(checkpoint.loc[checkpoint_index, "주요내용_정제"]) != dedup_label(
            target["주요내용_정제"]
        ):
            raise ValueError(f"checkpoint 연결값 불일치: {key}")
        checkpoint.loc[checkpoint_index, "주요내용_정제"] = new_cleaned

        region = target["지역"]
        wide_path = data_root / region / f"2023_{region}_세부사업_정제.csv"
        long_path = data_root / region / f"2023_{region}_세부사업_정제_long.csv"
        if region not in region_updates:
            region_updates[region] = (
                wide_path,
                _read_csv(wide_path),
                long_path,
                _read_csv(long_path),
            )
        _, region_wide, _, region_long = region_updates[region]
        wide_mask = region_wide["원본행"].eq(target["원본행"]) & region_wide["세부사업명"].eq(
            target["세부사업명"]
        )
        long_mask = region_long["원본행"].eq(target["원본행"]) & region_long["세부사업명"].eq(
            target["세부사업명"]
        )
        if int(wide_mask.sum()) != 1 or int(long_mask.sum()) != 2:
            raise ValueError(f"wide·long 키 건수 불일치: {key}")
        if not region_wide.loc[wide_mask, "주요내용_정제"].eq(target["주요내용_정제"]).all():
            raise ValueError(f"wide 수정전 값 불일치: {key}")
        if not region_long.loc[long_mask, "주요내용_정제"].eq(target["주요내용_정제"]).all():
            raise ValueError(f"long 수정전 값 불일치: {key}")
        region_wide.loc[wide_mask, "주요내용_정제"] = new_cleaned
        region_long.loc[long_mask, "주요내용_정제"] = new_cleaned
        review_mask = review["지역"].eq(region) & review["원본행"].eq(target["원본행"])
        if int(review_mask.sum()) != 1:
            raise ValueError(f"감사표 키 연결 실패: {key}")
        review.loc[review_mask, "사람검토결과"] = "의미보존 오류 수정"
        review.loc[review_mask, "조치내용"] = rule.reason
        review.loc[review_mask, "수정후_주요내용_정제"] = new_cleaned
        review.loc[review_mask, "TFIDF_영향"] = "예측 입력 변경·후속 재예측 필요"

    # 모든 키·기존값 검증이 끝난 뒤에만 산출물을 기록한다.
    for wide_path, region_wide, long_path, region_long in region_updates.values():
        _write_csv(region_wide, wide_path)
        _write_csv(region_long, long_path)
    _write_csv(checkpoint, checkpoint_path, index=True)
    return review


def refresh_prediction_targets(data_root: Path) -> None:
    existing = next(data_root.rglob("2016_2020_2022_2024_TFIDF_분류대상_통합.csv"))
    combined, qa = consolidate_prediction_targets(data_root)
    save_outputs(combined, qa, existing.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--audit-input",
        type=Path,
        default=Path("reports/yearly/2023/2023_LLM_의미보존_감사.csv"),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("reports/yearly/2023/2023_LLM_의미보존_전수검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = _read_csv(args.audit_input)
    if len(audit) != 14:
        raise ValueError(f"2023 감사 후보는 14건이어야 합니다: {len(audit)}")
    review = apply_corrections(args.data_root, audit)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(review, args.review_output)
    refresh_prediction_targets(args.data_root)
    print(f"2023 후보 전수검토: {len(review):,}건")
    print(f"의미보존 확인: {review['사람검토결과'].eq('의미보존 확인').sum():,}건")
    print(f"의미보존 오류 수정: {review['사람검토결과'].eq('의미보존 오류 수정').sum():,}건")
    print("TF-IDF 예측 입력 통합본 재생성 완료")


if __name__ == "__main__":
    main()
