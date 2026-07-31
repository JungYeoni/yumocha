"""2024년 의미보존 후보 전수검토의 확정 오류를 관련 산출물에 반영한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.apply_2023_blank_input_recovery import resolve_checkpoint_indices
from scripts.audit_llm_semantic_preservation import REGIONS
from scripts.consolidate_tfidf_prediction_targets import (
    consolidate_prediction_targets,
    save_outputs,
)
from src.features.text_match import dedup_label


@dataclass(frozen=True)
class CorrectionRule:
    region: str
    source_row: str
    action: str
    reason: str


CORRECTIONS = (
    CorrectionRule(
        "서울",
        "194",
        "strip_source_label",
        "사업명이 추가되고 `원문:` 라벨이 삽입된 정제문에서 라벨·사업명 제거",
    ),
    CorrectionRule(
        "인천",
        "1455",
        "strip_source_label",
        "`세부사업명(참고용):`과 `원문:` 메타 라벨이 결과에 포함된 오류 복구",
    ),
    CorrectionRule(
        "세종",
        "2654",
        "restore_original",
        "청소년 수련·체험활동 제공 의미를 삭제한 과도한 요약 복구",
    ),
    CorrectionRule(
        "경기",
        "3911",
        "restore_original",
        "원문의 금리 기호 `%`를 삭제한 수량 의미변경 복구",
    ),
    CorrectionRule(
        "전남",
        "6102",
        "restore_original",
        "산전 검진쿠폰 제공 내용을 사업명으로 대체한 의미변경 복구",
    ),
    CorrectionRule(
        "전남",
        "6121",
        "restore_original",
        "원문에 없던 `%` 단위를 추가한 수량 의미변경 복구",
    ),
    CorrectionRule(
        "경북",
        "6884",
        "restore_original",
        "여가활동 지원 내용을 삭제하고 사업명으로 축약한 의미변경 복구",
    ),
    CorrectionRule(
        "경북",
        "6945",
        "strip_source_label",
        "사업명이 추가되고 `원문:` 라벨이 삽입된 정제문에서 라벨·사업명 제거",
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


def load_2024_wide(data_root: Path) -> pd.DataFrame:
    frames = [
        _read_csv(data_root / region / f"2024_{region}_세부사업_정제.csv") for region in REGIONS
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["지역", "원본행"]).any():
        raise ValueError("2024 wide에 지역·원본행 중복이 있습니다.")
    return combined


def corrected_text(rule: CorrectionRule, *, original: str, cleaned: str) -> str:
    """규칙에 따라 검토 확정 정제문을 반환한다."""
    if rule.action == "restore_original":
        return original
    if rule.action == "strip_source_label":
        marker = "원문:"
        if marker not in cleaned:
            raise ValueError(f"`원문:` 라벨을 찾지 못했습니다: {rule.region}/{rule.source_row}")
        return cleaned.split(marker, 1)[1].strip()
    raise ValueError(f"알 수 없는 수정 action: {rule.action}")


def apply_corrections(data_root: Path, audit: pd.DataFrame) -> pd.DataFrame:
    """확정 오류를 checkpoint·wide·long에 반영하고 36건 검토표를 반환한다."""
    wide = load_2024_wide(data_root)
    checkpoint_path = data_root / "2024_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    correction_keys = {(rule.region, rule.source_row): rule for rule in CORRECTIONS}
    target_mask = wide.apply(
        lambda row: (row["지역"], row["원본행"]) in correction_keys,
        axis=1,
    )
    targets = wide.loc[target_mask].copy()
    if len(targets) != len(CORRECTIONS):
        raise ValueError(f"2024 확정 오류 키 누락: {len(targets)}/{len(CORRECTIONS)}")

    normalized_wide = wide.copy()
    normalized_checkpoint = checkpoint.copy()
    normalized_targets = targets.copy()
    normalized_wide["주요내용_정제"] = normalized_wide["주요내용_정제"].apply(dedup_label)
    normalized_checkpoint["주요내용_정제"] = normalized_checkpoint["주요내용_정제"].apply(
        dedup_label
    )
    normalized_targets["주요내용_정제"] = normalized_targets["주요내용_정제"].apply(dedup_label)
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

    for wide_index, target in targets.iterrows():
        key = (target["지역"], target["원본행"])
        rule = correction_keys[key]
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
        wide_path = data_root / region / f"2024_{region}_세부사업_정제.csv"
        long_path = data_root / region / f"2024_{region}_세부사업_정제_long.csv"
        region_wide = _read_csv(wide_path)
        region_long = _read_csv(long_path)
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
        _write_csv(region_wide, wide_path)
        _write_csv(region_long, long_path)

        review_mask = review["지역"].eq(region) & review["원본행"].eq(target["원본행"])
        if int(review_mask.sum()) != 1:
            raise ValueError(f"감사표 키 연결 실패: {key}")
        review.loc[review_mask, "사람검토결과"] = "의미보존 오류 수정"
        review.loc[review_mask, "조치내용"] = rule.reason
        review.loc[review_mask, "수정후_주요내용_정제"] = new_cleaned
        review.loc[review_mask, "TFIDF_영향"] = "예측 입력 변경·후속 재예측 필요"

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
        default=Path("reports/yearly/2024/2024_LLM_의미보존_감사.csv"),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("reports/yearly/2024/2024_LLM_의미보존_전수검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = _read_csv(args.audit_input)
    if len(audit) != 36:
        raise ValueError(f"2024 감사 후보는 36건이어야 합니다: {len(audit)}")
    review = apply_corrections(args.data_root, audit)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(review, args.review_output)
    refresh_prediction_targets(args.data_root)
    print(f"2024 후보 전수검토: {len(review):,}건")
    print(f"의미보존 확인: {review['사람검토결과'].eq('의미보존 확인').sum():,}건")
    print(f"의미보존 오류 수정: {review['사람검토결과'].eq('의미보존 오류 수정').sum():,}건")
    print("TF-IDF 예측 입력 통합본 재생성 완료")


if __name__ == "__main__":
    main()
