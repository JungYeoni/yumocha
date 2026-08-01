"""2020년 의미보존 후보 3건의 메타 라벨 삽입 오류를 복구한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.consolidate_tfidf_prediction_targets import (
    consolidate_prediction_targets,
    save_outputs,
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


def strip_source_label(cleaned: str) -> str:
    """정제문에 삽입된 사업명과 ``원문:`` 접두부를 제거한다."""
    marker = "원문:"
    if marker not in cleaned:
        # 이미 수정된 산출물에 다시 실행해도 값이 바뀌지 않게 한다.
        return cleaned
    return cleaned.split(marker, 1)[1].strip()


def apply_corrections(data_root: Path, audit: pd.DataFrame) -> pd.DataFrame:
    """감사 3건을 checkpoint·wide·long에 반영하고 검토표를 반환한다."""
    checkpoint_path = data_root / "2020_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    review = audit.copy()
    review["수정전_주요내용_정제"] = review["주요내용_정제"]
    review["수정후_주요내용_정제"] = ""
    review["사람검토결과"] = "의미보존 오류 수정"
    review["조치내용"] = "사업명과 `원문:` 메타 라벨 접두부 제거"
    review["TFIDF_영향"] = "예측 입력 변경·후속 재예측 필요"
    region_updates: dict[str, tuple[Path, pd.DataFrame, Path, pd.DataFrame]] = {}

    for review_index, row in review.iterrows():
        new_cleaned = strip_source_label(str(row["주요내용_정제"]))
        checkpoint_mask = (
            checkpoint["지역"].eq(row["지역"])
            & checkpoint["원본행"].eq(row["원본행"])
            & checkpoint["세부사업명"].eq(row["세부사업명"])
        )
        if int(checkpoint_mask.sum()) != 1:
            raise ValueError(f"checkpoint 키 건수 불일치: {row['지역']}/{row['원본행']}")
        if not checkpoint.loc[checkpoint_mask, "주요내용_정제"].eq(row["주요내용_정제"]).all():
            raise ValueError(f"checkpoint 수정전 값 불일치: {row['지역']}/{row['원본행']}")
        checkpoint.loc[checkpoint_mask, "주요내용_정제"] = new_cleaned
        checkpoint.loc[checkpoint_mask, "LLM_상태"] = "보존위반"
        checkpoint.loc[checkpoint_mask, "LLM_오류유형"] = "PreservationViolation"
        checkpoint.loc[checkpoint_mask, "LLM_보존위반"] = "정제문 원문 라벨 추가"

        region = row["지역"]
        wide_path = data_root / region / f"2020_{region}_세부사업_정제.csv"
        long_path = data_root / region / f"2020_{region}_세부사업_정제_long.csv"
        if region not in region_updates:
            region_updates[region] = (
                wide_path,
                _read_csv(wide_path),
                long_path,
                _read_csv(long_path),
            )
        _, wide, _, long = region_updates[region]
        wide_mask = wide["원본행"].eq(row["원본행"]) & wide["세부사업명"].eq(row["세부사업명"])
        long_mask = long["원본행"].eq(row["원본행"]) & long["세부사업명"].eq(row["세부사업명"])
        if int(wide_mask.sum()) != 1 or int(long_mask.sum()) != 2:
            raise ValueError(f"wide·long 키 건수 불일치: {region}/{row['원본행']}")
        if not wide.loc[wide_mask, "주요내용_정제"].eq(row["주요내용_정제"]).all():
            raise ValueError(f"wide 수정전 값 불일치: {region}/{row['원본행']}")
        if not long.loc[long_mask, "주요내용_정제"].eq(row["주요내용_정제"]).all():
            raise ValueError(f"long 수정전 값 불일치: {region}/{row['원본행']}")
        wide.loc[wide_mask, "주요내용_정제"] = new_cleaned
        long.loc[long_mask, "주요내용_정제"] = new_cleaned
        review.loc[review_index, "수정후_주요내용_정제"] = new_cleaned

    # 모든 키·기존값 검증이 끝난 뒤에만 산출물을 기록한다.
    for wide_path, wide, long_path, long in region_updates.values():
        _write_csv(wide, wide_path)
        _write_csv(long, long_path)
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
        default=Path("reports/yearly/2020/2020_LLM_의미보존_감사.csv"),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("reports/yearly/2020/2020_LLM_의미보존_전수검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = _read_csv(args.audit_input)
    if len(audit) != 3:
        raise ValueError(f"2020 감사 후보는 3건이어야 합니다: {len(audit)}")
    review = apply_corrections(args.data_root, audit)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(review, args.review_output)
    refresh_prediction_targets(args.data_root)
    print(f"2020 후보 전수검토·오류 수정: {len(review):,}건")
    print("checkpoint·wide·long 및 TF-IDF 예측 입력 반영 완료")


if __name__ == "__main__":
    main()
