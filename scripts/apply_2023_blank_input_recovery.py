"""2023년 빈 원문에서 생성된 LLM 정제문을 안전하게 복구한다."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.audit_llm_semantic_preservation import REGIONS
from scripts.consolidate_2021_area_labels import normalize_text
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


def load_2023_wide(data_root: Path) -> pd.DataFrame:
    """지역별 wide를 순서대로 읽고 원본 경로를 부착한다."""
    frames: list[pd.DataFrame] = []
    for region in REGIONS:
        path = data_root / region / f"2023_{region}_세부사업_정제.csv"
        frame = _read_csv(path)
        frame["_wide_path"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["지역", "원본행"]).any():
        raise ValueError("2023 wide에 지역·원본행 중복이 있습니다.")
    return combined


def find_blank_input_changes(wide: pd.DataFrame) -> pd.DataFrame:
    """원문은 비었지만 정제문이 생성된 행을 반환한다."""
    original_blank = wide["주요내용"].astype(str).str.strip().eq("")
    cleaned_nonblank = wide["주요내용_정제"].astype(str).str.strip().ne("")
    return wide.loc[original_blank & cleaned_nonblank].copy()


def resolve_checkpoint_indices(
    wide: pd.DataFrame,
    checkpoint: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[int, object]:
    """고유값 또는 전후 고유 앵커로 target wide 인덱스를 checkpoint에 연결한다."""
    wide_values = Counter(wide["주요내용_정제"].astype(str))
    checkpoint_values = Counter(checkpoint["주요내용_정제"].astype(str))
    unique_checkpoint_index = {
        value: index
        for index, value in checkpoint["주요내용_정제"].astype(str).items()
        if checkpoint_values[value] == 1 and wide_values[value] == 1
    }
    resolved: dict[int, object] = {}

    for wide_index, row in targets.iterrows():
        value = str(row["주요내용_정제"])
        candidates = checkpoint.index[checkpoint["주요내용_정제"].astype(str).eq(value)].tolist()
        if len(candidates) == 1:
            resolved[wide_index] = candidates[0]
            continue
        if not candidates:
            raise ValueError(f"체크포인트 값 없음: {row['지역']}/{row['원본행']}={value}")

        region = row["지역"]
        previous_anchor = next(
            (
                unique_checkpoint_index[str(wide.loc[index, "주요내용_정제"])]
                for index in range(wide_index - 1, -1, -1)
                if wide.loc[index, "지역"] == region
                and str(wide.loc[index, "주요내용_정제"]) in unique_checkpoint_index
            ),
            None,
        )
        next_anchor = next(
            (
                unique_checkpoint_index[str(wide.loc[index, "주요내용_정제"])]
                for index in range(wide_index + 1, len(wide))
                if wide.loc[index, "지역"] == region
                and str(wide.loc[index, "주요내용_정제"]) in unique_checkpoint_index
            ),
            None,
        )
        bounded = [
            index
            for index in candidates
            if (previous_anchor is None or index > previous_anchor)
            and (next_anchor is None or index < next_anchor)
        ]
        if len(bounded) != 1:
            raise ValueError(
                f"체크포인트 중복값 연결 실패: {region}/{row['원본행']} "
                f"candidates={candidates}, bounds=({previous_anchor}, {next_anchor})"
            )
        resolved[wide_index] = bounded[0]

    if len(set(resolved.values())) != len(resolved):
        raise ValueError("여러 복구 대상이 동일한 체크포인트 행에 연결됐습니다.")
    return resolved


def apply_recovery(
    data_root: Path,
) -> pd.DataFrame:
    """빈 원문 변경 행을 checkpoint·wide·long에서 빈 값으로 복구한다."""
    checkpoint_path = data_root / "2023_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    wide = load_2023_wide(data_root)
    targets = find_blank_input_changes(wide)
    if targets.empty:
        return pd.DataFrame()

    resolved = resolve_checkpoint_indices(wide, checkpoint, targets)
    review = targets[["연도", "지역", "원본행", "세부사업명", "주요내용", "주요내용_정제"]].rename(
        columns={"주요내용_정제": "복구전_주요내용_정제"}
    )
    review["복구후_주요내용_정제"] = ""
    review["검토결과"] = "빈 원문 환각 복구"
    review["판단근거"] = "원문 주요내용이 빈 값이므로 정제문도 빈 값을 유지해야 함"
    review["기존_분류텍스트"] = [
        f"{normalize_text(name)} {normalize_text(cleaned)}".strip()
        for name, cleaned in zip(
            review["세부사업명"],
            review["복구전_주요내용_정제"],
            strict=True,
        )
    ]
    review["복구후_분류텍스트"] = review["세부사업명"].map(normalize_text)
    review["TFIDF_영향"] = "예측 입력 변경·후속 재예측 필요"

    for wide_index, checkpoint_index in resolved.items():
        expected = wide.loc[wide_index, "주요내용_정제"]
        if checkpoint.loc[checkpoint_index, "주요내용_정제"] != expected:
            raise ValueError(
                f"체크포인트 연결값 불일치: {wide.loc[wide_index, '지역']}/"
                f"{wide.loc[wide_index, '원본행']}"
            )
        checkpoint.loc[checkpoint_index, "주요내용_정제"] = ""

    for region, region_targets in targets.groupby("지역", sort=False):
        wide_path = data_root / region / f"2023_{region}_세부사업_정제.csv"
        long_path = data_root / region / f"2023_{region}_세부사업_정제_long.csv"
        region_wide = _read_csv(wide_path)
        region_long = _read_csv(long_path)
        for _, target in region_targets.iterrows():
            key_mask = region_wide["원본행"].eq(target["원본행"]) & region_wide["세부사업명"].eq(
                target["세부사업명"]
            )
            long_mask = region_long["원본행"].eq(target["원본행"]) & region_long["세부사업명"].eq(
                target["세부사업명"]
            )
            if int(key_mask.sum()) != 1 or int(long_mask.sum()) != 2:
                raise ValueError(f"wide·long 키 건수 불일치: {region}/{target['원본행']}")
            if not region_wide.loc[key_mask, "주요내용_정제"].eq(target["주요내용_정제"]).all():
                raise ValueError(f"wide 복구전 값 불일치: {region}/{target['원본행']}")
            if not region_long.loc[long_mask, "주요내용_정제"].eq(target["주요내용_정제"]).all():
                raise ValueError(f"long 복구전 값 불일치: {region}/{target['원본행']}")
            region_wide.loc[key_mask, "주요내용_정제"] = ""
            region_long.loc[long_mask, "주요내용_정제"] = ""
        _write_csv(region_wide, wide_path)
        _write_csv(region_long, long_path)

    _write_csv(checkpoint, checkpoint_path, index=True)
    return review.reset_index(drop=True)


def refresh_prediction_targets(data_root: Path) -> None:
    """수정된 정제본으로 TF-IDF 예측 입력 통합본을 재생성한다."""
    existing = next(data_root.rglob("2016_2020_2022_2024_TFIDF_분류대상_통합.csv"))
    combined, qa = consolidate_prediction_targets(data_root)
    save_outputs(combined, qa, existing.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("reports/yearly/2023/2023_LLM_빈원문_환각_복구검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = apply_recovery(args.data_root)
    if review.empty and args.review_output.exists():
        review = _read_csv(args.review_output)
    elif review.empty:
        raise ValueError("복구 대상이 없고 기존 복구 검토표도 없습니다.")
    else:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(review, args.review_output)
    refresh_prediction_targets(args.data_root)
    print(f"빈 원문 환각 복구: {len(review):,}건")
    print(f"영향 지역: {review['지역'].nunique()}개")
    print("TF-IDF 예측 입력 통합본 재생성 완료")


if __name__ == "__main__":
    main()
