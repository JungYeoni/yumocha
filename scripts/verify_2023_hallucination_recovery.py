"""2023년 LLM 환각 복구 상태를 체크포인트·wide·long에서 재검증한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.audit_llm_semantic_preservation import REGIONS
from src.features.review_keys import normalize_review_keys

KNOWN_FAKE_SIGNATURES = (
    "1,200,000,000원",
    "043-123-4567",
    "youth@boeun.go.kr",
)
KNOWN_RECOVERED_EXAMPLES = (
    ("충북", "5030", "아동급식 확대 지원"),
    ("충북", "5033", "지역아동센터 냉난방비 지원"),
    ("충북", "5339", "보은군 청년 활성화 공모사업"),
)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        dtype={"원본행": "string"},
        **kwargs,
    )


def _load_region_files(data_root: Path, *, long: bool) -> pd.DataFrame:
    suffix = "_세부사업_정제_long.csv" if long else "_세부사업_정제.csv"
    frames = [_read_csv(data_root / region / f"2023_{region}{suffix}") for region in REGIONS]
    return pd.concat(frames, ignore_index=True)


def verify_recovery(data_root: Path) -> dict[str, object]:
    checkpoint_path = data_root / "2023_llm_정제_체크포인트.csv"
    checkpoint = _read_csv(checkpoint_path, index_col=0)
    wide = _load_region_files(data_root, long=False)
    long = _load_region_files(data_root, long=True)
    wide = normalize_review_keys(wide.assign(연도=2023)).drop(columns="연도")
    long = normalize_review_keys(long.assign(연도=2023)).drop(columns="연도")

    key_columns = ["지역", "원본행"]
    wide_duplicate_keys = int(wide.duplicated(key_columns, keep=False).sum())
    long_group_sizes = long.groupby(key_columns, dropna=False).size()
    invalid_long_group_count = int(long_group_sizes.ne(2).sum())

    long_cleaned = (
        long.groupby(key_columns, dropna=False)["주요내용_정제"]
        .agg(lambda values: tuple(dict.fromkeys(values)))
        .reset_index()
    )
    merged = wide.merge(
        long_cleaned,
        on=key_columns,
        how="outer",
        indicator=True,
        suffixes=("_wide", "_long"),
    )
    wide_only_key_count = int(merged["_merge"].eq("left_only").sum())
    long_only_key_count = int(merged["_merge"].eq("right_only").sum())
    wide_long_mismatch = int(
        merged.loc[merged["_merge"].eq("both")]
        .apply(
            lambda row: row["주요내용_정제_long"] != (row["주요내용_정제_wide"],),
            axis=1,
        )
        .sum()
    )

    checkpoint_values = Counter(checkpoint["주요내용_정제"].astype(str))
    wide_values = Counter(wide["주요내용_정제"].astype(str))
    checkpoint_wide_multiset_match = checkpoint_values == wide_values
    checkpoint_only = checkpoint_values - wide_values
    wide_only = wide_values - checkpoint_values

    signature_counts: dict[str, dict[str, int]] = {}
    for signature in KNOWN_FAKE_SIGNATURES:
        signature_counts[signature] = {
            "checkpoint": int(
                checkpoint["주요내용_정제"].astype(str).str.contains(signature, regex=False).sum()
            ),
            "wide": int(
                wide["주요내용_정제"].astype(str).str.contains(signature, regex=False).sum()
            ),
            "long": int(
                long["주요내용_정제"].astype(str).str.contains(signature, regex=False).sum()
            ),
        }

    recovered_examples: list[dict[str, object]] = []
    for region, source_row, name in KNOWN_RECOVERED_EXAMPLES:
        matched = wide.loc[
            wide["지역"].eq(region) & wide["원본행"].eq(source_row) & wide["세부사업명"].eq(name)
        ]
        recovered_examples.append(
            {
                "region": region,
                "source_row": source_row,
                "name": name,
                "matched_rows": len(matched),
                "original_blank": bool(
                    len(matched) == 1 and not str(matched.iloc[0]["주요내용"]).strip()
                ),
                "cleaned_blank": bool(
                    len(matched) == 1 and not str(matched.iloc[0]["주요내용_정제"]).strip()
                ),
            }
        )

    blank_original = wide["주요내용"].astype(str).str.strip().eq("")
    nonblank_cleaned = wide["주요내용_정제"].astype(str).str.strip().ne("")
    blank_input_changed = wide.loc[blank_original & nonblank_cleaned]

    known_incident_recovery_supported = (
        all(not any(counts.values()) for counts in signature_counts.values())
        and all(
            example["matched_rows"] == 1 and example["original_blank"] and example["cleaned_blank"]
            for example in recovered_examples
        )
        and not wide_long_mismatch
        and not wide_only_key_count
        and not long_only_key_count
        and not wide_duplicate_keys
        and not invalid_long_group_count
    )

    return {
        "checkpoint_rows": len(checkpoint),
        "wide_rows": len(wide),
        "long_rows": len(long),
        "wide_duplicate_keys": wide_duplicate_keys,
        "invalid_long_group_count": invalid_long_group_count,
        "wide_long_cleaned_mismatch": wide_long_mismatch,
        "wide_only_key_count": wide_only_key_count,
        "long_only_key_count": long_only_key_count,
        "checkpoint_wide_cleaned_multiset_match": checkpoint_wide_multiset_match,
        "checkpoint_only_value_count": sum(checkpoint_only.values()),
        "wide_only_value_count": sum(wide_only.values()),
        "checkpoint_only_samples": list(checkpoint_only.keys())[:5],
        "wide_only_samples": list(wide_only.keys())[:5],
        "known_fake_signature_counts": signature_counts,
        "known_recovered_examples": recovered_examples,
        "blank_original_rows": int(blank_original.sum()),
        "blank_original_cleaned_nonblank_rows": len(blank_input_changed),
        "blank_original_cleaned_blank_rows": int((blank_original & ~nonblank_cleaned).sum()),
        "blank_input_changed_samples": blank_input_changed[
            ["지역", "원본행", "세부사업명", "주요내용_정제"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "known_incident_recovery_supported": known_incident_recovery_supported,
        "artifact_consistency_complete": (
            checkpoint_wide_multiset_match
            and not wide_long_mismatch
            and not wide_duplicate_keys
            and not invalid_long_group_count
            and not wide_only_key_count
            and not long_only_key_count
        ),
        "broader_blank_input_recovery_complete": len(blank_input_changed) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/yearly/2023/2023_LLM_환각_복구_재검증.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_recovery(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
