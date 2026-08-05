"""#70 기준패널에 가족친화 인증기업 공식 관측값을 반영해 최신 상태로 만든다.

`apply_confirmed_observations()`(build_family_friendly_candidates.py)는 21개 지표 wide 표
전용이라 wide→long 변환이 끝난 통합 패널(`구조환경지표_28개_보간전_기준패널.csv`)에는 직접
쓸 수 없다. 이 스크립트는 같은 37건 반영표(`20260804_가족친화_공식관측_반영값.csv`, 지역
34건 + 2024 정정 3건)를 롱 패널에 적용하는 대응 함수를 새로 두고, 이미 롱 패널용으로 준비된
전국 5건 함수(`apply_national_observations_to_panel`)와 함께 실행해 패널을 최신 상태로 만든다.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_family_friendly_candidates import CONFIRMED_PATH, INDICATOR_ID, PANEL_PATH
from scripts.build_family_friendly_national_candidates import (
    CANDIDATE_PATH as NATIONAL_CANDIDATE_PATH,
)
from scripts.build_family_friendly_national_candidates import apply_national_observations_to_panel

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "reports" / "20260805_가족친화_패널반영_통합감사.csv"
REPORT_PATH = (
    REPO_ROOT / "reports" / "methodology" / "20260805_구조환경지표_패널_공식관측_최신화_QA.md"
)

RESTORATION_TYPE = "공식 원자료 직접 복원"
CORRECTION_TYPE = "공식 집계 오류 정정"
ALLOWED_APPLICATION_TYPES = {RESTORATION_TYPE, CORRECTION_TYPE}
EXPECTED_REGIONAL_ROWS = 37
EXPECTED_CORRECTION_ROWS = 3
EXPECTED_RESTORATION_ROWS = EXPECTED_REGIONAL_ROWS - EXPECTED_CORRECTION_ROWS


def apply_regional_confirmed_observations_to_panel(
    panel: pd.DataFrame, confirmed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """37건 공식 관측 반영표(지역 34건 + 2024 정정 3건)를 28개 지표 롱 패널에 적용한다.

    복원(`공식 원자료 직접 복원`) 대상은 기존 값이 결측이거나 이미 후보값과 같아야 하고,
    정정(`공식 집계 오류 정정`) 대상은 기존 값이 있어도(알려진 오류값이므로) 덮어쓴다.
    """
    required = {"지역", "지표_id", "연도", "측정값"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing_columns, key=str)}")

    keys = ["지역", "지표_id", "연도"]
    if len(confirmed) != EXPECTED_REGIONAL_ROWS or confirmed.duplicated(keys).any():
        raise ValueError(f"공식 관측 반영표는 중복 없는 {EXPECTED_REGIONAL_ROWS}개 키여야 합니다.")
    if not confirmed["지표_id"].eq(INDICATOR_ID).all():
        raise ValueError("공식 관측 반영표에 다른 지표가 포함됐습니다.")
    if not confirmed["QA_상태"].eq("PASS").all():
        raise ValueError("QA를 통과하지 않은 공식 관측값은 반영할 수 없습니다.")
    invalid_types = set(confirmed["반영유형"]) - ALLOWED_APPLICATION_TYPES
    if invalid_types:
        raise ValueError(f"알 수 없는 반영유형입니다: {sorted(invalid_types)}")

    result = panel.copy()
    audit_rows = []
    for row in confirmed.to_dict("records"):
        mask = (
            result["지역"].eq(row["지역"])
            & result["지표_id"].eq(row["지표_id"])
            & result["연도"].eq(row["연도"])
        )
        matched = int(mask.sum())
        if matched != 1:
            raise ValueError(
                f"공식 관측 반영 키가 1개가 아닙니다({matched}개): {row['지역']}, {row['연도']}"
            )
        before = result.loc[mask, "측정값"].iloc[0]
        if row["반영유형"] == RESTORATION_TYPE:
            acceptable = pd.isna(before) or np.isclose(
                float(before), float(row["측정값"]), rtol=0, atol=1e-12
            )
            if not acceptable:
                raise ValueError(
                    f"직접 복원 대상의 기존 패널값이 예상과 다릅니다: {row['지역']}, {row['연도']}"
                )
        result.loc[mask, "측정값"] = float(row["측정값"])
        if "원본행존재" in result.columns:
            result.loc[mask, "원본행존재"] = True
        audit_rows.append(
            {
                "지역": row["지역"],
                "지표_id": row["지표_id"],
                "연도": row["연도"],
                "반영전값": before,
                "반영후값": float(row["측정값"]),
                "반영유형": row["반영유형"],
                "관측상태": row["관측상태"],
                "QA_상태": "PASS",
            }
        )
    return result, pd.DataFrame(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--confirmed", type=Path, default=CONFIRMED_PATH)
    parser.add_argument("--national-candidates", type=Path, default=NATIONAL_CANDIDATE_PATH)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="패널 원본 백업을 생략한다(기본은 백업 생성).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        raise FileNotFoundError(f"패널이 없습니다: {args.panel}")

    panel = pd.read_csv(args.panel, encoding="utf-8-sig")
    confirmed = pd.read_csv(args.confirmed, encoding="utf-8-sig")
    national_candidates = pd.read_csv(args.national_candidates, encoding="utf-8-sig")

    before_missing = int(panel["측정값"].isna().sum())

    updated, regional_audit = apply_regional_confirmed_observations_to_panel(panel, confirmed)
    updated, national_audit = apply_national_observations_to_panel(updated, national_candidates)

    after_missing = int(updated["측정값"].isna().sum())
    audit = pd.concat([regional_audit, national_audit], ignore_index=True)

    expected_audit_rows = EXPECTED_REGIONAL_ROWS + len(national_candidates)
    if len(audit) != expected_audit_rows:
        raise ValueError(f"반영 건수가 예상({expected_audit_rows}건)과 다릅니다: {len(audit)}건")

    expected_missing_reduction = EXPECTED_RESTORATION_ROWS + len(national_candidates)
    actual_missing_reduction = before_missing - after_missing
    if actual_missing_reduction != expected_missing_reduction:
        raise ValueError(
            f"신규 관측 반영으로 줄어든 결측 건수가 예상과 다릅니다: "
            f"실제={actual_missing_reduction}건, 기대={expected_missing_reduction}건 "
            f"(지역 복원 {EXPECTED_RESTORATION_ROWS}건 + 전국 {len(national_candidates)}건; "
            f"2024 정정 {EXPECTED_CORRECTION_ROWS}건은 이미 관측값이 있었으므로 결측 감소분에 포함 안 됨)"
        )

    if not args.no_backup:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = args.panel.with_name(f"{args.panel.stem}_백업_{timestamp}{args.panel.suffix}")
        shutil.copy2(args.panel, backup_path)
        print(f"백업 생성: {backup_path}")

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output, index=False, encoding="utf-8-sig")
    updated.to_csv(args.panel, index=False, encoding="utf-8-sig")

    report_lines = [
        "# 구조환경지표 기준패널 공식 관측 최신화 QA",
        "",
        f"- 처리 시각: {datetime.now(timezone.utc).isoformat()}",
        f"- 반영 전 결측: {before_missing:,}건",
        f"- 반영 후 결측: {after_missing:,}건",
        f"- 신규 반영 건수: {len(audit):,}건"
        f"(지역 복원 {EXPECTED_RESTORATION_ROWS} + 2024 정정 {EXPECTED_CORRECTION_ROWS}"
        f" + 전국 복원 {len(national_candidates)})",
        f"- 2024 정정 {EXPECTED_CORRECTION_ROWS}건은 기존 값이 결측이 아니었으므로 결측 감소분에는 포함되지 않음",
        "",
        "## 상세",
        "",
        f"- 감사 로그: `{args.audit_output.relative_to(REPO_ROOT)}`",
        f"- 패널: `{args.panel}` (in-place 갱신, 백업 파일 생성됨)",
    ]
    args.report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"반영 완료: 결측 {before_missing:,} → {after_missing:,}건")
    print(f"감사 로그: {args.audit_output}")


if __name__ == "__main__":
    main()
