"""이슈 #81 파이프라인 추출값을 #73 전수 감사가 이미 검증한 wide 파일과 대조한다.

#73(`reports/20260804_2016_2024_예산_행_오연결_전수_감사.md`)은 `data/interim/
{지역}/{연도}_{지역}_세부사업_정제.csv`(wide, 153개 파일 57,979행)를 원본 Table 1과
대조해 예산 행 오연결이 0건임을 이미 확인했다. 하지만 그 감사는 이 이슈(#81)의
`src/provisional/loader.py` 추출 경로(`_classify_rows`, `select_total_budget_rows`
등 독자적인 재구현)를 검증한 게 아니다.

이 스크립트는 #81의 `read_raw_file_list()` 출력을 그 #73-감사 완료 wide 파일과
`지역·연도·원본행` 키로 직접 대조해, #81의 추출 로직이 이미 검증된 값과 같은
세부사업명·예산을 만드는지 확인한다. 완전히 새로 원본을 재검증하는 게 아니라
"두 독립 구현이 같은 답을 내는가"를 보는 교차검증이다. 파이프라인 코드는 호출만
하고 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

from src.provisional.loader import EXPECTED_YEARS, STANDARD_REGIONS, read_raw_file_list

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = REPO_ROOT / "data" / "interim"
QA_OUTPUT_PATH = REPO_ROOT / "reports" / "20260807_잠정재정패널_73감사wide_대조_QA.csv"
MISMATCH_OUTPUT_PATH = REPO_ROOT / "reports" / "20260807_잠정재정패널_73감사wide_불일치_상세.csv"

# 이미 근거를 남기고 의도적으로 보정한 값 — 대조에서 "알려진 차이"로 분리한다.
# (원본행 8200 "여성지도자 육성" 2018년 예산 -20 → 20, 재정팀 확인 2026-08-07,
# reports/methodology/20260807_경북_2018_여성지도자육성_예산_보정_QA.md)
KNOWN_CORRECTIONS = {("경북", 2018, 8200)}

# 감사wide 파일엔 있지만 §81 leaf엔 의도적으로 없는 키 — "소 계"(소계 행)라
# leaf(세부사업) 추출 대상이 아니다. 라벨 워크북에서도 같은 행이
# 명칭_내용_불일치_복합대응="소계 표기"로 표시돼 있다(build_provisional_area_labels.py
# 참고). 세부사업명까지 확인해 진짜 이 행인지 검증한 뒤에만 예외로 인정한다.
KNOWN_SUBTOTAL_ONLY_IN_AUDITED_WIDE = {("전남", 2017, 9505): "소 계"}


def load_audited_wide(interim_dir: Path, *, regions: list[str], years: list[int]) -> pd.DataFrame:
    """#73이 감사한 지역·연도별 wide 파일을 전부 읽어 하나로 합친다."""
    frames = []
    for region in regions:
        region_dir = interim_dir / region
        if not region_dir.is_dir():
            raise FileNotFoundError(f"wide 파일 디렉터리가 없습니다: {region_dir}")
        for year in years:
            expected_name = f"{year}_{region}_세부사업_정제.csv"
            match = _find_normalized(region_dir, expected_name)
            if match is None:
                raise FileNotFoundError(
                    f"wide 파일을 찾지 못했습니다: {region_dir}/{expected_name}"
                )
            frame = pd.read_csv(match)
            required = {"연도", "지역", "세부사업명", "당해예산", "원본행"}
            missing = required - set(frame.columns)
            if missing:
                raise KeyError(f"{match}: 필수 열 누락 {sorted(missing)}")
            frames.append(frame[["연도", "지역", "세부사업명", "당해예산", "원본행"]])
    combined = pd.concat(frames, ignore_index=True)
    combined["원본행"] = combined["원본행"].astype("Int64")
    return combined


def _find_normalized(directory: Path, expected_name_nfc: str) -> Path | None:
    """파일명 유니코드 정규화(NFC/NFD) 차이를 무시하고 찾는다(#73에서 이미 확인된 문제)."""
    for candidate in directory.iterdir():
        if unicodedata.normalize("NFC", candidate.name) == expected_name_nfc:
            return candidate
    return None


def compare_detail_to_audited_wide(
    detail: pd.DataFrame, audited: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§81 추출(detail)과 #73 감사 wide를 키 대조해 QA 요약과 불일치 상세를 만든다."""
    keys = ["지역", "연도", "원본행"]
    left = detail[[*keys, "세부사업명", "예산액"]].copy()
    left["원본행"] = left["원본행"].astype("Int64")
    right = audited.rename(columns={"당해예산": "예산액_감사wide"})

    merged = left.merge(
        right.rename(columns={"세부사업명": "세부사업명_감사wide"}),
        on=keys,
        how="outer",
        indicator=True,
    )

    only_in_81 = merged.loc[merged["_merge"].eq("left_only")]
    only_in_audit = merged.loc[merged["_merge"].eq("right_only")].copy()

    only_in_audit["키"] = list(
        zip(only_in_audit["지역"], only_in_audit["연도"], only_in_audit["원본행"], strict=False)
    )
    is_known_subtotal = only_in_audit["키"].map(KNOWN_SUBTOTAL_ONLY_IN_AUDITED_WIDE)
    name_matches_expected = only_in_audit["세부사업명_감사wide"] == is_known_subtotal
    unexpected_only_in_audit = only_in_audit.loc[is_known_subtotal.isna() | ~name_matches_expected]

    both = merged.loc[merged["_merge"].eq("both")].copy()

    name_mismatch = both["세부사업명"] != both["세부사업명_감사wide"]
    budget_mismatch = (both["예산액"] - both["예산액_감사wide"]).abs().gt(1e-6)

    both["키"] = list(zip(both["지역"], both["연도"], both["원본행"], strict=False))
    is_known = both["키"].isin(KNOWN_CORRECTIONS)
    unexpected_budget_mismatch = budget_mismatch & ~is_known

    checks = [
        {"검사항목": "§81에만 있는 키(감사wide에 없음)", "기대값": 0, "실제값": len(only_in_81)},
        {
            "검사항목": "감사wide에만 있는 키(§81에 없음, 알려진 소계 행 제외)",
            "기대값": 0,
            "실제값": len(unexpected_only_in_audit),
        },
        {"검사항목": "세부사업명 불일치", "기대값": 0, "실제값": int(name_mismatch.sum())},
        {
            "검사항목": "예산 불일치(알려진 보정 제외)",
            "기대값": 0,
            "실제값": int(unexpected_budget_mismatch.sum()),
        },
        {
            "검사항목": "예산 불일치(알려진 보정만, 참고용)",
            "기대값": len(KNOWN_CORRECTIONS),
            "실제값": int((budget_mismatch & is_known).sum()),
        },
        {
            "검사항목": "감사wide에만 있는 키(알려진 소계 행만, 참고용)",
            "기대값": len(KNOWN_SUBTOTAL_ONLY_IN_AUDITED_WIDE),
            "실제값": len(only_in_audit) - len(unexpected_only_in_audit),
        },
    ]
    for check in checks:
        check["판정"] = "PASS" if check["기대값"] == check["실제값"] else "FAIL"
    qa = pd.DataFrame(checks)

    mismatches = pd.concat(
        [
            only_in_81.assign(불일치유형="§81에만 존재"),
            unexpected_only_in_audit.assign(불일치유형="감사wide에만 존재(미확인)"),
            only_in_audit.loc[~only_in_audit.index.isin(unexpected_only_in_audit.index)].assign(
                불일치유형="감사wide에만 존재(알려진 소계 행)"
            ),
            both.loc[name_mismatch].assign(불일치유형="세부사업명 불일치"),
            both.loc[budget_mismatch].assign(
                불일치유형=both.loc[budget_mismatch, "키"]
                .isin(KNOWN_CORRECTIONS)
                .map({True: "예산 불일치(알려진 보정)", False: "예산 불일치(미확인)"})
            ),
        ],
        ignore_index=True,
    )
    return qa, mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", type=Path, nargs="+", required=True)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--qa-output", type=Path, default=QA_OUTPUT_PATH)
    parser.add_argument("--mismatch-output", type=Path, default=MISMATCH_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = read_raw_file_list(
        args.input_files, expected_regions=STANDARD_REGIONS, expected_years=EXPECTED_YEARS
    )
    audited = load_audited_wide(
        args.interim_dir, regions=list(STANDARD_REGIONS), years=list(EXPECTED_YEARS)
    )

    qa, mismatches = compare_detail_to_audited_wide(detail, audited)

    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    mismatches.to_csv(args.mismatch_output, index=False, encoding="utf-8-sig")
    print(qa.to_string(index=False))
    print(f"QA 저장: {args.qa_output}")
    print(f"불일치 상세({len(mismatches)}건): {args.mismatch_output}")

    failures = qa.loc[qa["판정"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(f"#73 감사 wide 대조 실패: {failures.to_dict('records')}")


if __name__ == "__main__":
    main()
