"""Build verified 2020-2021 family-friendly certification rate candidates.

This script intentionally produces a standalone candidate artifact. It never writes to the
missing-value policy, the baseline panel, or any completed/imputed panel.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = (
    REPO_ROOT
    / "data"
    / "raw"
    / "2. 구조환경지수 원데이터"
    / "3. 제주여성가족연구원 지표별 측정값(raw)"
    / "4-1.2. 가족친화인증기업 비율"
)
PANEL_PATH = REPO_ROOT / "data" / "processed" / "구조환경지표_28개_보간전_기준패널.csv"
MAPPING_PATH = REPO_ROOT / "reports" / "20260804_구조환경지표_결측정책_전수매핑.csv"
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_후보.csv"
SOURCE_METADATA_PATH = REPO_ROOT / "reports" / "20260804_가족친화_공식원본_메타데이터.csv"
QA_PATH = REPO_ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_QA.csv"
REPORT_PATH = REPO_ROOT / "reports" / "methodology" / "20260804_가족친화_2020_2021_직접복원_QA.md"

INDICATOR_ID = "family_friendly_certification_rate"
TARGET_YEARS = (2020, 2021)
UNRESOLVED_YEARS = (2016, 2017, 2019)
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
REGION_ALIASES = {
    "전국": "전국",
    "서울": "서울",
    "서울특별시": "서울",
    "부산": "부산",
    "부산광역시": "부산",
    "대구": "대구",
    "대구광역시": "대구",
    "인천": "인천",
    "인천광역시": "인천",
    "광주": "광주",
    "광주광역시": "광주",
    "대전": "대전",
    "대전광역시": "대전",
    "울산": "울산",
    "울산광역시": "울산",
    "세종": "세종",
    "세종특별자치시": "세종",
    "경기": "경기",
    "경기도": "경기",
    "강원": "강원",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충북": "충북",
    "충청북도": "충북",
    "충남": "충남",
    "충청남도": "충남",
    "전북": "전북",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전남": "전남",
    "전라남도": "전남",
    "경북": "경북",
    "경상북도": "경북",
    "경남": "경남",
    "경상남도": "경남",
    "제주": "제주",
    "제주도": "제주",
    "제주특별자치도": "제주",
}
ADDRESS_PREFIXES = sorted(REGION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)

OFFICIAL_ROSTERS: dict[int, dict[str, Any]] = {
    2020: {
        "provider": "여성가족부·한국건강가정진흥원 가족친화지원사업",
        "post_url": ("https://www.ffsb.kr/ffsbbod/bs/boardView.do?boardSeq=1&conSeq=1630"),
        "download_url": ("https://www.ffsb.kr/ffsbbod/boardFileDownAct.do?fileSeq=1349"),
        "downloaded_at": "2026-08-04T21:40:04+09:00",
        "reference_date": "2020-12-17",
        "file_name": "2020 가족친화인증기업(관) 리스트(2020년 12월 4340개).xlsx",
        "file_size": 211_042,
        "sha256": "e3bd6762f24a5b66588c242f3e4598f69a328950d8985e11a39e257438d43f64",
        "sheet_name": "Sheet1",
        "header": 1,
        "columns": {
            "serial": "연번",
            "first_year": "최초인증년도",
            "name": "기업(관)명",
            "category": "기업(관)분류",
            "region": "시도",
            "address": "주소",
        },
        "total": 4_340,
        "category_totals": {"대기업": 456, "중소기업": 2_839, "공공기관": 1_045},
        "address_region_mismatches": 0,
    },
    2021: {
        "provider": "여성가족부·한국건강가정진흥원 가족친화지원사업",
        "post_url": ("https://www.ffsb.kr/ffsbbod/bs/boardView.do?boardSeq=1&conSeq=1734"),
        "download_url": ("https://www.ffsb.kr/ffsbbod/boardFileDownAct.do?fileSeq=1495"),
        "downloaded_at": "2026-08-04T21:40:04+09:00",
        "reference_date": "2021-12-16",
        "file_name": "2021 가족친화인증(관) 현황(2021년 12월, 총4918개사).xlsx",
        "file_size": 242_583,
        "sha256": "af79a43769a4b7db59fa2446f832f181cff0f93e8c7978517a350c21dff14b1f",
        "sheet_name": "2021년 4918사",
        "header": 1,
        "columns": {
            "serial": "연번",
            "first_year": "최초인증년도",
            "name": "기업(관)명",
            "category": "기업(관)분류",
            "region": "시도",
            "address": "주소지",
        },
        "total": 4_918,
        "category_totals": {"대기업": 520, "중소기업": 3_317, "공공기관": 1_081},
        "address_region_mismatches": 1,
    },
}

DENOMINATOR_SOURCE = {
    "file_name": (
        "2017-2023_시도별__산업별__규모별__사업체수_및_종사자수_성별__20260715194147.csv.xlsx"
    ),
    "file_size": 11_852,
    "sha256": "9e85827feb15d98ecf0f9d9b3b8af9f9a5316b50ac4e1a763f75281decf00702",
    "table": "사업체노동실태현황: 시도별, 산업별, 규모별, 사업체수 및 종사자수(성별)",
    "selection": "산업분류별=전체; 규모별=전규모; 연도 열=해당 연도 사업체수 (개)",
}

REFERENCE_ROSTERS = {
    2018: {
        "file_name": "가족친화인증기업현황_20181231.csv.xlsx",
        "file_size": 120_569,
        "sha256": "dc5eab3fd5cc00ace2aea8b1ea7ec3af416662368c2ea5a0ff5239c809ec477e",
        "total": 3_328,
        "region_column": "시도",
    },
    2022: {
        "file_name": "가족친화인증기업현황_20221231.csv.xlsx",
        "file_size": 189_351,
        "sha256": "71ae5370af8a5d12942cdafca891129127d68892c7dfc6807b1b71ea4d1cbce7",
        "total": 5_415,
        "region_column": "지역",
    },
    2023: {
        "file_name": "가족친화인증기업현황_20231231.csv.xlsx",
        "file_size": 205_120,
        "sha256": "3a2de0e483edb98c275d1e5b61171438945977114a11d42a3fb20b45f7a73571",
        "total": 5_911,
        "region_column": "지역",
    },
}


def normalize_region(value: object) -> str | None:
    """Normalize a province name or an address prefix to the panel's 17-region labels."""
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    direct = REGION_ALIASES.get(cleaned)
    if direct is not None:
        return direct
    for prefix, normalized in ADDRESS_PREFIXES:
        if cleaned.startswith(prefix):
            return normalized
    return None


def file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_verified_bytes(path: Path, expected_size: int, expected_sha256: str) -> bytes:
    """Read once and reject any source whose frozen byte identity changed."""
    if not path.is_file():
        raise FileNotFoundError(f"원자료 파일이 없습니다: {path}")
    payload = path.read_bytes()
    actual_size = len(payload)
    actual_sha256 = file_sha256(payload)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(
            "원자료 무결성 검증 실패: "
            f"{path.name}, expected=({expected_size}, {expected_sha256}), "
            f"actual=({actual_size}, {actual_sha256})"
        )
    return payload


def add_qa(
    records: list[dict[str, object]],
    section: str,
    check: str,
    expected: object,
    actual: object,
    *,
    note: str = "",
) -> None:
    records.append(
        {
            "구분": section,
            "검증항목": check,
            "기대값": expected,
            "실제값": actual,
            "판정": "PASS" if expected == actual else "FAIL",
            "비고": note,
        }
    )


def load_official_roster(
    raw_dir: Path, year: int, qa_records: list[dict[str, object]]
) -> tuple[pd.DataFrame, pd.Series]:
    spec = OFFICIAL_ROSTERS[year]
    path = raw_dir / spec["file_name"]
    payload = read_verified_bytes(path, spec["file_size"], spec["sha256"])
    frame = pd.read_excel(io.BytesIO(payload), sheet_name=spec["sheet_name"], header=spec["header"])
    columns = spec["columns"]
    required = set(columns.values())
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{year} 명단 필수 열 누락: {sorted(missing)}")

    selected = frame[list(columns.values())].copy()
    selected.columns = list(columns)
    for column in ("name", "category", "region", "address"):
        selected[column] = selected[column].astype("string").str.strip()
    selected["serial"] = pd.to_numeric(selected["serial"], errors="raise").astype(int)
    selected["first_year"] = pd.to_numeric(selected["first_year"], errors="raise").astype(int)
    selected["normalized_region"] = selected["region"].map(normalize_region)
    selected["address_region"] = selected["address"].map(normalize_region)

    empty_address = int(selected["address"].isna().sum() + selected["address"].eq("").sum())
    unclassified = int(selected["normalized_region"].isna().sum())
    address_unclassified = int(selected["address_region"].isna().sum())
    address_mismatch = int(
        selected["normalized_region"].ne(selected["address_region"]).fillna(True).sum()
    )
    serial_duplicates = int(selected["serial"].duplicated().sum())
    exact_duplicates = int(selected.duplicated().sum())
    identity_duplicates = int(
        selected.duplicated(subset=["name", "category", "normalized_region", "address"]).sum()
    )
    name_duplicates = int(selected["name"].duplicated().sum())
    categories = selected["category"].value_counts().to_dict()
    region_counts = selected.groupby("normalized_region", dropna=False).size().sort_index()

    section = f"분자 {year}"
    add_qa(qa_records, section, "파일 크기 bytes", spec["file_size"], len(payload))
    add_qa(qa_records, section, "SHA-256", spec["sha256"], file_sha256(payload))
    add_qa(qa_records, section, "전체 행 수", spec["total"], len(selected))
    add_qa(qa_records, section, "연번 중복", 0, serial_duplicates)
    add_qa(qa_records, section, "완전 동일 행 중복", 0, exact_duplicates)
    add_qa(qa_records, section, "명칭·분류·지역·주소 중복", 0, identity_duplicates)
    add_qa(
        qa_records,
        section,
        "기업(관)명만 중복된 후속 행",
        name_duplicates,
        name_duplicates,
        note="동명이거나 소재지가 다른 공식 행이며 분자에서 임의 제거하지 않는다.",
    )
    add_qa(qa_records, section, "빈 주소", 0, empty_address)
    add_qa(qa_records, section, "시도 미분류", 0, unclassified)
    add_qa(qa_records, section, "주소 미분류", 0, address_unclassified)
    mismatch_note = ""
    if year == 2021:
        mismatch_note = (
            "공식 원본 연번 2400은 시도=충남, 주소지=충청북도 청주시이다. "
            "집계용 전용 시도 열(충남)을 적용하고 원자료 불일치를 보존한다."
        )
    add_qa(
        qa_records,
        section,
        "시도-주소 지역 불일치",
        spec["address_region_mismatches"],
        address_mismatch,
        note=mismatch_note,
    )
    add_qa(qa_records, section, "시도 수", 17, int(region_counts.index.nunique()))
    add_qa(qa_records, section, "시도별 합계", spec["total"], int(region_counts.sum()))
    add_qa(qa_records, section, "기업 분류별 합계", spec["category_totals"], categories)

    if any(record["판정"] == "FAIL" for record in qa_records if record["구분"] == section):
        raise ValueError(f"{year} 공식 명단 QA 실패")
    if set(region_counts.index) != set(REGION_ORDER):
        raise ValueError(f"{year} 지역 집합 불일치: {sorted(region_counts.index)}")
    return selected, region_counts.astype(int)


def load_denominators(
    raw_dir: Path, years: tuple[int, ...], qa_records: list[dict[str, object]]
) -> pd.DataFrame:
    path = raw_dir / DENOMINATOR_SOURCE["file_name"]
    payload = read_verified_bytes(
        path, DENOMINATOR_SOURCE["file_size"], DENOMINATOR_SOURCE["sha256"]
    )
    frame = pd.read_excel(io.BytesIO(payload), skiprows=[1])
    required = {"시도별(17개)", "산업분류별", "규모별", *years}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"분모 필수 열 누락: {sorted(missing, key=str)}")

    selected = frame.loc[
        frame["산업분류별"].eq("전체") & frame["규모별"].eq("전규모"),
        ["시도별(17개)", *years],
    ].copy()
    selected["지역"] = selected["시도별(17개)"].map(normalize_region)
    section = "분모"
    add_qa(qa_records, section, "파일 크기 bytes", DENOMINATOR_SOURCE["file_size"], len(payload))
    add_qa(
        qa_records,
        section,
        "SHA-256",
        DENOMINATOR_SOURCE["sha256"],
        file_sha256(payload),
    )
    add_qa(qa_records, section, "전체·17개 시도 행 수", 18, len(selected))
    add_qa(qa_records, section, "지역 미분류", 0, int(selected["지역"].isna().sum()))
    add_qa(qa_records, section, "지역 중복", 0, int(selected["지역"].duplicated().sum()))

    national = selected.loc[selected["지역"].eq("전국")].set_index("지역")
    provincial = selected.loc[selected["지역"].isin(REGION_ORDER)].set_index("지역")
    add_qa(qa_records, section, "시도 수", 17, len(provincial))
    for year in years:
        provincial[year] = pd.to_numeric(provincial[year], errors="raise").astype(int)
        expected = int(national.loc["전국", year])
        actual = int(provincial[year].sum())
        add_qa(qa_records, section, f"{year} 시도 합=전국", expected, actual)
        add_qa(
            qa_records,
            section,
            f"{year} 양의 사업체 수",
            17,
            int(provincial[year].gt(0).sum()),
        )
    if any(record["판정"] == "FAIL" for record in qa_records if record["구분"] == section):
        raise ValueError("분모 QA 실패")
    return provincial[list(years)].copy()


def load_reference_counts(raw_dir: Path, year: int) -> pd.Series:
    spec = REFERENCE_ROSTERS[year]
    payload = read_verified_bytes(raw_dir / spec["file_name"], spec["file_size"], spec["sha256"])
    frame = pd.read_excel(io.BytesIO(payload))
    if spec["region_column"] not in frame.columns:
        raise ValueError(f"{year} 교차검증 명단에 지역 열이 없습니다.")
    normalized = frame[spec["region_column"]].map(normalize_region)
    if normalized.isna().any():
        raise ValueError(f"{year} 교차검증 명단에 미분류 지역이 있습니다.")
    counts = normalized.value_counts().sort_index().astype(int)
    if len(frame) != spec["total"] or int(counts.sum()) != spec["total"]:
        raise ValueError(f"{year} 교차검증 명단 합계가 다릅니다.")
    if set(counts.index) != set(REGION_ORDER):
        raise ValueError(f"{year} 교차검증 지역 집합이 다릅니다.")
    return counts


def build_candidate_table(
    numerator_counts: dict[int, pd.Series], denominators: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in TARGET_YEARS:
        spec = OFFICIAL_ROSTERS[year]
        for region in REGION_ORDER:
            numerator = int(numerator_counts[year].loc[region])
            denominator = int(denominators.loc[region, year])
            rows.append(
                {
                    "지역": region,
                    "지표_id": INDICATOR_ID,
                    "연도": year,
                    "공식_분자": numerator,
                    "사업체수_분모": denominator,
                    "계산_비율": numerator / denominator * 100,
                    "분자_출처": spec["file_name"],
                    "분모_출처": (
                        f"{DENOMINATOR_SOURCE['file_name']}::열={year};산업분류별=전체;규모별=전규모"
                    ),
                    "QA_상태": "PASS",
                }
            )
    return pd.DataFrame(rows)


def validate_candidates(
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> None:
    keys = ["지역", "지표_id", "연도"]
    section = "34개 후보"
    add_qa(qa_records, section, "후보 행 수", 34, len(candidates))
    add_qa(qa_records, section, "후보 중복 키", 0, int(candidates.duplicated(keys).sum()))
    add_qa(qa_records, section, "후보 지역 수", 17, int(candidates["지역"].nunique()))
    add_qa(
        qa_records,
        section,
        "후보 연도",
        list(TARGET_YEARS),
        sorted(candidates["연도"].unique().tolist()),
    )
    add_qa(
        qa_records,
        section,
        "후보 지표",
        [INDICATOR_ID],
        sorted(candidates["지표_id"].unique().tolist()),
    )
    formula = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    add_qa(
        qa_records,
        section,
        "비율 산식 일치",
        34,
        int(np.isclose(candidates["계산_비율"], formula, rtol=0, atol=1e-15).sum()),
    )

    panel_target = panel.loc[
        panel["지표_id"].eq(INDICATOR_ID)
        & panel["지역"].isin(REGION_ORDER)
        & panel["연도"].isin(TARGET_YEARS),
        [*keys, "측정값"],
    ]
    merged = candidates[keys].merge(panel_target, on=keys, how="outer", indicator=True)
    add_qa(qa_records, section, "패널 키 1:1 대응", 34, int(merged["_merge"].eq("both").sum()))
    add_qa(qa_records, section, "패널 누락 키", 0, int(merged["_merge"].eq("left_only").sum()))
    add_qa(qa_records, section, "후보 누락 키", 0, int(merged["_merge"].eq("right_only").sum()))
    add_qa(qa_records, section, "대응 패널값 결측", 34, int(merged["측정값"].isna().sum()))

    unresolved = mapping.loc[
        mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["지역"].isin(REGION_ORDER)
        & mapping["연도"].isin(UNRESOLVED_YEARS)
    ].copy()
    add_qa(qa_records, "미확보 51개", "행 수", 51, len(unresolved))
    add_qa(
        qa_records,
        "미확보 51개",
        "pending_review 유지",
        51,
        int(unresolved["imputation_policy"].eq("pending_review").sum()),
    )
    add_qa(
        qa_records,
        "미확보 51개",
        "block_imputation=True 유지",
        51,
        int(unresolved["block_imputation"].sum()),
    )
    overlap = candidates[keys].merge(unresolved[keys], on=keys, how="inner")
    add_qa(qa_records, "미확보 51개", "후보와 중복", 0, len(overlap))
    add_qa(
        qa_records,
        "미확보 51개",
        "후보에 미확보 연도 포함",
        0,
        int(candidates["연도"].isin(UNRESOLVED_YEARS).sum()),
    )


def validate_reference_years(
    raw_dir: Path,
    panel: pd.DataFrame,
    denominators: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> dict[int, int]:
    matches: dict[int, int] = {}
    for year in REFERENCE_ROSTERS:
        counts = load_reference_counts(raw_dir, year)
        expected_rates = pd.Series(
            {
                region: counts.loc[region] / denominators.loc[region, year] * 100
                for region in REGION_ORDER
            }
        )
        observed = panel.loc[
            panel["지표_id"].eq(INDICATOR_ID)
            & panel["지역"].isin(REGION_ORDER)
            & panel["연도"].eq(year),
            ["지역", "측정값"],
        ].set_index("지역")["측정값"]
        aligned = observed.reindex(REGION_ORDER)
        match_count = int(np.isclose(expected_rates, aligned, rtol=0, atol=1e-12).sum())
        matches[year] = match_count
        add_qa(qa_records, "교차검증", f"{year} 공식 재집계-패널 일치", 17, match_count)
    return matches


def build_source_metadata(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for year, spec in OFFICIAL_ROSTERS.items():
        path = raw_dir / spec["file_name"]
        payload = read_verified_bytes(path, spec["file_size"], spec["sha256"])
        rows.append(
            {
                "연도": year,
                "기준일": spec["reference_date"],
                "제공기관": spec["provider"],
                "공식_게시물_URL": spec["post_url"],
                "다운로드_URL": spec["download_url"],
                "다운로드일시_KST": spec["downloaded_at"],
                "파일명": spec["file_name"],
                "파일크기_bytes": len(payload),
                "SHA-256": file_sha256(payload),
                "로컬_원본경로": path.relative_to(REPO_ROOT).as_posix(),
                "Git_추적": False,
                "원본_검증상태": "PASS",
            }
        )
    return pd.DataFrame(rows)


def render_report(
    metadata: pd.DataFrame,
    numerator_counts: dict[int, pd.Series],
    candidates: pd.DataFrame,
    reference_matches: dict[int, int],
    qa: pd.DataFrame,
) -> str:
    lines = [
        "# 가족친화 인증기업 비율 2020·2021 직접 복원 후보 QA",
        "",
        "## 범위",
        "",
        "- 공식 2020·2021년 연말 유효 인증기업·기관 명단을 고정하고 34개 후보값만 산출했다.",
        "- 산식: `(연도 말 유효 인증기업·기관 수 ÷ 같은 연도 전산업·전규모 사업체 수) × 100`",
        "- 정책 YAML, 기준 패널, 처리 완료 패널은 수정하지 않았고 `structural_missing.py`도 실행하지 않았다.",
        "- 후보 CSV는 기준 패널에 병합하지 않는다.",
        "",
        "## 공식 원본",
        "",
        "| 연도 | 기준일 | 파일명 | bytes | SHA-256 | 게시물 | 다운로드 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for _, row in metadata.iterrows():
        lines.append(
            f"| {row['연도']} | {row['기준일']} | `{row['파일명']}` | "
            f"{row['파일크기_bytes']:,} | `{row['SHA-256']}` | "
            f"[원문]({row['공식_게시물_URL']}) | [XLSX]({row['다운로드_URL']}) |"
        )
    lines.extend(
        [
            "",
            "원본은 저장소 규칙에 따라 `data/raw/`에 원형 그대로 보관하며 Git으로 추적하지 않는다. "
            "다운로드 메타데이터와 승인 해시는 `reports/20260804_가족친화_공식원본_메타데이터.csv`에 기록한다.",
            "",
            "## 분자 집계",
            "",
            "| 지역 | 2020 | 2021 |",
            "|---|---:|---:|",
        ]
    )
    for region in REGION_ORDER:
        lines.append(
            f"| {region} | {int(numerator_counts[2020].loc[region]):,} | "
            f"{int(numerator_counts[2021].loc[region]):,} |"
        )
    lines.extend(
        [
            f"| 전국 합계 | {int(numerator_counts[2020].sum()):,} | "
            f"{int(numerator_counts[2021].sum()):,} |",
            "",
            "대기업·중소기업·공공기관을 모두 포함했다. 기업(관)명만 같은 후속 행은 "
            "2020년 13건, 2021년 19건이나 명칭·분류·지역·주소가 모두 같은 중복은 0건이다. "
            "공식 행을 기업명만으로 임의 제거하지 않았다.",
            "2021년 공식 원본 연번 2400은 `시도=충남`, `주소지=충청북도 청주시`로 서로 다르다. "
            "지역 집계는 명단의 전용 `시도` 열을 적용했으며 이 1건은 QA에 원자료 불일치로 남겼다.",
            "",
            "## 분모와 후보",
            "",
            f"- 분모 파일: `{DENOMINATOR_SOURCE['file_name']}`",
            f"- 승인 SHA-256: `{DENOMINATOR_SOURCE['sha256']}`",
            f"- 선택: {DENOMINATOR_SOURCE['selection']}",
            "- 지역 정규화: KOSIS의 전체 시도명과 명단의 시도·주소 앞부분을 패널의 17개 축약명으로 변환",
            f"- 후보: `{CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()}` ({len(candidates)}행)",
            "",
            "## 교차검증",
            "",
            "| 검증 | 결과 |",
            "|---|---:|",
            f"| 2018 공식 명단 재집계와 기존 패널 | {reference_matches[2018]}/17 |",
            f"| 2022 공식 명단 재집계와 기존 패널 | {reference_matches[2022]}/17 |",
            f"| 2023 공식 명단 재집계와 기존 패널 | {reference_matches[2023]}/17 |",
            f"| 2020 공식 전국 합계 | {int(numerator_counts[2020].sum()):,}/4,340 |",
            f"| 2021 공식 전국 합계 | {int(numerator_counts[2021].sum()):,}/4,918 |",
            "| 후보-패널 결측 키 1:1 | 34/34 |",
            "| 중복·누락·무관 키 | 0/0/0 |",
            "",
            "## 미확보 51개",
            "",
            "2016·2017·2019년 17개 시도, 총 51개는 후보를 생성하지 않았다. 기존 매핑의 "
            "`pending_review`, `block_imputation=True`를 그대로 유지한다.",
            "",
            "공식 자료 요청 대상은 각 연도 12월 31일 현재 유효한 전체 기업·기관 명단이다. "
            "필수 필드는 기업·기관 식별자와 명칭, 기업 분류, 시도·소재지, 최초 인증일, "
            "신규·연장·재인증 구분, 각 유효기간 시작·종료일, 취소·철회일, 소재지 변경 이력이다.",
            "",
            "## 2024 불일치 추적 후보",
            "",
            "- `data/raw/2. 구조환경지수 원데이터/3. 제주여성가족연구원 지표별 측정값(raw)/"
            "4-1.2. 가족친화 인증기업 비율.xlsx`: `연도&시도 가족친화 인증기업 집계` "
            "시트의 2024 대구 277·광주 160 선언값과 `가족친화인증기업 비율 계산` 시트",
            "- `data/raw/2. 구조환경지수 원데이터/3. 제주여성가족연구원 지표별 측정값(raw)/"
            "4-1.2. 가족친화인증기업 비율/가족친화인증기업현황_20241231.csv.xlsx`: "
            "공식 6,502행 명단, "
            "직접 집계 대구 218·광주 140",
            "- `notebooks/20260724_EDA_구조환경지표_21개_원자료기반_전수검증.ipynb`: "
            "계산 시트 로드 4397줄 부근, 직접 집계 4561~4597줄 부근",
            "- `notebooks/20260725_EDA_구조환경지표_21개_검증본_interim_생성.ipynb`: "
            "불일치 메모 400~403줄과 검증본 저장 469~472줄",
            "- `notebooks/20260804_EDA_구조환경지표_28개_보간전_통합패널.ipynb`: "
            "검증본 입력 SHA 고정 104줄, 지표 ID 매핑 160줄 부근",
            "- `reports/20260724_구조환경지표_21개_검증_진행상황.md`: 162~169줄",
            "",
            "이번 단계에서는 위 계보만 식별했으며 2024 패널 값은 수정하지 않았다.",
            "",
            "## QA 요약",
            "",
            f"- QA 항목: {len(qa)}개",
            f"- PASS: {int(qa['판정'].eq('PASS').sum())}개",
            f"- FAIL: {int(qa['판정'].eq('FAIL').sum())}개",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--metadata-output", type=Path, default=SOURCE_METADATA_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_hash_before = file_sha256(args.panel.read_bytes())
    mapping_hash_before = file_sha256(args.mapping.read_bytes())
    qa_records: list[dict[str, object]] = []

    panel = pd.read_csv(args.panel)
    mapping = pd.read_csv(args.mapping)
    numerator_counts: dict[int, pd.Series] = {}
    for year in TARGET_YEARS:
        _, numerator_counts[year] = load_official_roster(args.raw_dir, year, qa_records)
    denominators = load_denominators(args.raw_dir, (2018, 2020, 2021, 2022, 2023), qa_records)
    candidates = build_candidate_table(numerator_counts, denominators)
    validate_candidates(candidates, panel, mapping, qa_records)
    reference_matches = validate_reference_years(args.raw_dir, panel, denominators, qa_records)
    metadata = build_source_metadata(args.raw_dir)
    qa = pd.DataFrame(qa_records)
    if not qa["판정"].eq("PASS").all():
        failed = qa.loc[qa["판정"].eq("FAIL")]
        raise ValueError(f"QA 실패:\n{failed.to_string(index=False)}")

    add_qa(
        qa_records,
        "입력 불변",
        "기준 패널 SHA-256 유지",
        panel_hash_before,
        file_sha256(args.panel.read_bytes()),
    )
    add_qa(
        qa_records,
        "입력 불변",
        "정책 매핑 SHA-256 유지",
        mapping_hash_before,
        file_sha256(args.mapping.read_bytes()),
    )
    qa = pd.DataFrame(qa_records)
    if not qa["판정"].eq("PASS").all():
        raise ValueError("입력 불변 QA 실패")

    for output in (
        args.candidate_output,
        args.metadata_output,
        args.qa_output,
        args.report_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.candidate_output, index=False, encoding="utf-8-sig", float_format="%.15f"
    )
    metadata.to_csv(args.metadata_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    args.report_output.write_text(
        render_report(metadata, numerator_counts, candidates, reference_matches, qa),
        encoding="utf-8",
    )
    print(f"후보 저장: {args.candidate_output} ({len(candidates)}행)")
    print(f"원본 메타데이터 저장: {args.metadata_output} ({len(metadata)}행)")
    print(f"QA 저장: {args.qa_output} ({len(qa)}개 PASS)")
    print(f"보고서 저장: {args.report_output}")


if __name__ == "__main__":
    main()
