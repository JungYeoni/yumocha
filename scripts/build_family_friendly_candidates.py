"""Build verified family-friendly certification observations and QA artifacts.

The raw workbooks are immutable inputs. This script freezes the official 2020, 2021, and 2024
rosters, reconstructs the province-level rates, and emits a reviewed observation artifact for
the canonical panel notebooks. It never writes to a panel or runs missing-value imputation.
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
CONFIRMED_PATH = REPO_ROOT / "reports" / "20260804_가족친화_공식관측_반영값.csv"
SOURCE_METADATA_PATH = REPO_ROOT / "reports" / "20260804_가족친화_공식원본_메타데이터.csv"
QA_PATH = REPO_ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_QA.csv"
FALSE_MATCH_PATH = REPO_ROOT / "reports" / "20260804_가족친화_2024_부분일치_중복행_QA.csv"
REGION_COUNTS_2024_PATH = REPO_ROOT / "reports" / "20260804_가족친화_2024_공식시도별_집계_QA.csv"
REPORT_PATH = REPO_ROOT / "reports" / "methodology" / "20260804_가족친화_2020_2021_직접복원_QA.md"
LEGACY_WORKBOOK_PATH = RAW_DIR.parent / "4-1.2. 가족친화 인증기업 비율.xlsx"

INDICATOR_ID = "family_friendly_certification_rate"
TARGET_YEARS = (2020, 2021)
OFFICIAL_YEARS = (2020, 2021, 2024)
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
    2024: {
        "provider": "여성가족부 가족친화인증기업 현황",
        "post_url": "",
        "download_url": "",
        "downloaded_at": "",
        "reference_date": "2024-12-31",
        "file_name": "가족친화인증기업현황_20241231.csv.xlsx",
        "file_size": 242_330,
        "sha256": "6026921d157c499156d6ddbce820833c6f96a6709b3814e940dae5a47f3d4696",
        "sheet_name": "가족친화인증기업현황_20241231",
        "header": 0,
        "columns": {
            "first_year": "최초인증년도",
            "name": "기업(관)명",
            "category": "기업(관)분류",
            "region": "지역",
            "address": "소재지",
        },
        "total": 6_502,
        "category_totals": {"대기업": 784, "중소기업": 4_552, "공공기관": 1_166},
        "address_region_mismatches": 0,
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

DENOMINATOR_2024_SOURCE = {
    "file_name": (
        "2024_시도별__산업별__규모별__사업체수_및_종사자수_성별__20260715194241.csv.xlsx"
    ),
    "file_size": 11_130,
    "sha256": "7bdc86d26b7a042f89b20eeb18ca92d1af51da66e08ed963702ff1bf5cea8586",
    "table": "사업체노동실태현황: 시도별, 산업별, 규모별, 사업체수 및 종사자수(성별)",
    "selection": "산업분류별=전체; 규모별=전규모; 2024 열=사업체수 (개)",
}

LEGACY_WORKBOOK_SOURCE = {
    "file_size": 1_282_581,
    "sha256": "ffbd6e0dbc0a43d6b9de62d47b710e0084a3e70a31aff25928009d1ce4834bce",
    "aggregate_sheet": "연도&시도 가족친화 인증기업 집계",
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
    if "serial" in selected:
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
    serial_duplicates = int(selected["serial"].duplicated().sum()) if "serial" in selected else 0
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
    add_qa(
        qa_records,
        section,
        "연번 중복",
        0,
        serial_duplicates,
        note="2024 공식 파일에는 연번 열이 없으며 나머지 중복 키를 별도로 검증한다."
        if year == 2024
        else "",
    )
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


def load_denominator_source(
    raw_dir: Path,
    source: dict[str, object],
    years: tuple[int, ...],
    qa_records: list[dict[str, object]],
    section: str,
) -> pd.DataFrame:
    path = raw_dir / str(source["file_name"])
    payload = read_verified_bytes(path, int(source["file_size"]), str(source["sha256"]))
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
    add_qa(qa_records, section, "파일 크기 bytes", source["file_size"], len(payload))
    add_qa(qa_records, section, "SHA-256", source["sha256"], file_sha256(payload))
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
    return selected.set_index("지역")[list(years)].apply(pd.to_numeric, errors="raise").astype(int)


def load_denominators(
    raw_dir: Path, years: tuple[int, ...], qa_records: list[dict[str, object]]
) -> pd.DataFrame:
    requested = set(years)
    frames = []
    historical_years = tuple(year for year in years if year <= 2023)
    if historical_years:
        frames.append(
            load_denominator_source(
                raw_dir,
                DENOMINATOR_SOURCE,
                historical_years,
                qa_records,
                "분모 2017-2023",
            )
        )
    if 2024 in requested:
        frames.append(
            load_denominator_source(
                raw_dir,
                DENOMINATOR_2024_SOURCE,
                (2024,),
                qa_records,
                "분모 2024",
            )
        )
    denominators = pd.concat(frames, axis="columns").reindex(columns=list(years))
    if denominators.isna().any().any():
        raise ValueError(f"분모 연도 결합 실패: {years}")
    return denominators


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


def build_confirmed_table(
    candidates: pd.DataFrame,
    numerator_counts: dict[int, pd.Series],
    denominators: pd.DataFrame,
) -> pd.DataFrame:
    """Build the exact 37 direct-observation/correction rows consumed upstream."""
    confirmed = candidates.rename(columns={"계산_비율": "측정값"}).copy()
    confirmed["반영유형"] = "공식 원자료 직접 복원"
    confirmed["관측상태"] = "관측"

    rows = []
    for region in ("전국", "대구", "광주"):
        numerator = 6_502 if region == "전국" else int(numerator_counts[2024].loc[region])
        denominator = int(denominators.loc[region, 2024])
        rows.append(
            {
                "지역": region,
                "지표_id": INDICATOR_ID,
                "연도": 2024,
                "공식_분자": numerator,
                "사업체수_분모": denominator,
                "측정값": numerator / denominator * 100,
                "분자_출처": OFFICIAL_ROSTERS[2024]["file_name"],
                "분모_출처": (
                    f"{DENOMINATOR_2024_SOURCE['file_name']}::열=2024;산업분류별=전체;규모별=전규모"
                ),
                "QA_상태": "PASS",
                "반영유형": "공식 집계 오류 정정",
                "관측상태": "관측",
            }
        )
    confirmed = pd.concat([confirmed, pd.DataFrame(rows)], ignore_index=True)
    return confirmed[
        [
            "지역",
            "지표_id",
            "연도",
            "공식_분자",
            "사업체수_분모",
            "측정값",
            "분자_출처",
            "분모_출처",
            "반영유형",
            "관측상태",
            "QA_상태",
        ]
    ]


def validate_confirmed_table(
    confirmed: pd.DataFrame,
    numerator_counts: dict[int, pd.Series],
    denominators: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> None:
    keys = ["지역", "지표_id", "연도"]
    section = "공식 관측 반영 37개"
    add_qa(qa_records, section, "행 수", 37, len(confirmed))
    add_qa(qa_records, section, "중복 키", 0, int(confirmed.duplicated(keys).sum()))
    add_qa(
        qa_records,
        section,
        "2020·2021 직접 관측",
        34,
        int(confirmed["반영유형"].eq("공식 원자료 직접 복원").sum()),
    )
    add_qa(
        qa_records,
        section,
        "2024 정정 키",
        ["전국", "대구", "광주"],
        confirmed.loc[confirmed["연도"].eq(2024), "지역"].tolist(),
    )
    formula = confirmed["공식_분자"] / confirmed["사업체수_분모"] * 100
    add_qa(
        qa_records,
        section,
        "전체 산식 일치",
        37,
        int(np.isclose(confirmed["측정값"], formula, rtol=0, atol=1e-15).sum()),
    )
    expected_2024 = {"전국": 6_502, "대구": 218, "광주": 140}
    actual_2024 = confirmed.loc[confirmed["연도"].eq(2024)].set_index("지역")["공식_분자"].to_dict()
    add_qa(qa_records, section, "2024 확정 분자", expected_2024, actual_2024)
    add_qa(
        qa_records,
        section,
        "2024 17개 시도 합계",
        6_502,
        int(numerator_counts[2024].reindex(REGION_ORDER).sum()),
    )
    expected_rates = {
        region: numerator / int(denominators.loc[region, 2024]) * 100
        for region, numerator in expected_2024.items()
    }
    actual_rates = confirmed.loc[confirmed["연도"].eq(2024)].set_index("지역")["측정값"]
    add_qa(
        qa_records,
        section,
        "2024 확정 비율",
        3,
        int(
            sum(
                np.isclose(actual_rates.loc[region], value, rtol=0, atol=1e-15)
                for region, value in expected_rates.items()
            )
        ),
    )


def build_2024_region_counts(
    numerator_counts: dict[int, pd.Series], denominators: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        numerator = int(numerator_counts[2024].loc[region])
        denominator = int(denominators.loc[region, 2024])
        rows.append(
            {
                "지역": region,
                "연도": 2024,
                "공식_분자": numerator,
                "사업체수_분모": denominator,
                "계산_비율": numerator / denominator * 100,
                "패널_값_변경": region in {"대구", "광주"},
                "분자_출처": OFFICIAL_ROSTERS[2024]["file_name"],
                "분모_출처": DENOMINATOR_2024_SOURCE["file_name"],
                "QA_상태": "PASS",
            }
        )
    return pd.DataFrame(rows)


def validate_2024_panel(
    panel: pd.DataFrame,
    region_counts: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> None:
    observed = panel.loc[
        panel["지표_id"].eq(INDICATOR_ID)
        & panel["지역"].isin(REGION_ORDER)
        & panel["연도"].eq(2024),
        ["지역", "측정값"],
    ]
    comparison = region_counts.merge(observed, on="지역", validate="one_to_one")
    matched = np.isclose(comparison["계산_비율"], comparison["측정값"], rtol=0, atol=1e-12)
    add_qa(qa_records, "2024 패널 반영", "17개 시도 공식 비율 일치", 17, int(matched.sum()))
    add_qa(
        qa_records,
        "2024 패널 반영",
        "나머지 15개 시도 기존 공식값 유지",
        15,
        int((matched & ~comparison["지역"].isin(["대구", "광주"])).sum()),
    )


def apply_confirmed_observations(
    frame: pd.DataFrame,
    confirmed: pd.DataFrame,
    *,
    indicator_label: str = "가족친화인증기업 비율",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply reviewed official observations to the 21-indicator wide table only."""
    required = {"지역", "세부지표", *range(2016, 2025)}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"21개 검증본 필수 열 누락: {sorted(missing_columns, key=str)}")
    keys = ["지역", "지표_id", "연도"]
    if len(confirmed) != 37 or confirmed.duplicated(keys).any():
        raise ValueError("공식 관측 반영표는 중복 없는 37개 키여야 합니다.")
    if not confirmed["지표_id"].eq(INDICATOR_ID).all():
        raise ValueError("공식 관측 반영표에 다른 지표가 포함됐습니다.")
    if not confirmed["QA_상태"].eq("PASS").all():
        raise ValueError("QA를 통과하지 않은 공식 관측값은 반영할 수 없습니다.")

    result = frame.copy()
    audit_rows = []
    for row in confirmed.to_dict("records"):
        mask = result["세부지표"].eq(indicator_label) & result["지역"].eq(row["지역"])
        if int(mask.sum()) != 1:
            raise ValueError(f"21개 검증본 반영 키가 1개가 아닙니다: {row['지역']}, {row['연도']}")
        before = result.loc[mask, row["연도"]].iloc[0]
        if row["연도"] in TARGET_YEARS and row["지역"] in REGION_ORDER:
            acceptable = pd.isna(before) or np.isclose(
                float(before), float(row["측정값"]), rtol=0, atol=1e-15
            )
            if not acceptable:
                raise ValueError(
                    f"2020·2021 직접 복원 대상의 기존 값이 예상과 다릅니다: {row['지역']}"
                )
        result.loc[mask, row["연도"]] = float(row["측정값"])
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


def build_2024_false_match_qa(
    official: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> pd.DataFrame:
    """Reproduce the legacy 6,581 assignments and isolate all 79 false region matches."""
    payload = read_verified_bytes(
        LEGACY_WORKBOOK_PATH,
        LEGACY_WORKBOOK_SOURCE["file_size"],
        LEGACY_WORKBOOK_SOURCE["sha256"],
    )
    aggregate = pd.read_excel(
        io.BytesIO(payload), sheet_name=LEGACY_WORKBOOK_SOURCE["aggregate_sheet"]
    )
    legacy_counts = aggregate.loc[:17, ["지역.1", 2024]].set_index("지역.1")[2024]

    expanded_rows = []
    official_only = 0
    for source_row, row in official.reset_index(drop=True).iterrows():
        address = str(row["address"])
        permissive_regions = {
            normalized for alias, normalized in REGION_ALIASES.items() if alias in address
        }
        if row["normalized_region"] not in permissive_regions:
            official_only += 1
        for assigned_region in sorted(permissive_regions):
            if assigned_region == row["normalized_region"]:
                continue
            expanded_rows.append(
                {
                    "공식원본행": source_row + 2,
                    "기업(관)명": row["name"],
                    "기업(관)분류": row["category"],
                    "공식_지역": row["normalized_region"],
                    "부분일치_오배정_지역": assigned_region,
                    "소재지": row["address"],
                    "오류원인": "주소 전체 문자열에서 시도명 부분일치",
                    "정상화_검증": normalize_region(row["address"]),
                }
            )
    false_matches = pd.DataFrame(expanded_rows)
    permissive_total = len(official) + len(false_matches)
    section = "2024 주소 부분일치 오류 재현"
    add_qa(qa_records, section, "기존 원본 파일 크기 bytes", 1_282_581, len(payload))
    add_qa(
        qa_records,
        section,
        "기존 원본 SHA-256",
        LEGACY_WORKBOOK_SOURCE["sha256"],
        file_sha256(payload),
    )
    add_qa(qa_records, section, "공식 명단 행", 6_502, len(official))
    add_qa(qa_records, section, "기존 집계 전국", 6_581, int(legacy_counts.loc["전국"]))
    add_qa(qa_records, section, "부분일치 재현 행", 6_581, permissive_total)
    add_qa(qa_records, section, "기존 집계에만 있는 행", 79, len(false_matches))
    add_qa(qa_records, section, "공식 명단에만 있는 행", 0, official_only)
    error_counts = false_matches["부분일치_오배정_지역"].value_counts().to_dict()
    add_qa(qa_records, section, "오배정 지역별 행", {"대구": 59, "광주": 20}, error_counts)
    add_qa(qa_records, section, "기존 집계 대구", 277, int(legacy_counts.loc["대구"]))
    add_qa(qa_records, section, "기존 집계 광주", 160, int(legacy_counts.loc["광주"]))
    cross_region_duplicates = false_matches["공식_지역"].ne(false_matches["부분일치_오배정_지역"])
    add_qa(
        qa_records,
        section,
        "79행 모두 지역 중복",
        79,
        int(cross_region_duplicates.sum()),
    )
    prefix_examples = {
        "부산광역시 해운대구": int(
            false_matches["소재지"].str.startswith("부산광역시 해운대구").sum()
        ),
        "경기도 광주시": int(false_matches["소재지"].str.startswith("경기도 광주시").sum()),
    }
    add_qa(
        qa_records,
        section,
        "오류 주소 접두어",
        {"부산광역시 해운대구": 59, "경기도 광주시": 20},
        prefix_examples,
    )
    return false_matches


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
    candidate_values = candidates[keys + ["계산_비율"]].merge(
        panel_target, on=keys, how="left", validate="one_to_one"
    )
    missing_or_confirmed = candidate_values["측정값"].isna() | np.isclose(
        candidate_values["측정값"],
        candidate_values["계산_비율"],
        rtol=0,
        atol=1e-14,
        equal_nan=False,
    )
    add_qa(
        qa_records,
        section,
        "패널값이 반영 전 결측 또는 확정값",
        34,
        int(missing_or_confirmed.sum()),
    )
    add_qa(
        qa_records,
        section,
        "현재 패널 직접 관측 표시",
        int(candidate_values["측정값"].notna().sum()),
        int(candidate_values["측정값"].notna().sum()),
        note="재생성 전에는 0, 공식 관측 반영 후에는 34여야 한다.",
    )

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
    confirmed: pd.DataFrame,
    false_matches: pd.DataFrame,
    reference_matches: dict[int, int],
    qa: pd.DataFrame,
) -> str:
    lines = [
        "# 가족친화 인증기업 비율 공식 관측 반영 QA",
        "",
        "## 범위",
        "",
        "- 공식 2020·2021년 연말 명단으로 17개 시도×2개 연도 34개 직접 관측값을 복원했다.",
        "- 공식 2024년 말 명단으로 전국·대구·광주 3개 기존 관측값을 정정했다.",
        "- 산식: `(연도 말 유효 인증기업·기관 수 ÷ 같은 연도 전산업·전규모 사업체 수) × 100`",
        "- 대기업·중소기업·공공기관을 모두 포함하며 `structural_missing.py`는 실행하지 않는다.",
        "- 2016·2017·2019년 51개 시도 키에는 어떤 후보값이나 대체값도 만들지 않는다.",
        "",
        "## 공식 원본",
        "",
        "| 연도 | 기준일 | 파일명 | bytes | SHA-256 | 게시물 | 다운로드 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for _, row in metadata.iterrows():
        post = f"[원문]({row['공식_게시물_URL']})" if row["공식_게시물_URL"] else "-"
        download = f"[XLSX]({row['다운로드_URL']})" if row["다운로드_URL"] else "-"
        lines.append(
            f"| {row['연도']} | {row['기준일']} | `{row['파일명']}` | "
            f"{row['파일크기_bytes']:,} | `{row['SHA-256']}` | "
            f"{post} | {download} |"
        )
    lines.extend(
        [
            "",
            "원본은 저장소 규칙에 따라 `data/raw/`에 원형 그대로 보관하며 Git으로 추적하지 않는다. "
            "다운로드 메타데이터와 승인 해시는 `reports/20260804_가족친화_공식원본_메타데이터.csv`에 기록한다.",
            "",
            "## 분자 집계",
            "",
            "| 지역 | 2020 | 2021 | 2024 |",
            "|---|---:|---:|---:|",
        ]
    )
    for region in REGION_ORDER:
        lines.append(
            f"| {region} | {int(numerator_counts[2020].loc[region]):,} | "
            f"{int(numerator_counts[2021].loc[region]):,} | "
            f"{int(numerator_counts[2024].loc[region]):,} |"
        )
    lines.extend(
        [
            f"| 전국 합계 | {int(numerator_counts[2020].sum()):,} | "
            f"{int(numerator_counts[2021].sum()):,} | "
            f"{int(numerator_counts[2024].sum()):,} |",
            "",
            "기업(관)명만 같은 후속 행은 "
            "2020년 13건, 2021년 19건이나 명칭·분류·지역·주소가 모두 같은 중복은 0건이다. "
            "공식 행을 기업명만으로 임의 제거하지 않았다.",
            "2021년 공식 원본 연번 2400은 `시도=충남`, `주소지=충청북도 청주시`로 서로 다르다. "
            "지역 집계는 명단의 전용 `시도` 열을 적용했으며 이 1건은 QA에 원자료 불일치로 남겼다.",
            "",
            "2024년 유형 합계는 대기업 784, 중소기업 4,552, 공공기관 1,166으로 총 6,502다.",
            "",
            "## 분모와 반영값",
            "",
            f"- 2017–2023 분모: `{DENOMINATOR_SOURCE['file_name']}`, SHA-256 `{DENOMINATOR_SOURCE['sha256']}`",
            f"- 2024 분모: `{DENOMINATOR_2024_SOURCE['file_name']}`, SHA-256 `{DENOMINATOR_2024_SOURCE['sha256']}`",
            "- 지역 정규화: 공식 `시도`/`지역` 열을 우선하며 주소는 `startswith` 검증에만 사용",
            f"- 후보: `{CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()}` ({len(candidates)}행)",
            f"- canonical 반영값: `{CONFIRMED_PATH.relative_to(REPO_ROOT).as_posix()}` ({len(confirmed)}행)",
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
            f"| 2024 공식 전국 합계 | {int(numerator_counts[2024].sum()):,}/6,502 |",
            "| 2020·2021 직접 관측 키 | 34 |",
            "| 2024 정정 키 | 전국·대구·광주 3 |",
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
            "## 2024 불일치 원인",
            "",
            f"- 공식 파일: `{OFFICIAL_ROSTERS[2024]['file_name']}`, 242,330 bytes, SHA-256 `{OFFICIAL_ROSTERS[2024]['sha256']}`",
            "- 공식 지역 열 집계는 6,502개이고 기존 `연도&시도 가족친화 인증기업 집계` 시트는 6,581개다.",
            "- 주소 전체 문자열 부분일치를 재현하면 부산광역시 해운대구 59행이 대구에, 경기도 광주시 20행이 광주에 추가 배정되어 정확히 6,581개가 된다.",
            f"- 기존 집계에만 있는 {len(false_matches)}행은 모두 공식 지역에도 이미 포함된 지역 중복이고, 공식 명단에만 있는 행은 0건이다.",
            f"- 17개 시도 공식 집계: `{REGION_COUNTS_2024_PATH.relative_to(REPO_ROOT).as_posix()}`",
            f"- 행 단위 근거: `{FALSE_MATCH_PATH.relative_to(REPO_ROOT).as_posix()}`",
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
    parser.add_argument("--confirmed-output", type=Path, default=CONFIRMED_PATH)
    parser.add_argument("--metadata-output", type=Path, default=SOURCE_METADATA_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument("--false-match-output", type=Path, default=FALSE_MATCH_PATH)
    parser.add_argument("--region-counts-2024-output", type=Path, default=REGION_COUNTS_2024_PATH)
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
    official_frames = {}
    for year in OFFICIAL_YEARS:
        official_frames[year], numerator_counts[year] = load_official_roster(
            args.raw_dir, year, qa_records
        )
    denominators = load_denominators(args.raw_dir, (2018, 2020, 2021, 2022, 2023, 2024), qa_records)
    candidates = build_candidate_table(numerator_counts, denominators)
    confirmed = build_confirmed_table(candidates, numerator_counts, denominators)
    region_counts_2024 = build_2024_region_counts(numerator_counts, denominators)
    validate_candidates(candidates, panel, mapping, qa_records)
    validate_confirmed_table(confirmed, numerator_counts, denominators, qa_records)
    validate_2024_panel(panel, region_counts_2024, qa_records)
    false_matches = build_2024_false_match_qa(official_frames[2024], qa_records)
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
        args.confirmed_output,
        args.metadata_output,
        args.qa_output,
        args.false_match_output,
        args.region_counts_2024_output,
        args.report_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.candidate_output, index=False, encoding="utf-8-sig", float_format="%.15f"
    )
    confirmed.to_csv(args.confirmed_output, index=False, encoding="utf-8-sig", float_format="%.16f")
    metadata.to_csv(args.metadata_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    false_matches.to_csv(args.false_match_output, index=False, encoding="utf-8-sig")
    region_counts_2024.to_csv(
        args.region_counts_2024_output,
        index=False,
        encoding="utf-8-sig",
        float_format="%.16f",
    )
    args.report_output.write_text(
        render_report(
            metadata,
            numerator_counts,
            candidates,
            confirmed,
            false_matches,
            reference_matches,
            qa,
        ),
        encoding="utf-8",
    )
    print(f"후보 저장: {args.candidate_output} ({len(candidates)}행)")
    print(f"공식 관측 반영값 저장: {args.confirmed_output} ({len(confirmed)}행)")
    print(f"원본 메타데이터 저장: {args.metadata_output} ({len(metadata)}행)")
    print(f"QA 저장: {args.qa_output} ({len(qa)}개 PASS)")
    print(f"2024 부분일치 중복행 QA: {args.false_match_output} ({len(false_matches)}행)")
    print(
        f"2024 공식 시도별 집계 QA: {args.region_counts_2024_output} ({len(region_counts_2024)}행)"
    )
    print(f"보고서 저장: {args.report_output}")


if __name__ == "__main__":
    main()
