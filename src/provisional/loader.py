"""Strict workbook loader for the provisional #81 budget panel.

The authoritative tab is ``정리본_자동``. ``Table 1`` is retained as the
raw-document cross-check, while ``검증_자동`` records the block-level column
mapping QA. These roles and the year-specific structural corrections mirror
the reviewed yearly pipelines that produced the #52 long files.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.features.pipeline_common import (
    SUBTOTAL_LABEL_PATTERN,
    UNIT_NOTATION_PATTERN,
    assign_labels,
    classify_row,
    clean_text,
    drop_exact_duplicate_rows,
    select_total_budget_rows,
    to_numeric_budget,
)


EXPECTED_YEARS = tuple(range(2016, 2025))
STANDARD_REGIONS = (
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
)
RAW_SHEET = "Table 1"
NORMALIZED_SHEET = "정리본_자동"
VALIDATION_SHEET = "검증_자동"
BUDGET_UNIT = "백만원"
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _extract_year(path: Path) -> int:
    years = {int(value) for value in YEAR_PATTERN.findall(path.name)}
    if len(years) != 1:
        raise ValueError(f"파일명에서 연도를 하나만 확인할 수 있어야 합니다: {path.name}")
    return years.pop()


def validate_input_files(
    paths: Sequence[str | Path],
    *,
    expected_years: Sequence[int] = EXPECTED_YEARS,
) -> list[tuple[int, Path]]:
    """Validate an explicit one-workbook-per-year input list."""

    expected = [int(year) for year in expected_years]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_years에 중복이 있습니다.")
    if len(paths) != len(expected):
        raise ValueError(f"입력 Excel 수 불일치: 기대={len(expected)}, 실제={len(paths)}")

    validated: list[tuple[int, Path]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"입력 Excel 파일이 없습니다: {path}")
        if path.suffix.lower() != ".xlsx":
            raise ValueError(f"입력은 .xlsx여야 합니다: {path}")
        validated.append((_extract_year(path), path))

    actual = [year for year, _ in validated]
    if len(actual) != len(set(actual)):
        raise ValueError(f"입력 Excel 연도가 중복됩니다: {actual}")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(f"입력 Excel 연도 불일치: 누락={missing}, 예상외={unexpected}")
    return sorted(validated)


def _required_normalized_columns(year: int) -> list[str]:
    return [
        "지역",
        "세부사업명",
        "사업분류재정구분",
        f"{year}년 예산",
        f"{year - 1}년 예산",
        "증감액",
        "비율",
        "주요내용",
        "원본표구간",
        "머리글행",
        "원본행",
        "행유형",
    ]


def _required_validation_columns(year: int) -> list[str]:
    return [
        "원본표구간",
        "머리글행",
        "블록끝행",
        "사업명열",
        "구분열",
        f"{year}년 예산열",
        f"{year - 1}년 예산열",
        "증감액열",
        "비율열",
        "내용열",
        "추출행수",
        "점검내용",
    ]


def _read_workbook(path: Path, year: int) -> tuple[pd.DataFrame, dict]:
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        required_sheets = {RAW_SHEET, NORMALIZED_SHEET, VALIDATION_SHEET}
        missing_sheets = sorted(required_sheets - set(workbook.sheet_names))
        if missing_sheets:
            raise ValueError(f"{path}: 필수 시트 누락={missing_sheets}")

        frame = pd.read_excel(workbook, sheet_name=NORMALIZED_SHEET, header=0)
        validation = pd.read_excel(workbook, sheet_name=VALIDATION_SHEET, header=0)

    expected_columns = _required_normalized_columns(year)
    # 실제 2021 정리본에는 과거 분류 흔적이 두 행에만 남은 선행
    # ``Unnamed: 0/1`` 열이 있다. 권위 스키마에 속하지 않는 이 두 열은 이름과
    # 위치가 모두 확인된 경우에만 제외한다. 다른 예상외 열은 아래에서 거부한다.
    legacy_prefix = ["Unnamed: 0", "Unnamed: 1"]
    if year == 2021 and list(frame.columns) == legacy_prefix + expected_columns:
        frame = frame[expected_columns].copy()
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{path}: {NORMALIZED_SHEET} 스키마 불일치; "
            f"기대={expected_columns}, 실제={list(frame.columns)}"
        )

    validation = validation.dropna(axis=0, how="all").dropna(axis=1, how="all")
    expected_validation = _required_validation_columns(year)
    if list(validation.columns) != expected_validation:
        raise ValueError(
            f"{path}: {VALIDATION_SHEET} 스키마 불일치; "
            f"기대={expected_validation}, 실제={list(validation.columns)}"
        )
    if validation.empty:
        raise ValueError(f"{path}: {VALIDATION_SHEET}에 QA 행이 없습니다.")
    statuses = validation["점검내용"].astype("string").str.strip()
    if statuses.isna().any() or not statuses.eq("정상").all():
        counts = statuses.fillna("<blank>").value_counts().to_dict()
        raise ValueError(f"{path}: {VALIDATION_SHEET} 비정상 상태={counts}")

    metadata = {
        "year": year,
        "authoritative_sheet": NORMALIZED_SHEET,
        "header_row": 1,
        "schema": expected_columns,
        "unit": BUDGET_UNIT,
        "sheet_roles": {
            RAW_SHEET: "원문 대조용 원본 시트",
            NORMALIZED_SHEET: "권위 있는 표준 입력 시트",
            VALIDATION_SHEET: "원본 블록별 열 매핑 QA 로그",
        },
        "validation_rows": len(validation),
        "validation_status": "정상",
    }
    return frame, metadata


def _adjacent_duplicate_rows(frame: pd.DataFrame) -> list[object]:
    compare_columns = [column for column in frame.columns if column != "원본행"]
    fingerprint = pd.util.hash_pandas_object(frame[compare_columns], index=False)
    return frame.loc[fingerprint.eq(fingerprint.shift()), "원본행"].tolist()


def _prepare_funding_rows(
    frame: pd.DataFrame,
    *,
    year: int,
    current_column: str,
    previous_column: str,
) -> pd.DataFrame:
    if year == 2018:
        truncated = frame["지역"].eq("광주") & frame["사업분류재정구분"].astype(
            "string"
        ).str.strip().eq("방비")
        frame.loc[truncated, "사업분류재정구분"] = "지방비"

        # 경북 "여성지도자 육성"(원본행 8200)의 2018년 예산이 원본 PDF에 "-20"으로
        # 인쇄돼 있다(전년도 45, 증감액 -65와 산술적으로는 일관됨). 예산이 음수일 수
        # 없어 원본 작성 단계의 오탈자로 판단했다.
        #
        # 소계·총계로 이 값을 독립 검증하는 건 구조적으로 불가능하다: 이 사업이
        # 속한 16개 세부사업 묶음(원본표구간 358) 자체에 소계 행이 없고, 더 상위
        # 총계(자체사업 합계 4,702억원)에 대면 40백만원 차이는 0.0000088%라
        # 어떤 허용오차로도 못 잡는다. 실제로 기존 예산 QA 파이프라인
        # (reports/yearly/2018/2018_전국_QA_검증결과.csv, 이슈 #15/#21)도 경북의
        # 중분류 6개 전부를 "판정불가(원본 소계값 결측)"로 남겨뒀다 — 경북 2018년
        # 원본 문서 자체가 이 방식의 검증을 지원하지 않는 지역이다.
        #
        # 검증 불가능함을 확인한 뒤, 재정팀 확인(2026-08-07, 팀 채팅)에 따라
        # 20으로 보정하기로 결정했다 — 방향(음수→양수)은 확실하지만 정확한 크기는
        # 사람 판단에 의존하는 값이다. reports/20260726_..._생성.md,
        # 이슈 #81 코멘트 참고.
        gyeongbuk_typo = frame["지역"].eq("경북") & frame["원본행"].eq(8200)
        matched = int(gyeongbuk_typo.sum())
        if matched != 1:
            raise ValueError(f"2018 경북 여성지도자 육성 보정 대상 행이 1개가 아닙니다: {matched}")
        original_value = frame.loc[gyeongbuk_typo, current_column].iloc[0]
        if float(original_value) != -20.0:
            raise ValueError(
                f"2018 경북 여성지도자 육성 원본값이 예상과 다릅니다: {original_value}"
            )
        frame.loc[gyeongbuk_typo, current_column] = 20

    if year == 2019:
        finance_type = frame["사업분류재정구분"].astype("string").str.strip()
        frame["사업분류재정구분"] = finance_type.replace({"｣계": "계"})
        known_blank_totals = {3618.0, 3621.0, 3624.0}
        present = known_blank_totals.intersection(set(frame["원본행"].dropna()))
        if present and present != known_blank_totals:
            raise ValueError(f"2019 경기 빈칸 총액 행 일부만 존재합니다: {sorted(present)}")
        frame.loc[frame["원본행"].isin(known_blank_totals), "사업분류재정구분"] = "계"

    confirmed = [] if year == 2016 else _adjacent_duplicate_rows(frame)
    frame = drop_exact_duplicate_rows(frame, confirmed_duplicate_rows=confirmed)
    return select_total_budget_rows(
        frame,
        budget_cols=[current_column, previous_column],
        zero_tokens=("-",),
    )


def _normalize_2019_titles(frame: pd.DataFrame) -> pd.DataFrame:
    detail_name = frame["세부사업명"].astype("string").str.strip()
    frame["세부사업명"] = detail_name.replace(
        {"I. 공통사업": "Ⅰ. 공통사업", "II. 자체사업": "Ⅱ. 자체사업"}
    )
    parts = (
        frame["세부사업명"]
        .astype("string")
        .str.strip()
        .str.extract(r"^(?P<number>\d+)\.\s*\((?P<budget_type>공통|자체)\)\s*(?P<title>.+?)\s*$")
    )
    matched = parts.notna().all(axis=1)
    frame.loc[matched, "세부사업명"] = (
        parts.loc[matched, "number"]
        + ". "
        + parts.loc[matched, "title"].str.strip()
        + "("
        + parts.loc[matched, "budget_type"]
        + "사업)"
    )
    return frame


def _classify_rows(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    patterns: list[str | re.Pattern[str]] = [UNIT_NOTATION_PATTERN]
    if year <= 2019:
        patterns.append(SUBTOTAL_LABEL_PATTERN)
    if year == 2018:
        patterns.append(re.compile(r"^\s*2018년도\s+지방자치단체\s+시행계획"))
    if year == 2019:
        patterns.append(re.compile(r"^(?:공통사업|자체사업)\s*총괄$"))
    if year == 2022:
        patterns.append(
            re.compile(r"고령사회기본계획|지방자치단체\s*시행계획|세부사업별\s*예산\s*현황")
        )

    frame["사업행구분"] = frame["세부사업명"].apply(
        lambda value: classify_row(value, extra_header_patterns=patterns)
    )
    names = frame["세부사업명"].astype("string").str.strip()

    if year == 2016:
        aggregate = frame["지역"].eq("광주") & names.isin(
            {"공통사업+자체사업", "공통사업 (저출산+고령사회)"}
        )
        gyeonggi = (
            frame["지역"].eq("경기")
            & frame["원본행"].between(3162, 3175)
            & ~frame["원본행"].eq(3165)
        )
        daegu = frame["지역"].eq("대구") & frame["원본행"].isin([1095, 1096])
        daejeon = frame["지역"].eq("대전")
        major = daejeon & names.str.contains(
            r"^\s*\[\s*(?:공\s*통|자\s*체)\s*사\s*업\s*\]", regex=True, na=False
        )
        medium = daejeon & names.str.contains(r"^[Ⅰ-Ⅿ]", regex=True, na=False)
        lower = (
            daejeon
            & ~major
            & ~medium
            & names.str.contains(r"\(\s*\d+\s*개\s*과제\s*\)\s*$", regex=True, na=False)
        )
        frame.loc[aggregate | gyeonggi | daegu | lower, "사업행구분"] = "헤더반복"
        frame.loc[major, "사업행구분"] = "대분류_소계"
        frame.loc[medium, "사업행구분"] = "중분류_소계"

    if year == 2017:
        gyeonggi = (
            frame["지역"].eq("경기")
            & frame["원본행"].between(5111, 5142)
            & ~frame["원본행"].between(5118, 5120)
        )
        seoul = frame["지역"].eq("서울") & names.isin({"1. 저출산 대책", "2. 고령화 대책"})
        daejeon = frame["지역"].eq("대전")
        medium = daejeon & names.str.contains(
            r"^[1-3]\.[\s\S]*\(\s*\d+\s*개\s*과제\s*\)\s*$", regex=True, na=False
        )
        lower = daejeon & names.str.match(r"^\d+\)", na=False)
        total = daejeon & names.str.match(r"^총\s*계", na=False)
        frame.loc[gyeonggi | lower | total, "사업행구분"] = "헤더반복"
        frame.loc[seoul | medium, "사업행구분"] = "중분류_소계"

    if year == 2018:
        aggregate = frame["지역"].eq("광주") & names.isin(
            {"공통사업+자체사업", "공통사업 (저출산+고령사회)"}
        )
        gwangju_medium = frame["지역"].eq("광주") & names.eq("3. 저출산·고령사회 대응기반 강화")
        gwangju_lower = (
            frame["지역"].eq("광주")
            & frame["사업행구분"].eq("세부사업")
            & names.str.match(r"^\s*(?:\d+-\d+|\d+\.)\s*", na=False)
        )
        seoul = frame["지역"].eq("서울") & names.isin({"1. 저출산 대책", "2. 고령화 대책"})
        daejeon = frame["지역"].eq("대전")
        daejeon_medium = daejeon & names.str.contains(
            r"^[1-3]\.[\s\S]*\(\s*\d+\s*개\s*과제\s*\)\s*$", regex=True, na=False
        )
        daejeon_lower = (
            daejeon
            & ~daejeon_medium
            & ~names.str.match(r"^[Ⅰ-Ⅿ]", na=False)
            & names.str.contains(r"\(?\s*\d+\s*개\s*과제\s*\)\s*$", regex=True, na=False)
        )
        daejeon_total = daejeon & names.str.match(r"^총\s*계", na=False)
        ulsan = frame["지역"].eq("울산") & names.eq("3. 대응기반 강화")
        frame.loc[
            aggregate | gwangju_lower | daejeon_lower | daejeon_total,
            "사업행구분",
        ] = "헤더반복"
        frame.loc[
            gwangju_medium | seoul | daejeon_medium | ulsan,
            "사업행구분",
        ] = "중분류_소계"

    if year == 2020:
        medium = names.str.match(r"^\d+\.", na=False) & names.str.contains(
            r"\((?:공통|자체)사업\)$", regex=True, na=False
        )
        auxiliary = names.str.match(r"^\d+\.", na=False) & names.str.contains(
            r"\((?:도|시|도비|시비)\s*자체사업\)$", regex=True, na=False
        )
        grand_total = names.str.replace(r"\s+", "", regex=True).isin(
            {"총계", "총계(공통사업+자체사업)", "합계"}
        )
        gyeongnam_heading = frame["지역"].eq("경남") & frame["원본행"].eq(7494)
        frame.loc[medium, "사업행구분"] = "중분류_소계"
        frame.loc[auxiliary | grand_total | gyeongnam_heading, "사업행구분"] = "헤더반복"

    return frame


def _merge_continuation_rows(
    frame: pd.DataFrame, targets: dict[int, int], *, label: str
) -> pd.DataFrame:
    """줄바꿈으로 잘린 세부사업명·주요내용을 뒤 연속행에서 원래 행으로 합친다.

    ``targets``는 ``{연속행_원본행: 대상행_원본행}``이다. 두 행 다 있어야만
    적용하고(일부만 있으면 실패), 합친 뒤에도 연속행 자체는 그대로 두므로
    호출부에서 별도로 leaf 제외(헤더반복 등) 처리를 해야 한다.
    """
    present = set(frame["원본행"].dropna()).intersection(targets)
    if not present:
        return frame
    if present != set(targets):
        raise ValueError(f"{label} 연속행 후보 일부만 존재합니다: {sorted(present)}")

    def append_once(original: object, continuation: object) -> str:
        left = "" if pd.isna(original) else re.sub(r"\s+", " ", str(original)).strip()
        right = "" if pd.isna(continuation) else re.sub(r"\s+", " ", str(continuation)).strip()
        if not right or left.endswith(right):
            return left
        return f"{left} {right}".strip()

    for continuation_row, target_row in targets.items():
        continuation_index = frame.index[frame["원본행"].eq(continuation_row)]
        target_index = frame.index[frame["원본행"].eq(target_row)]
        if len(continuation_index) != 1 or len(target_index) != 1:
            raise ValueError(f"{label} 연속행 병합 키가 유일하지 않습니다: {continuation_row}")
        continuation_index = continuation_index[0]
        target_index = target_index[0]
        frame.at[target_index, "세부사업명"] = append_once(
            frame.at[target_index, "세부사업명"], frame.at[continuation_index, "세부사업명"]
        )
        if pd.notna(frame.at[continuation_index, "주요내용"]):
            frame.at[target_index, "주요내용"] = append_once(
                frame.at[target_index, "주요내용"], frame.at[continuation_index, "주요내용"]
            )
    return frame


def _extract_leaf_rows(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    current_column = f"{year}년 예산"
    previous_column = f"{year - 1}년 예산"

    if year == 2019:
        frame = _normalize_2019_titles(frame)
    if year <= 2019:
        frame = _prepare_funding_rows(
            frame,
            year=year,
            current_column=current_column,
            previous_column=previous_column,
        )
    if year == 2021:
        frame = _merge_continuation_rows(
            frame,
            {175: 166, 482: 473, 7454: 7445, 7643: 7630, 7760: 7751, 7949: 7936},
            label="2021",
        )
    if year == 2022:
        # 서울 원본행 28("위한 안전망 구축")이 24행("학대피해아동 보호를")의
        # 줄바꿈 연속행이다. 기존 코드가 28행을 leaf에서 제외(헤더반복)하는
        # 것만 해뒀고 텍스트 병합을 빠뜨려 24행 사업명이 잘려 있었다 —
        # #73 감사 wide 파일(정상 병합됨) 대조로 발견
        # (reports/20260807_잠정재정패널_73감사wide_불일치_상세.csv).
        frame = _merge_continuation_rows(frame, {28: 24}, label="2022")

    frame = _classify_rows(frame, year)
    zero_tokens = ("-", "·", "비예산") if year == 2023 else ("-",)
    current_numeric = to_numeric_budget(frame[current_column], zero_tokens=zero_tokens)
    previous_numeric = to_numeric_budget(frame[previous_column], zero_tokens=zero_tokens)

    if year == 2021:
        frame.loc[current_numeric.isna() & previous_numeric.isna(), "사업행구분"] = "헤더반복"

    if year == 2022:
        continuation = frame["지역"].eq("서울") & frame["원본행"].eq(28)
        frame.loc[continuation, "사업행구분"] = "헤더반복"
        labeled = frame.groupby("지역", group_keys=False).apply(assign_labels)
        labeled["지역"] = frame.loc[labeled.index, "지역"]
        structure_invalid = (
            labeled["대분류"].isna()
            | labeled["중분류"].isna()
            | ~labeled["사업분류재정구분"].isin(["공통", "자체"])
        )
        leaf = labeled.loc[labeled["사업행구분"].eq("세부사업") & ~structure_invalid].copy()
    else:
        leaf = frame.loc[frame["사업행구분"].eq("세부사업")].copy()

    leaf["예산액_원본"] = frame.loc[leaf.index, current_column]
    leaf["예산액"] = current_numeric.loc[leaf.index]
    leaf["전년도예산_백만원"] = previous_numeric.loc[leaf.index]
    leaf["예산_비수치"] = leaf["예산액_원본"].notna() & leaf["예산액"].isna()
    leaf["예산_결측"] = leaf["예산액"].isna()
    leaf["예산_음수"] = leaf["예산액"].lt(0)
    # 원본 셀의 줄바꿈을 그대로 남기면 #73 감사 wide 파일(줄바꿈을 공백 한 칸으로
    # 정리)과 텍스트가 달라 보인다 — 같은 정리 유틸을 적용해 일치시킨다.
    leaf["세부사업명"] = clean_text(leaf["세부사업명"])
    leaf["주요내용"] = clean_text(leaf["주요내용"])
    return leaf


def read_raw_file_list(
    paths: Sequence[str | Path],
    *,
    expected_regions: Sequence[str] = STANDARD_REGIONS,
    expected_years: Sequence[int] = EXPECTED_YEARS,
) -> pd.DataFrame:
    """Read an explicit, validated list of normalized annual workbooks.

    No directory discovery and no first-sheet fallback are performed.
    """

    regions = [str(region).strip() for region in expected_regions]
    years = [int(year) for year in expected_years]
    if len(regions) != len(set(regions)) or len(years) != len(set(years)):
        raise ValueError("예상 지역·연도 인자에 중복이 있습니다.")

    details: list[pd.DataFrame] = []
    workbook_metadata: list[dict] = []
    for year, path in validate_input_files(paths, expected_years=years):
        frame, metadata = _read_workbook(path, year)
        if year == 2019:
            frame["지역"] = frame["지역"].replace({"충청": "충북"})
        actual_regions = set(frame["지역"].dropna().astype("string").str.strip())
        missing = sorted(set(regions) - actual_regions)
        unexpected = sorted(actual_regions - set(regions))
        if missing or unexpected:
            raise ValueError(f"{path}: 지역 불일치; 누락={missing}, 예상외={unexpected}")

        leaf = _extract_leaf_rows(frame.copy(), year)
        if leaf.empty:
            raise ValueError(f"{path}: 세부사업 행이 없습니다.")
        leaf["지역"] = leaf["지역"].astype("string").str.strip()
        leaf["연도"] = year
        leaf["예산구분"] = "당해예산"
        leaf["원본파일"] = str(path.resolve())
        leaf["원본시트"] = NORMALIZED_SHEET
        leaf["예산단위"] = BUDGET_UNIT
        details.append(
            leaf[
                [
                    "지역",
                    "연도",
                    "세부사업명",
                    "예산구분",
                    "예산액",
                    "예산액_원본",
                    "전년도예산_백만원",
                    "사업분류재정구분",
                    "원본행",
                    "예산_비수치",
                    "예산_결측",
                    "예산_음수",
                    "원본파일",
                    "원본시트",
                    "예산단위",
                ]
            ]
        )
        metadata["path"] = str(path.resolve())
        metadata["region_count"] = len(actual_regions)
        metadata["leaf_rows"] = len(leaf)
        workbook_metadata.append(metadata)

    detail = pd.concat(details, ignore_index=True)
    duplicate_keys = detail.duplicated(["지역", "연도", "원본행"], keep=False)
    if duplicate_keys.any():
        samples = detail.loc[duplicate_keys, ["지역", "연도", "원본행"]].head(10)
        raise ValueError(f"세부사업 원본 키 중복: {samples.to_dict(orient='records')}")

    actual_pairs = set(
        map(tuple, detail[["지역", "연도"]].drop_duplicates().itertuples(index=False, name=None))
    )
    expected_pairs = {(region, year) for region in regions for year in years}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        unexpected = sorted(actual_pairs - expected_pairs)
        raise ValueError(
            f"세부사업 지역×연도 불일치: 누락={missing[:10]}, 예상외={unexpected[:10]}"
        )

    region_order = {region: index for index, region in enumerate(regions)}
    detail["_지역순서"] = detail["지역"].map(region_order)
    detail = detail.sort_values(["연도", "_지역순서", "원본행"]).drop(columns="_지역순서")
    detail = detail.reset_index(drop=True)
    detail.attrs["workbooks"] = workbook_metadata
    detail.attrs["schema"] = _required_normalized_columns(years[0])
    detail.attrs["unit"] = BUDGET_UNIT
    return detail
