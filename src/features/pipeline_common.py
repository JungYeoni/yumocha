from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.features.text_patterns import PUA_PATTERN

FUNDING_SOURCE_TOKENS = ("계", "국비", "지방비", "도비", "시군비", "시비", "기금", "비예산")
TOTAL_FUNDING_TOKEN = "계"
SIDO_TITLE_PATTERN = re.compile(r"붙\s*임\s*\(([^)]+)\)")
MAJOR_CATEGORY_PATTERN = re.compile(r"^[Ⅰ-Ⅿ]")
MEDIUM_CATEGORY_PATTERN = re.compile(r"^\d+\.")
BUDGET_TYPE_SUFFIX_PATTERN = re.compile(r"\((공통|자체)(사업)?\)$|(공통|자체)사업$|\((도|시군)\)$")

# 2016~2020(제3차 기본계획) 원본 특유의 단위표기 헤더 행. 2021년 이후 원본에는 없으므로
# classify_row에 내장하지 않고, 해당 연도 노트북에서 extra_header_patterns로 전달한다.
# 예: "(단위 : 백만원)", "(단위：백만원)", "(단위:백만원)"
UNIT_NOTATION_PATTERN = re.compile(r"^\(\s*단위\s*[:：]")

# 2016~2020(제3차 기본계획) 원본 특유의 소계/합계 라벨 행. 대분류·중분류 제목 행과
# 숫자(계/국비/지방비)가 서로 다른 행으로 분리돼 있고, 숫자가 있는 행은 카테고리명 없이
# "소계"/"합계" 같은 일반 라벨만 붙어 있다. classify_row 텍스트 매칭만으로는 이 숫자 행을
# 세부사업과 구분할 수 없어 내장하지 않고 extra_header_patterns로 전달한다.
# 예: "총 계", "총계", "소계", "공통사업 합계", "자체사업 합계",
#     "총 계(230개 과제) (공통 88, 자체 142)"(대전, 괄호 부가설명 포함)
SUBTOTAL_LABEL_PATTERN = re.compile(r"^(총\s*계|소계)(\s*\(.*\))?$|.*합계$")

# 문자열 맨 앞의 불릿만 제거
# 문장 중간의 하이픈이나 가운데점은 보존
LEADING_BULLET_PATTERN = re.compile(r"^\s*[ㅇ○◦□▪·•o\-]\s*")


def get_sido_dir(interim_dir: str | Path, sido: str) -> Path:
    """시도별 중간 산출물 디렉터리를 생성하고 경로를 반환한다."""
    path = Path(interim_dir) / sido
    path.mkdir(parents=True, exist_ok=True)  # 해당 폴더가 없으면 생성
    return path


def classify_row(
    detail_name: object,
    *,
    extra_header_patterns: Iterable[str | re.Pattern[str]] = (),
) -> str:
    """
    세부사업명 값을 이용해 원본 행의 유형을 분류

    parmeters:
      detail_name: 세부사업명 값
      extra_header_patterns: 특정 연도에서만 나타나는 반복 머리글 패턴 (문자열 또는 컴파일된 정규식 전달)

    returns:
      str: 대분류_소계, 중분류_소계, 헤더반복, 세부사업 중 하나

    """
    if pd.isna(detail_name):
        return "헤더반복"

    name = str(detail_name).strip()

    if not name or name == "세부사업명":
        return "헤더반복"

    if SIDO_TITLE_PATTERN.search(name):
        return "헤더반복"

    for pattern in extra_header_patterns:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        if compiled.search(name):
            return "헤더반복"

    if MAJOR_CATEGORY_PATTERN.match(name):
        return "대분류_소계"

    if MEDIUM_CATEGORY_PATTERN.match(name) and BUDGET_TYPE_SUFFIX_PATTERN.search(name):
        return "중분류_소계"

    return "세부사업"


def assign_labels(
    df_sido: pd.DataFrame,
    *,
    row_order_col: str = "원본행",
    row_type_col: str = "사업행구분",
    detail_name_col: str = "세부사업명",
) -> pd.DataFrame:
    """대분류 / 중분류 소계의 명칭을 뒤따르는 행에 전파한다."""

    required_cols = {
        row_order_col,
        row_type_col,
        detail_name_col,
    }
    missing_cols = required_cols.difference(df_sido.columns)

    if missing_cols:
        raise KeyError(f"계층 라벨 전파에 필요한 컬럼이 없습니다. {sorted(missing_cols)}")

    result = df_sido.sort_values(row_order_col).copy()

    major_mask = result[row_type_col].eq("대분류_소계")
    medium_mask = result[row_type_col].eq("중분류_소계")

    result["대분류"] = result[detail_name_col].where(major_mask).ffill()
    result["중분류"] = result[detail_name_col].where(medium_mask).ffill()

    return result


def clean_text(
    series: pd.Series,
    *,
    strip_leading_bullet: bool = False,
) -> pd.Series:
    """
    PUA 문자를 제거하고 줄바꿈 / 연속 공백을 일반 공백 한 칸으로 정리한다.
    세부사업명의 단어 사이 공백은 제거하지 않는다.
    strip_leading_bullet=True 인 경우 문자열 맨 앞의 불릿만 제거한다.
    """

    def _clean(value: object) -> object:
        if pd.isna(value):
            return value

        text = PUA_PATTERN.sub("", str(value))
        text = re.sub(r"\s+", " ", text).strip()

        if strip_leading_bullet:
            text = LEADING_BULLET_PATTERN.sub("", text).strip()

        return text

    return series.apply(_clean)


def normalize_budget_type(series: pd.Series) -> pd.Series:
    """공통/자체 구분값의 불필요한 공백 제거"""
    return clean_text(series).astype("string").str.replace(r"\s+", "", regex=True)


def to_numeric_budget(
    series: pd.Series,
    *,
    zero_tokens: Iterable[str] = ("-",),
) -> pd.Series:
    """
    예산 문자열을 숫자형으로 변환 (실제 결측값은 0으로 변환하지 않고 결측 상태를 유지 / 숫자로 해석할 수 없는 값도 NaN으로 변환)

    parameters:
      series: 반환할 예산 컬럼
      zero_tokens: 예산 0으로 해석하기로 확정한 원본 표기

    """
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()

    normalized_zero_tokens = {str(token).strip() for token in zero_tokens}

    cleaned = cleaned.mask(cleaned.isin(normalized_zero_tokens), "0")

    return pd.to_numeric(cleaned, errors="coerce")


def calculate_budget_changes(
    current_budget: pd.Series,
    previous_budget: pd.Series,
    *,
    rate_digits: int = 1,
) -> pd.DataFrame:
    """당해·전년도 예산으로 증감액과 증감율을 재계산한다.

    두 입력은 같은 인덱스를 가져야 한다. 전년도예산이 0이거나 어느 한쪽이
    결측이면 증감율은 결측으로 유지한다.
    """
    if not current_budget.index.equals(previous_budget.index):
        raise ValueError("당해예산과 전년도예산의 인덱스가 같아야 합니다.")

    if rate_digits < 0:
        raise ValueError("rate_digits는 0 이상이어야 합니다.")

    current = pd.to_numeric(current_budget, errors="coerce")
    previous = pd.to_numeric(previous_budget, errors="coerce")
    change = current - previous
    denominator = previous.mask(previous.eq(0))
    rate = change.div(denominator).mul(100).round(rate_digits)

    return pd.DataFrame(
        {
            "당해예산": current,
            "전년도예산": previous,
            "증감액": change,
            "증감율": rate,
        },
        index=current.index,
    )


def build_subtotal_qa(
    df_labeled: pd.DataFrame,
    *,
    budget_col: str,
    group_cols: tuple[str, ...] = ("지역", "대분류", "중분류"),
    tolerance: float = 0,
    rate_tolerance: float = 10.0,  # 허용 오차율 상한
    row_type_col: str = "사업행구분",
) -> pd.DataFrame:
    """원본 중분류 소계와 세부사업 예산 합계를 비교한다.

    ``budget_col``은 숫자 변환이 완료된 컬럼이어야 한다. ``rate_tolerance``는
    절대 오차율의 허용 상한이다. 한쪽 그룹이 없거나 비교값이 결측인 경우는
    허용 여부를 판정하지 않는다.
    """
    if tolerance < 0:
        raise ValueError("tolerance는 0 이상이어야 합니다.")
    if rate_tolerance < 0:
        raise ValueError("rate_tolerance는 0 이상이어야 합니다.")

    if not group_cols:
        raise ValueError("group_cols에는 하나 이상의 컬럼이 필요합니다.")

    required_cols = {*group_cols, budget_col, row_type_col}
    missing_cols = required_cols.difference(df_labeled.columns)
    if missing_cols:
        raise KeyError(f"소계 QA에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    if not is_numeric_dtype(df_labeled[budget_col]):
        raise TypeError(f"{budget_col} 컬럼은 숫자형이어야 합니다.")

    leaf = df_labeled.loc[df_labeled[row_type_col].eq("세부사업")]
    subtotal = df_labeled.loc[
        df_labeled[row_type_col].eq("중분류_소계"),
        [*group_cols, budget_col],
    ].rename(columns={budget_col: "원본_소계값"})

    duplicate_subtotal = subtotal.duplicated(list(group_cols), keep=False)
    if duplicate_subtotal.any():
        duplicate_groups = subtotal.loc[duplicate_subtotal, list(group_cols)].drop_duplicates()
        raise ValueError(
            "같은 QA 그룹에 중분류 소계가 여러 행 존재합니다: "
            f"{duplicate_groups.to_dict(orient='records')}"
        )

    leaf_sum = (
        leaf.groupby(list(group_cols), dropna=False)[budget_col]
        .sum(min_count=1)
        .rename("leaf_합계")
        .reset_index()
    )

    qa = subtotal.merge(
        leaf_sum,
        on=list(group_cols),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    qa["QA_병합상태"] = qa.pop("_merge").map(
        {
            "left_only": "원본소계만",
            "right_only": "leaf합계만",
            "both": "양쪽존재",
        }
    )
    qa["차이"] = qa["leaf_합계"] - qa["원본_소계값"]
    subtotal_denominator = qa["원본_소계값"].mask(qa["원본_소계값"].eq(0))

    qa["오차율(%)"] = qa["차이"].div(subtotal_denominator).mul(100).round(2)

    qa["허용기준결과"] = "판정불가"

    rate_comparable = qa["QA_병합상태"].eq("양쪽존재") & qa["오차율(%)"].notna()

    qa.loc[
        rate_comparable & qa["오차율(%)"].abs().le(rate_tolerance),
        "허용기준결과",
    ] = "허용"

    qa.loc[
        rate_comparable & qa["오차율(%)"].abs().gt(rate_tolerance),
        "허용기준결과",
    ] = "초과"

    comparable = (
        qa["QA_병합상태"].eq("양쪽존재") & qa["원본_소계값"].notna() & qa["leaf_합계"].notna()
    )
    qa["결과"] = "불일치"
    qa.loc[comparable & qa["차이"].abs().le(tolerance), "결과"] = "일치"

    return qa[
        [
            *group_cols,
            "원본_소계값",
            "leaf_합계",
            "차이",
            "오차율(%)",
            "QA_병합상태",
            "결과",
            "허용기준결과",
        ]
    ]


def show_table1_around(
    df_table1: pd.DataFrame,
    center_excel_row: int,
    *,
    window: int = 3,
    label: str = "",
    column_indices: tuple[int, ...] = (0, 2, 3),
    column_names: tuple[str, ...] = (
        "세부사업명",
        "공통/자체",
        "예산",
    ),
) -> pd.DataFrame:
    """Table 1에서 특정 엑셀 행 주변을 확인할 DataFrame으로 반환한다.

    ``center_excel_row``는 엑셀 기준의 1부터 시작하는 행 번호다.
    기본 컬럼 인덱스는 수정된 기준인 ``(0, 2, 3)``을 사용한다.
    """
    if center_excel_row < 1:
        raise ValueError("center_excel_row는 1 이상이어야 합니다.")

    if window < 0:
        raise ValueError("window는 0 이상이어야 합니다.")

    if len(column_indices) != len(column_names):
        raise ValueError("column_indices와 column_names의 길이가 같아야 합니다.")

    start_excel_row = max(1, center_excel_row - window)
    end_excel_row = center_excel_row + window

    view = df_table1.iloc[
        start_excel_row - 1 : end_excel_row,
        list(column_indices),
    ].copy()

    view = view.dropna(axis=1, how="all")

    if view.shape[1] == len(column_names):
        view.columns = list(column_names)

    if label:
        print(f"--- {label} (Table 1 엑셀행 {start_excel_row}~{end_excel_row}) ---")

    return view


def drop_exact_duplicate_rows(
    df_raw: pd.DataFrame,
    *,
    ignore_cols: tuple[str, ...] = ("원본행",),
) -> pd.DataFrame:
    """행 위치 정보(원본행)를 제외한 나머지 컬럼이 완전히 동일한 행을 하나만 남긴다.

    2016~2019 `정리본_자동` 생성 스크립트가 병합셀을 셀 단위로 잘못 분리하면서
    같은 재정구분 라벨(예: "비예산")이 완전히 같은 값으로 두 번 찍히는 경우가 있다
    (실제 확인: 부산 "출산 기념 축하 미역 지원" 비예산 행이 원본행만 다르고 나머지는
    전부 동일하게 2회 반복). `사업분류재정구분`(계/국비/지방비 등) 라벨까지 포함해서
    모든 값이 같아야 중복으로 보므로, 국비와 지방비 값이 우연히 같은 정상적인 재원
    분할(예: 국비=지방비=2300인 5:5 분할)은 라벨이 다르기 때문에 건드리지 않는다.
    """
    compare_cols = [column for column in df_raw.columns if column not in ignore_cols]
    is_duplicate = df_raw.duplicated(subset=compare_cols, keep="first")
    return df_raw.loc[~is_duplicate].copy()


def select_total_budget_rows(
    df_raw: pd.DataFrame,
    *,
    budget_cols: Sequence[str],
    finance_type_col: str = "사업분류재정구분",
    group_cols: tuple[str, ...] = ("지역", "머리글행", "세부사업명"),
    funding_source_tokens: Iterable[str] = FUNDING_SOURCE_TOKENS,
    total_token: str = TOTAL_FUNDING_TOKEN,
    zero_tokens: Iterable[str] = ("-",),
) -> pd.DataFrame:
    """계/국비/지방비 등 재원별로 나뉜 행을 세부사업당 한 행으로 정리한다.

    2016~2019(제3차 기본계획) 원본은 세부사업 하나가 계/국비/지방비(도비·시군비·비예산 등)
    최대 3~4개 행으로 나뉘어 있다(국비+지방비=계로 검증됨). `drop_exact_duplicate_rows`로
    셀 분리 버그성 완전 중복을 먼저 제거한 뒤 이 함수를 호출해야 한다.

    같은 그룹(지역·머리글행·세부사업명) 안에 "계" 행이 있으면 그 행만 남기고 국비·
    지방비 등 나머지는 버린다. "계"가 없으면 그룹 내 재원구분 행들의 budget_cols
    값을 숫자로 변환해 합산한 대표 행 하나로 축약한다(전액 지방비인 경우 지방비
    값이 그대로 합산 결과가 되고, 국비+지방비만 있고 계가 없는 경우 둘을 더한다).
    대표 행의 budget_cols 외 컬럼은 그룹의 첫 행 기준이다. finance_type_col 값이
    재원구분 토큰이 아닌 행(헤더, 결측 등)은 건드리지 않고 그대로 통과시킨다.

    2020년 이후(공통/자체 체계) 원본에는 이 토큰들이 없으므로 이 함수를 적용할
    필요가 없다.
    """
    missing_cols = {finance_type_col, *group_cols, *budget_cols}.difference(df_raw.columns)
    if missing_cols:
        raise KeyError(f"재원구분 필터링에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    finance_type = df_raw[finance_type_col].astype("string").str.strip()
    normalized_tokens = {str(token).strip() for token in funding_source_tokens}
    is_funding_row = finance_type.isin(normalized_tokens)

    other_rows = df_raw.loc[~is_funding_row]
    funding_rows = df_raw.loc[is_funding_row].copy()
    funding_rows["_재원구분"] = finance_type.loc[is_funding_row]

    group_has_total = funding_rows.groupby(list(group_cols), dropna=False)["_재원구분"].transform(
        lambda s: s.eq(total_token).any()
    )
    is_total_row = funding_rows["_재원구분"].eq(total_token)

    total_rows = funding_rows.loc[is_total_row & group_has_total].drop(columns="_재원구분")

    remainder = funding_rows.loc[~group_has_total].drop(columns="_재원구분").copy()
    if remainder.empty:
        aggregated = remainder
    else:
        for column in budget_cols:
            remainder[column] = to_numeric_budget(remainder[column], zero_tokens=zero_tokens)

        summed_budget = remainder.groupby(list(group_cols), as_index=False, dropna=False)[
            list(budget_cols)
        ].sum(min_count=1)
        representative = remainder.drop_duplicates(subset=list(group_cols), keep="first").drop(
            columns=list(budget_cols)
        )
        aggregated = representative.merge(summed_budget, on=list(group_cols), how="left")
        aggregated[finance_type_col] = total_token

    return pd.concat([other_rows, total_rows, aggregated], ignore_index=True)


__all__ = [
    "FUNDING_SOURCE_TOKENS",
    "SUBTOTAL_LABEL_PATTERN",
    "TOTAL_FUNDING_TOKEN",
    "UNIT_NOTATION_PATTERN",
    "assign_labels",
    "build_subtotal_qa",
    "calculate_budget_changes",
    "classify_row",
    "clean_text",
    "drop_exact_duplicate_rows",
    "get_sido_dir",
    "normalize_budget_type",
    "select_total_budget_rows",
    "show_table1_around",
    "to_numeric_budget",
]
