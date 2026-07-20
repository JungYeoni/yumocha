from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.features.text_patterns import PUA_PATTERN

SIDO_TITLE_PATTERN = re.compile(r"붙\s*임\s*\(([^)]+)\)")
MAJOR_CATEGORY_PATTERN = re.compile(r"^[Ⅰ-Ⅿ]")
MEDIUM_CATEGORY_PATTERN = re.compile(r"^\d+\.")
BUDGET_TYPE_SUFFIX_PATTERN = re.compile(r"\((공통|자체)\)$")

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
    row_type_col: str = "사업행구분",
) -> pd.DataFrame:
    """원본 중분류 소계와 세부사업 예산 합계를 비교한다.

    ``budget_col``은 숫자 변환이 완료된 컬럼이어야 한다. 한쪽 그룹이 없거나
    비교값이 결측인 경우는 일치로 처리하지 않는다.
    """
    if tolerance < 0:
        raise ValueError("tolerance는 0 이상이어야 합니다.")

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


__all__ = [
    "assign_labels",
    "build_subtotal_qa",
    "calculate_budget_changes",
    "classify_row",
    "clean_text",
    "get_sido_dir",
    "normalize_budget_type",
    "show_table1_around",
    "to_numeric_budget",
]
