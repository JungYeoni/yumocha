"""청년층 정규직 근로자 비율의 전국(9개년) 값을 지역별고용조사 원자료에서 직접 계산한다.

`notebooks/20260727_EDA_지역별고용조사_청년층_정규직_근로자_비율_산출.ipynb`가 이미
17개 시도값을 산출한 것과 정확히 같은 산식·컬럼 별칭·가중치 선택을 그대로 쓰되,
`시도코드`로 그룹핑하지 않고 전체 응답을 합산해 전국값을 만든다. 반기→연도 집계
방식(연평균, 반올림 전 반기 비율의 단순평균을 한 번만 반올림)도 그 노트북에서
팀 논의로 확정한 것과 동일하게 따른다(재발명 금지).

## 알려진 한계

`가중치` 컬럼은 반기에 따라 다르다 — 상반기는 이름에 "전국"이 명시된
`시도전국가중값`을, 하반기는 `시도가중값`을 쓴다. 기존 노트북은 시도별로만
그룹핑해 써서 이 차이가 문제되지 않았지만, 이번처럼 지역 구분 없이 전체를
합산할 때 하반기 가중치가 전국 합산에도 통계적으로 타당한지는 공식 코드북·
통계설명자료로 확인하지 못했다. 두 가중치를 이미 노트북이 동일한 `가중치`
컬럼으로 취급해온 기존 관례를 그대로 확장했을 뿐, 이 가정 자체를 새로
검증한 것은 아니다.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "구조환경지수 원데이터 구축용" / "지역별고용조사"
PROVINCE_ANNUAL_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "analysis"
    / "2016-2024_지역별고용조사_청년층_정규직_근로자_비율_연도평균.csv"
)
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260805_청년층_정규직_근로자_비율_전국_직접계산_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260805_청년층_정규직_근로자_비율_전국_직접계산_QA.csv"

INDICATOR_ID = "youth_regular_employment_rate"
HALF_ORDER = {"상반기": 0, "하반기": 1}
COLUMN_ALIASES = {
    "시도코드": ["행정구역시도코드", "2자리_행정구역시도코드"],
    "만연령": ["만연령"],
    "경제활동상태": ["경제활동인구상태코드", "경제활동구분코드"],
    "종사상지위": ["종사상지위코드", "현직장종사상지위코드"],
    "근무시간구분": ["주업부업총계시간구분코드"],
    "가중치": ["시도전국가중값", "시도가중값"],
}
COMMON_COLUMNS = list(COLUMN_ALIASES)


def index_raw_files(raw_dir: Path) -> pd.DataFrame:
    file_records = []
    for path in raw_dir.glob("*.csv"):
        match = re.match(r"^(20\d{2})_(상반기|하반기)\(C형", path.name)
        if match:
            year, half = match.groups()
            file_records.append({"연도": int(year), "반기": half, "경로": path})
    file_index = (
        pd.DataFrame(file_records)
        .assign(반기순서=lambda x: x["반기"].map(HALF_ORDER))
        .sort_values(["연도", "반기순서"])
        .drop(columns="반기순서")
        .reset_index(drop=True)
    )
    if len(file_index) != 18:
        raise ValueError(f"반기 원자료 파일이 18개가 아닙니다: {len(file_index)}개")
    return file_index


def _rename_common_columns(data: pd.DataFrame, path_name: str) -> pd.DataFrame:
    rename_map = {}
    for common, aliases in COLUMN_ALIASES.items():
        matched = [alias for alias in aliases if alias in data.columns]
        if not matched:
            raise KeyError(
                f"{path_name}: '{common}'에 해당하는 컬럼을 찾지 못했습니다 (후보: {aliases})"
            )
        if len(matched) > 1:
            raise ValueError(f"{path_name}: '{common}' 후보가 둘 이상입니다({matched})")
        rename_map[matched[0]] = common
    return data.rename(columns=rename_map)[COMMON_COLUMNS]


def calculate_national_rate(path: Path) -> float:
    """시도 구분 없이 전체 응답을 합산해 전국 청년층 정규직 근로자 비율을 계산한다."""
    data = pd.read_csv(path, encoding="cp949")
    data = _rename_common_columns(data, path.name)

    denominator = (
        data["만연령"].between(19, 34)
        & data["경제활동상태"].eq(1)
        & data["종사상지위"].isin([1, 2])
    )
    numerator = denominator & data["종사상지위"].eq(1) & data["근무시간구분"].eq(3)

    denominator_weight = data.loc[denominator, "가중치"].sum()
    if not denominator_weight > 0:
        raise ValueError(f"{path.name}: 분모 가중치 합이 0 이하입니다({denominator_weight}).")
    numerator_weight = data.loc[numerator, "가중치"].sum()
    return float(numerator_weight / denominator_weight * 100)


def build_candidates(
    raw_dir: Path, province_annual_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_index = index_raw_files(raw_dir)
    half_rates = {
        (row.연도, row.반기): calculate_national_rate(row.경로)
        for row in file_index.itertuples(index=False)
    }
    years = sorted({year for year, _ in half_rates})

    qa_records: list[dict[str, object]] = []

    def check(item: str, expected: object, actual: object) -> None:
        qa_records.append(
            {
                "검사항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "PASS" if expected == actual else "FAIL",
            }
        )

    check("반기 원자료 파일 수", 18, len(file_index))
    check("연도 수", 9, len(years))

    province_annual = pd.read_csv(province_annual_path, encoding="utf-8-sig")
    province_annual = province_annual.set_index("시도")

    rows = []
    for year in years:
        raw_annual = (half_rates[(year, "상반기")] + half_rates[(year, "하반기")]) / 2
        rounded = round(raw_annual, 1)
        province_values = province_annual[str(year)]
        check(
            f"{year} 전국값이 17개 시도 값 범위 안에 있음",
            True,
            bool(province_values.min() <= raw_annual <= province_values.max()),
        )
        rows.append(
            {
                "지역": "전국",
                "지표_id": INDICATOR_ID,
                "연도": year,
                "측정값": rounded,
                "반기별_원값_상반기": half_rates[(year, "상반기")],
                "반기별_원값_하반기": half_rates[(year, "하반기")],
                "관측상태": "관측",
                "QA_상태": "PASS",
                "반영유형": "원자료 마이크로데이터 직접계산",
                "방법": "원자료 마이크로데이터 전국 가중합(연평균)",
                "한계": (
                    "하반기 가중치(시도가중값)의 전국 합산 타당성은 공식 코드북으로 "
                    "검증하지 못함 — 기존 노트북이 시도별 계산에 쓰던 것과 동일한 "
                    "가중치를 전국 단위로 확장 적용"
                ),
            }
        )

    candidates = pd.DataFrame(rows)
    check("전국 후보 행 수", 9, len(candidates))
    check("측정값 결측 없음", 0, int(candidates["측정값"].isna().sum()))

    qa = pd.DataFrame(qa_records)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"전국 직접계산 QA 실패: {failures.to_dict('records')}")
    return candidates, qa


def apply_national_candidates_to_panel(
    panel: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전국 9개년 후보를 28개 지표 롱 패널에 반영한다.

    `apply_national_observations_to_panel`(build_family_friendly_national_candidates.py)과
    같은 원칙(기존 값이 결측이 아니면 예외)을 쓰지만, 그 함수는 후보 5행을 하드코딩해
    검증하므로 9행인 이 지표에는 그대로 재사용할 수 없어 대응 버전을 새로 둔다.
    """
    required = {"지역", "지표_id", "연도", "측정값"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing_columns, key=str)}")

    keys = ["지역", "지표_id", "연도"]
    if len(candidates) != 9 or candidates.duplicated(keys).any():
        raise ValueError("전국 후보표는 중복 없는 9개 키여야 합니다.")
    if not candidates["지역"].eq("전국").all():
        raise ValueError("전국 후보표에 전국 이외 지역이 포함됐습니다.")
    if not candidates["QA_상태"].eq("PASS").all():
        raise ValueError("QA를 통과하지 않은 후보값은 반영할 수 없습니다.")

    result = panel.copy()
    audit_rows = []
    for row in candidates.to_dict("records"):
        mask = (
            result["지역"].eq(row["지역"])
            & result["지표_id"].eq(row["지표_id"])
            & result["연도"].eq(row["연도"])
        )
        matched = int(mask.sum())
        if matched != 1:
            raise ValueError(f"전국 후보 반영 키가 1개가 아닙니다({matched}개): {row['연도']}")
        before = result.loc[mask, "측정값"].iloc[0]
        if pd.notna(before):
            raise ValueError(f"전국 직접계산 대상의 기존 패널값이 이미 존재합니다: {row['연도']}")
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


def remove_resolved_rows_from_mapping(
    mapping: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """이제 관측값이 생긴 전국 9개 키를 결측정책 매핑에서 제거한다."""
    keys = ["지역", "지표_id", "연도"]
    mask = (
        mapping["지역"].eq("전국")
        & mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["연도"].isin(candidates["연도"])
    )
    matched = mapping.loc[mask]
    if len(matched) != 9:
        raise ValueError(f"매핑에서 제거 대상 전국 9건과 일치하지 않습니다: {len(matched)}건")
    if not matched.merge(candidates[keys], on=keys, how="inner").shape[0] == 9:
        raise ValueError("매핑의 전국 행 키가 후보표 키와 정확히 대응하지 않습니다.")
    if not (matched["block_imputation"] & matched["imputation_policy"].eq("pending_review")).all():
        raise ValueError("제거 대상 행이 예상한 pending_review/block_imputation 상태가 아닙니다.")

    return mapping.loc[~mask].reset_index(drop=True), matched.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--province-annual", type=Path, default=PROVINCE_ANNUAL_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="28개 지표 통합 패널 CSV. 존재할 때만 전국 9건을 반영한다(기본: 반영 안 함).",
    )
    parser.add_argument("--panel-output", type=Path, default=None)
    parser.add_argument(
        "--panel-audit-output",
        type=Path,
        default=Path("reports/20260805_청년층_정규직_근로자_비율_전국_패널반영_감사.csv"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="결측정책 전수매핑 CSV. 존재할 때만 해소된 전국 9행을 제거한다(기본: 반영 안 함).",
    )
    parser.add_argument("--mapping-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, qa = build_candidates(args.raw_dir, args.province_annual)

    for path in (args.candidate_output, args.qa_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(f"전국 후보: {len(candidates)}건 ({args.candidate_output})")
    print(f"QA {len(qa)}개 항목 모두 PASS")

    if args.panel is not None and not args.panel.exists():
        raise FileNotFoundError(f"--panel 경로가 존재하지 않습니다: {args.panel}")
    if args.panel is not None and args.panel.exists():
        panel = pd.read_csv(args.panel, encoding="utf-8-sig")
        updated_panel, panel_audit = apply_national_candidates_to_panel(panel, candidates)
        panel_output = args.panel_output or args.panel
        args.panel_audit_output.parent.mkdir(parents=True, exist_ok=True)
        updated_panel.to_csv(panel_output, index=False, encoding="utf-8-sig")
        panel_audit.to_csv(args.panel_audit_output, index=False, encoding="utf-8-sig")
        print(f"패널 반영 완료: {panel_output} (감사: {args.panel_audit_output})")

    if args.mapping is not None and not args.mapping.exists():
        raise FileNotFoundError(f"--mapping 경로가 존재하지 않습니다: {args.mapping}")
    if args.mapping is not None and args.mapping.exists():
        mapping = pd.read_csv(args.mapping, encoding="utf-8-sig")
        updated_mapping, removed = remove_resolved_rows_from_mapping(mapping, candidates)
        mapping_output = args.mapping_output or args.mapping
        updated_mapping.to_csv(mapping_output, index=False, encoding="utf-8-sig")
        print(f"매핑 갱신 완료: {mapping_output} ({len(removed)}행 제거)")


if __name__ == "__main__":
    main()
