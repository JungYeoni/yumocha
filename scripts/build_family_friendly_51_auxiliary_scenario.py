"""가족친화 인증기업 비율 17개 시도 34건(2017·2019)의 보조 시나리오 후보를 만든다.

`reports/methodology/20260805_가족친화_지역별_51건_보간_방법_제안.md`에서 제안한 방법을
실제로 계산한다. 이 34건은 여전히 공식 명단을 확보하지 못해 `pending_review`/
`block_imputation=True`로 본계열에서 차단돼 있으며, 이 스크립트는 그 상태를 바꾸지
않는다 — 결과는 `auxiliary_scenario_policy` 열에만 별도로 기록되는 민감도 비교용
후보값이다.

**2026-08-06 갱신**: 원래 51건(2016·2017·2019) 중 2016년치는 "2016년 말 기준" 전체
유효 명단을 확보해 실측으로 대체했다(`build_family_friendly_2016_regional_candidates.py`).
그 결과 2017년도 더 이상 "앞이 텅 빈 선행 결측"이 아니라 2016(실측)·2018(실측) 사이에 낀
"중간 결측"이 되어, 2019년과 같은 raking 방식(2016·2018 선형보간 후 전국 실측 합계로
비례 보정)으로 다시 계산했다 — 2018년 구성비를 그대로 빌려쓰던 이전 방식(`composition_ratio_2018`)
보다 실제 2016→2018 추세를 반영하므로 더 정확하다. 이전 51건(2016 포함) 산출물은
`reports/20260805_가족친화_지역별_51건_보조시나리오_후보.csv`에 비교용으로 남겨뒀다.

- 2017(중간 결측, 신규): 2016·2018 지역별 분자(공식 인증기업 수)로 선형보간한 뒤, 보간된
  17개 지역 분자의 합을 전국 실측 분자 합계(2,802)에 맞춰 비례 보정(raking)한다.
- 2019(중간 결측, 기존과 동일): 2018·2020 지역별 분자로 선형보간한 뒤 전국 실측 분자
  합계(3,833)로 raking한다.

두 경우 모두 분모(사업체수)는 해당 연도의 실제 지역별 사업체수를 그대로 쓰고, 분자만
추정한 뒤 최종 비율 = 추정 분자 / 지역별 분모 × 100으로 계산한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_family_friendly_2016_regional_candidates import (
    CANDIDATE_PATH as CANDIDATE_2016_PATH,
)
from scripts.build_family_friendly_candidates import (
    CONFIRMED_PATH,
    INDICATOR_ID,
    RAW_DIR,
    REGION_ORDER,
    add_qa,
    load_denominators,
    load_reference_counts,
)
from scripts.build_family_friendly_national_candidates import NATIONAL_CUMULATIVE_TOTALS

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260806_가족친화_지역별_34건_보조시나리오_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260806_가족친화_지역별_34건_보조시나리오_QA.csv"

RAKING_YEARS = (2017, 2019)
RAKING_METHOD = "raking"


def load_2016_confirmed_counts(candidate_2016_path: Path) -> pd.Series:
    """2016년 지역별 실측 복원 후보표에서 지역별 분자(공식 원자료 직접 복원)를 뽑는다."""
    confirmed = pd.read_csv(candidate_2016_path, encoding="utf-8-sig")
    if len(confirmed) != 17 or confirmed["지역"].duplicated().any():
        raise ValueError(f"2016년 지역별 분자가 17개 시도와 일치하지 않습니다: {len(confirmed)}건")
    return confirmed.set_index("지역")["공식_분자"].astype(int).reindex(REGION_ORDER)


def load_2018_reference_counts(raw_dir: Path, qa_records: list[dict[str, object]]) -> pd.Series:
    """2018년 지역별 공식 명단 재집계 분자(인증기업 수)를 불러온다."""
    counts = load_reference_counts(raw_dir, 2018)
    add_qa(qa_records, "2018 분자", "지역 수", 17, len(counts))
    add_qa(qa_records, "2018 분자", "합계", 3_328, int(counts.sum()))
    return counts.reindex(REGION_ORDER)


def load_2020_confirmed_counts(confirmed_path: Path) -> pd.Series:
    """37건 공식 관측 반영표에서 2020년 지역별 분자(공식 원자료 직접 복원)만 뽑는다."""
    confirmed = pd.read_csv(confirmed_path, encoding="utf-8-sig")
    selected = confirmed.loc[
        confirmed["지표_id"].eq(INDICATOR_ID)
        & confirmed["연도"].eq(2020)
        & confirmed["반영유형"].eq("공식 원자료 직접 복원")
    ]
    if len(selected) != 17 or selected["지역"].duplicated().any():
        raise ValueError(f"2020년 지역별 분자가 17개 시도와 일치하지 않습니다: {len(selected)}건")
    return selected.set_index("지역")["공식_분자"].astype(int).reindex(REGION_ORDER)


def build_raking_candidates(
    year: int,
    count_before: pd.Series,
    count_after: pd.Series,
    denominators: pd.DataFrame,
    qa_records: list[dict[str, object]],
    *,
    before_year: int,
    after_year: int,
) -> pd.DataFrame:
    """지역별 분자를 앞뒤 실측 연도로 선형보간한 뒤 전국 실측 합계로 raking한다."""
    interpolated = (count_before + count_after) / 2.0
    target_total = NATIONAL_CUMULATIVE_TOTALS[year]
    raked = interpolated * (target_total / interpolated.sum())
    denom = denominators[year]
    ratio = raked / denom * 100

    section = f"{year} raking"
    add_qa(qa_records, section, "raked 분자 합계", target_total, round(float(raked.sum()), 6))
    add_qa(qa_records, section, "지역 수", len(REGION_ORDER), len(raked))
    add_qa(qa_records, section, "음수·NA 없음", 0, int((raked < 0).sum() + raked.isna().sum()))

    return pd.DataFrame(
        {
            "지역": REGION_ORDER,
            "지표_id": INDICATOR_ID,
            "연도": year,
            "추정_분자": raked.reindex(REGION_ORDER).to_numpy(),
            "사업체수_분모": denom.reindex(REGION_ORDER).to_numpy(),
            "추정_비율": ratio.reindex(REGION_ORDER).to_numpy(),
            "방법": RAKING_METHOD,
            "근거": (
                f"{before_year}·{after_year} 지역 분자 선형보간 후 전국 실측 분자 합계"
                f"({target_total:,})로 raking"
            ),
        }
    )


def build_candidates(
    raw_dir: Path, confirmed_path: Path, candidate_2016_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_records: list[dict[str, object]] = []
    count_2016 = load_2016_confirmed_counts(candidate_2016_path)
    count_2018 = load_2018_reference_counts(raw_dir, qa_records)
    count_2020 = load_2020_confirmed_counts(confirmed_path)
    denominators = load_denominators(raw_dir, (2017, 2019), qa_records).reindex(REGION_ORDER)

    raking_2017 = build_raking_candidates(
        2017, count_2016, count_2018, denominators, qa_records, before_year=2016, after_year=2018
    )
    raking_2019 = build_raking_candidates(
        2019, count_2018, count_2020, denominators, qa_records, before_year=2018, after_year=2020
    )
    candidates = pd.concat([raking_2017, raking_2019], ignore_index=True).sort_values(
        ["연도", "지역"]
    )

    add_qa(qa_records, "전체", "총 행 수", 34, len(candidates))
    add_qa(
        qa_records, "전체", "고유 지역×연도", 34, len(candidates.drop_duplicates(["지역", "연도"]))
    )
    add_qa(
        qa_records,
        "전체",
        "음수·NA 비율 없음",
        0,
        int((candidates["추정_비율"] < 0).sum() + candidates["추정_비율"].isna().sum()),
    )

    qa = pd.DataFrame(qa_records)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"보조 시나리오 QA 실패: {failures.to_dict('records')}")
    return candidates.reset_index(drop=True), qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--confirmed", type=Path, default=CONFIRMED_PATH)
    parser.add_argument("--candidate-2016", type=Path, default=CANDIDATE_2016_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, qa = build_candidates(args.raw_dir, args.confirmed, args.candidate_2016)

    for path in (args.candidate_output, args.qa_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(f"보조 시나리오 후보: {len(candidates)}건 ({args.candidate_output})")
    print(f"QA: {len(qa)}개 항목 모두 PASS")


if __name__ == "__main__":
    main()
