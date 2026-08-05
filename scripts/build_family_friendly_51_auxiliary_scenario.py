"""가족친화 인증기업 비율 17개 시도 51건(2016·2017·2019)의 보조 시나리오 후보를 만든다.

`reports/methodology/20260805_가족친화_지역별_51건_보간_방법_제안.md`에서 제안한 방법을
실제로 계산한다. 이 51건은 여전히 공식 명단을 확보하지 못해 `pending_review`/
`block_imputation=True`로 본계열에서 차단돼 있으며, 이 스크립트는 그 상태를 바꾸지
않는다 — 결과는 `auxiliary_scenario_policy` 열에만 별도로 기록되는 민감도 비교용
후보값이다.

- 2019(중간 결측): 2018·2020 지역별 분자(공식 인증기업 수)로 선형보간한 뒤, 보간된
  17개 지역 분자의 합을 전국 실측 분자 합계(3,833)에 맞춰 비례 보정(raking)한다.
- 2016·2017(선행 결측): 2018년 지역별 분자 구성비(지역 분자 / 2018년 전국 분자)를
  구해서, 그 구성비를 해당 연도 전국 실측 분자 합계(1,828 / 2,802)에 곱한다.

두 경우 모두 분모(사업체수)는 해당 연도의 실제 지역별 사업체수를 그대로 쓰고, 분자만
추정한 뒤 최종 비율 = 추정 분자 / 지역별 분모 × 100으로 계산한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_family_friendly_candidates import (
    CONFIRMED_PATH,
    INDICATOR_ID,
    RAW_DIR,
    REGION_ORDER,
    add_qa,
    load_denominators,
    load_reference_counts,
)
from scripts.build_family_friendly_national_candidates import (
    DENOMINATOR_2016_SOURCE,
    NATIONAL_CUMULATIVE_TOTALS,
)
from scripts.build_family_friendly_candidates import load_denominator_source

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260805_가족친화_지역별_51건_보조시나리오_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260805_가족친화_지역별_51건_보조시나리오_QA.csv"

RAKING_YEAR = 2019
COMPOSITION_YEARS = (2016, 2017)
RAKING_METHOD = "raking"
COMPOSITION_METHOD = "composition_ratio_2018"


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


def load_province_denominators(
    raw_dir: Path, years: tuple[int, ...], qa_records: list[dict[str, object]]
) -> pd.DataFrame:
    """2016년은 전용 소스, 나머지는 기존 2017-2023 소스에서 지역별 분모를 모은다."""
    frames = []
    if 2016 in years:
        denom_2016 = load_denominator_source(
            raw_dir, DENOMINATOR_2016_SOURCE, (2016,), qa_records, "분모 2016(지역)"
        )
        frames.append(denom_2016.reindex(REGION_ORDER)[[2016]])
    other_years = tuple(year for year in years if year != 2016)
    if other_years:
        denom_rest = load_denominators(raw_dir, other_years, qa_records)
        frames.append(denom_rest.reindex(REGION_ORDER))
    combined = pd.concat(frames, axis="columns").reindex(columns=list(years))
    if combined.isna().any().any():
        raise ValueError(f"지역별 분모 연도 결합 실패: {years}")
    return combined.astype(int)


def build_raking_candidates(
    count_2018: pd.Series,
    count_2020: pd.Series,
    denominators: pd.DataFrame,
    qa_records: list[dict[str, object]],
) -> pd.DataFrame:
    """2019년 지역별 분자를 2018·2020 선형보간 후 전국 실측 합계로 raking한다."""
    interpolated = (count_2018 + count_2020) / 2.0
    target_total = NATIONAL_CUMULATIVE_TOTALS[RAKING_YEAR]
    raked = interpolated * (target_total / interpolated.sum())
    denom = denominators[RAKING_YEAR]
    ratio = raked / denom * 100

    add_qa(
        qa_records,
        "2019 raking",
        "raked 분자 합계",
        target_total,
        round(float(raked.sum()), 6),
    )
    add_qa(qa_records, "2019 raking", "지역 수", len(REGION_ORDER), len(raked))
    add_qa(
        qa_records, "2019 raking", "음수·NA 없음", 0, int((raked < 0).sum() + raked.isna().sum())
    )

    return pd.DataFrame(
        {
            "지역": REGION_ORDER,
            "지표_id": INDICATOR_ID,
            "연도": RAKING_YEAR,
            "추정_분자": raked.reindex(REGION_ORDER).to_numpy(),
            "사업체수_분모": denom.reindex(REGION_ORDER).to_numpy(),
            "추정_비율": ratio.reindex(REGION_ORDER).to_numpy(),
            "방법": RAKING_METHOD,
            "근거": (
                f"2018·2020 지역 분자 선형보간 후 전국 실측 분자 합계({target_total:,})로 raking"
            ),
        }
    )


def build_composition_candidates(
    count_2018: pd.Series,
    denominators: pd.DataFrame,
    years: tuple[int, ...],
    qa_records: list[dict[str, object]],
) -> pd.DataFrame:
    """2018년 지역 구성비를 해당 연도 전국 실측 분자 합계에 곱해 선행 결측을 추정한다."""
    share = count_2018 / count_2018.sum()
    add_qa(qa_records, "구성비", "2018 지역 구성비 합", 1.0, round(float(share.sum()), 12))

    frames = []
    for year in years:
        target_total = NATIONAL_CUMULATIVE_TOTALS[year]
        estimated = share * target_total
        denom = denominators[year]
        ratio = estimated / denom * 100

        add_qa(
            qa_records,
            f"{year} 구성비 고정",
            "추정 분자 합계",
            target_total,
            round(float(estimated.sum()), 6),
        )
        add_qa(qa_records, f"{year} 구성비 고정", "지역 수", len(REGION_ORDER), len(estimated))

        frames.append(
            pd.DataFrame(
                {
                    "지역": REGION_ORDER,
                    "지표_id": INDICATOR_ID,
                    "연도": year,
                    "추정_분자": estimated.reindex(REGION_ORDER).to_numpy(),
                    "사업체수_분모": denom.reindex(REGION_ORDER).to_numpy(),
                    "추정_비율": ratio.reindex(REGION_ORDER).to_numpy(),
                    "방법": COMPOSITION_METHOD,
                    "근거": (
                        f"2018년 지역 구성비 × {year}년 전국 실측 분자 합계({target_total:,})"
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_candidates(raw_dir: Path, confirmed_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_records: list[dict[str, object]] = []
    count_2018 = load_2018_reference_counts(raw_dir, qa_records)
    count_2020 = load_2020_confirmed_counts(confirmed_path)
    denominators = load_province_denominators(raw_dir, (2016, 2017, 2019, 2020), qa_records)

    raking = build_raking_candidates(count_2018, count_2020, denominators, qa_records)
    composition = build_composition_candidates(
        count_2018, denominators, COMPOSITION_YEARS, qa_records
    )
    candidates = pd.concat([composition, raking], ignore_index=True).sort_values(["연도", "지역"])

    add_qa(qa_records, "전체", "총 행 수", 51, len(candidates))
    add_qa(
        qa_records, "전체", "고유 지역×연도", 51, len(candidates.drop_duplicates(["지역", "연도"]))
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
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, qa = build_candidates(args.raw_dir, args.confirmed)

    for path in (args.candidate_output, args.qa_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(f"보조 시나리오 후보: {len(candidates)}건 ({args.candidate_output})")
    print(f"QA: {len(qa)}개 항목 모두 PASS")


if __name__ == "__main__":
    main()
