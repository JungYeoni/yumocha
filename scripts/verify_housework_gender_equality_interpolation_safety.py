"""가사분담 성평등 인식(housework_gender_equality)의 최종값 직접 선형보간이 안전한지 검증한다.

원산식은 `1 - |지역별 평균응답값 - 3| / 2`로 비선형(절댓값) 변환이다. 두 관측연도의
평균응답값이 모두 3보다 작은 경우(또는 모두 3보다 큰 경우) 이 변환은 그 구간에서
순수 선형함수(`(x-1)/2` 또는 `(5-x)/2`)와 같아지므로, 최종값을 직접 선형보간한 결과와
평균응답값을 먼저 보간한 뒤 변환한 결과가 정확히 같다. 이 스크립트는 원자료에서
평균응답값을 재현해 18개 지역 × 5개 관측연도(2016·2018·2020·2022·2024) 전부가
한쪽에서만 관측되는지(3을 넘나들지 않는지) 확인한다.

3을 넘나드는 관측연도 쌍이 하나도 없으면, `structural_missing.py`의 일반 선형보간을
이 지표의 최종값에 그대로 적용해도 방법론적으로 정확하다 — 별도의 구성요소 보간
로직을 새로 만들 필요가 없다는 뜻이다(재발명 금지).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.structural_validation import load_region_mapping, weighted_response_mean

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "지표별_측정값" / "4-2.3. 가사분담에 대한 성평등 인식.xlsx"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "구조환경지표_28개_보간전_기준패널.csv"
REPORT_PATH = (
    REPO_ROOT / "reports" / "methodology" / "20260805_가사분담_성평등_인식_직접보간_안전성_검증.md"
)

INDICATOR_ID = "housework_gender_equality"
OBSERVED_YEARS = (2016, 2018, 2020, 2022, 2024)
NEUTRAL_MIDPOINT = 3.0
HW_SCORES = {
    "아내가 전적으로 책임": 1,
    "아내가 주로 하지만 남편도 분담": 2,
    "부부가 공평하게 분담": 3,
    "남편이 주로 하지만 아내도 분담": 4,
    "남편이 전적으로 책임": 5,
}


def load_survey_block(
    raw: pd.DataFrame, start_row: int, end_row: int, years_row: pd.Series, cat_row: pd.Series
) -> pd.DataFrame:
    """원본 시트의 "원데이터(%)" 블록 하나를 지역×연도-카테고리 표로 만든다."""
    block = raw.iloc[start_row:end_row].reset_index(drop=True)
    block.columns = [
        f"{years_row[i]}_{cat_row[i]}" if pd.notna(cat_row[i]) else years_row[i]
        for i in range(len(cat_row))
    ]
    block = block.rename(columns={block.columns[0]: "지역", block.columns[1]: "항목"})
    block["지역"] = block["지역"].astype(str).str.strip()
    return block.set_index("지역")


def compute_weighted_means(raw_path: Path, expected_regions: list[str]) -> pd.DataFrame:
    """원자료(%)에서 연도별·지역별 평균응답값(1~5점)을 재현한다."""
    raw = pd.read_excel(raw_path, sheet_name="2016,2018,2020,2022,2024_가사_분담에", header=None)
    years_row = raw.iloc[0]
    cat_row = raw.iloc[1]
    block = load_survey_block(raw, 2, 20, years_row, cat_row)
    block = block.loc[block.index.isin(expected_regions)]

    means = {}
    for year in OBSERVED_YEARS:
        renamed = block.rename(columns={f"{year}_{cat}": cat for cat in HW_SCORES})
        means[year] = weighted_response_mean(
            renamed, scores=HW_SCORES, expected_regions=expected_regions
        )
    return pd.DataFrame(means).reindex(expected_regions)


def load_panel_official_values(panel_path: Path, expected_regions: list[str]) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, encoding="utf-8-sig")
    subset = panel.loc[panel["지표_id"].eq(INDICATOR_ID)]
    pivot = subset.pivot(index="지역", columns="연도", values="측정값")
    return pivot.reindex(index=expected_regions, columns=list(OBSERVED_YEARS))


def check_interpolation_safety(
    means: pd.DataFrame, official: pd.DataFrame, expected_regions: list[str]
) -> pd.DataFrame:
    """평균응답값·패널 실측값이 주어졌을 때 직접 선형보간이 안전한지 검증한다.

    파일 I/O 없이 순수 계산만 하므로 합성 데이터로 단위 테스트할 수 있다.
    """
    reproduced = 1 - (means - NEUTRAL_MIDPOINT).abs() / 2

    checks: list[dict[str, object]] = []

    def check(item: str, expected: object, actual: object) -> None:
        checks.append(
            {
                "검사항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "PASS" if expected == actual else "FAIL",
            }
        )

    check("지역 수", len(expected_regions), len(means))
    check("관측연도 수", len(OBSERVED_YEARS), means.shape[1])
    check("평균응답값 결측 없음", 0, int(means.isna().sum().sum()))

    diff = (reproduced - official).abs()
    check("원자료 재현값 대 패널 실측값 최대오차 <= 1e-6", True, bool((diff <= 1e-6).all().all()))

    crossings = (means - NEUTRAL_MIDPOINT >= 0).to_numpy()
    check(
        "평균응답값이 3 이상인 관측연도·지역 수(교차 위험)",
        0,
        int(crossings.sum()),
    )

    qa = pd.DataFrame(checks)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"직접 선형보간 안전성 검증 실패: {failures.to_dict('records')}")
    return qa


def verify(raw_path: Path, panel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_regions, _ = load_region_mapping(REPO_ROOT)
    means = compute_weighted_means(raw_path, expected_regions)
    official = load_panel_official_values(panel_path, expected_regions)
    qa = check_interpolation_safety(means, official, expected_regions)
    return means, qa


def render_report(means: pd.DataFrame, qa: pd.DataFrame) -> str:
    lines = [
        "# 가사분담 성평등 인식 최종값 직접 선형보간 안전성 검증",
        "",
        "## 검증 범위",
        "",
        f"- 데이터셋 버전·기간: `{RAW_PATH.name}`, 관측연도 {OBSERVED_YEARS[0]}–{OBSERVED_YEARS[-1]}"
        f"({', '.join(str(y) for y in OBSERVED_YEARS)}), 18개 지역",
        "- 분할 전략: 해당 없음 — 예측 모델 학습이 아니라 원자료 재현값과 패널 실측값을 "
        "직접 대조하는 결정론적 검증이다",
        "- 평가 지표·CV fold 수: 해당 없음(위와 같은 이유) — 대신 `최대오차 <= 1e-6`(원자료 "
        "재현값 대 패널 실측값)과 `평균응답값이 3점을 넘나드는 지역·연도 수`(교차 위험, "
        "0건이어야 안전)를 판정 기준으로 쓴다",
        "- 핵심 가정: 원산식 `1-|평균응답값-3|/2`이 관측구간 전체에서 3점(중간)을 넘나들지 "
        "않으면 순수 선형함수와 동치라는 수학적 사실에 의존한다",
        "- 방법론적 제약·한계: 이 결론은 관측된 5개년·18개 지역에만 적용된다. 향후 새 관측연도가 "
        "추가돼 평균응답값이 3점을 넘나들면(공평 분담 쪽에서 반대쪽으로 역전되면) 이 동치가 "
        "깨지므로, 새 연도가 추가될 때마다 이 검증을 재실행해야 한다",
        "",
        "## 결론",
        "",
        (
            "18개 지역 × 5개 관측연도(2016·2018·2020·2022·2024) 전부에서 평균응답값이 "
            f"{NEUTRAL_MIDPOINT}점(중간, 공평 분담) 미만이었다 — 3점을 넘나드는 경우가 "
            "없으므로 `1 - |평균응답값-3| / 2` 변환은 관측구간 전체에서 순수 선형함수와 "
            "같다. 따라서 최종값을 직접 선형보간해도 평균응답값을 먼저 보간한 뒤 변환한 "
            "결과와 정확히 같다 — 구성요소 보간을 별도로 구현할 필요가 없다."
        ),
        "",
        "## 평균응답값(1~5점, 재현값)",
        "",
        "```",
        means.round(3).to_string(),
        "```",
        "",
        "## QA",
        "",
        "```",
        qa.to_string(index=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    means, qa = verify(args.raw_path, args.panel)
    report = render_report(means, qa)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")
    print(f"검증 리포트: {args.report_output}")
    print(f"QA {len(qa)}개 항목 모두 PASS")


if __name__ == "__main__":
    main()
