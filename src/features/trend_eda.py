"""구조환경지표·계획예산 지역×연도 추세 EDA 준비 유틸."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.evaluation.structural_validation import (
    normalize_nationwide_labels,
    require_columns,
    to_numeric_strict,
)


STRUCTURAL_ID_COLUMNS = ["지역", "대영역", "세부영역", "세부지표", "검증상태"]

STRUCTURAL_INDICATOR_DIRECTIONS = {
    "청년고용률": "높을수록 양호",
    "소득만족도": "높을수록 양호",
    "소득수준": "높을수록 양호",
    "소득 수준": "높을수록 양호",
    "보육시설 보급률": "높을수록 양호",
    "방과후 돌봄시설 보급도": "높을수록 양호",
    "사교육비 지출액": "낮을수록 양호",
    "문화기반시설 보급도": "높을수록 양호",
    "도시공원 보급도": "높을수록 양호",
    "여가생활 만족도": "높을수록 양호",
    "(대체)분만실 병상수 보급도": "높을수록 양호",
    "(대체)소아청소년과 전문인력 보급도": "높을수록 양호",
    "산후조리원 보급도": "높을수록 양호",
    "산후조리원 이용 요금": "낮을수록 양호",
    "어린이 교통사고 발생률": "낮을수록 양호",
    "사회 안전에 대한 인식": "높을수록 양호",
    "근로시간": "낮을수록 양호",
    "육아휴직 사용률": "높을수록 양호",
    "가족친화인증기업 비율": "높을수록 양호",
    "가족친화 인증기업 비율": "높을수록 양호",
    "결혼에 대한 인식": "높을수록 양호",
    "출산에 대한 인식": "높을수록 양호",
    "가사 분담에 대한 성평등 인식": "높을수록 양호",
    "가사분담에 대한 성평등 인식": "높을수록 양호",
    "청년층 정규직 근로자 비율": "높을수록 양호",
    "현재 주택가격": "낮을수록 양호",
    "전체 가구 자가점유비율": "높을수록 양호",
    "임차가구 연간 주거비 HCC": "낮을수록 양호",
    "가사노동 공동 참여도": "높을수록 양호",
    "돌봄노동 공동 참여도": "높을수록 양호",
    "사회경제적 지위에 대한 인식": "높을수록 양호",
}


def classify_basic_plan_period(year: int) -> str:
    """연도를 저출산·고령사회 기본계획 기간으로 구분한다."""

    if 2016 <= year <= 2020:
        return "제3차 기본계획(2016~2020)"
    if 2021 <= year <= 2024:
        return "제4차 기본계획(2021~2024)"
    return "분석기간 외"


def _add_iqr_outlier_flag(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    change_col: str,
    flag_col: str,
) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby(list(group_cols))[change_col]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    result[flag_col] = result[change_col].notna() & (
        result[change_col].lt(q1 - 1.5 * iqr) | result[change_col].gt(q3 + 1.5 * iqr)
    )
    return result


def reshape_structural_indicators(
    df: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    years: Sequence[int] = tuple(range(2016, 2026)),
) -> pd.DataFrame:
    """구조환경지표 wide 표를 실측·결측 플래그가 있는 long 표로 변환한다."""

    year_cols = [str(year) for year in years]
    require_columns(
        df,
        [*STRUCTURAL_ID_COLUMNS, *year_cols],
        source_name="구조환경지표 검증본",
    )
    normalized = normalize_nationwide_labels(df)

    for indicator, group in normalized.groupby("세부지표"):
        actual_regions = set(group.loc[~group["지역"].eq("전국"), "지역"])
        missing_regions = sorted(set(expected_regions) - actual_regions)
        unexpected_regions = sorted(actual_regions - set(expected_regions))
        if missing_regions or unexpected_regions:
            raise ValueError(
                f"{indicator} 17개 시도 구성 불일치: "
                f"누락={missing_regions}, 예상외={unexpected_regions}"
            )

    long = normalized.melt(
        id_vars=STRUCTURAL_ID_COLUMNS,
        value_vars=year_cols,
        var_name="연도",
        value_name="측정값",
    )
    long["연도"] = long["연도"].astype(int)
    long["측정값"] = to_numeric_strict(long["측정값"])
    long["실측여부"] = long["측정값"].notna()
    long["결측상태"] = long["실측여부"].map({True: "실측", False: "원자료 결측"})
    long["관측치용도"] = long["지역"].eq("전국").map({True: "전국 참고", False: "17개 시도 분석"})
    long["기본계획기간"] = long["연도"].map(classify_basic_plan_period)

    duplicated = long.duplicated(["지역", "연도", "세부지표"], keep=False)
    if duplicated.any():
        keys = long.loc[duplicated, ["지역", "연도", "세부지표"]].drop_duplicates()
        raise ValueError(f"구조환경지표 지역×연도 키 중복: {keys.to_dict('records')}")

    long = long.sort_values(["세부지표", "지역", "연도"]).reset_index(drop=True)
    region_rows = long.loc[~long["지역"].eq("전국")].copy()
    grouped = region_rows.groupby(["세부지표", "지역"], sort=False)
    previous_year = grouped["연도"].shift(1)
    previous_value = grouped["측정값"].shift(1)
    consecutive = region_rows["연도"].sub(previous_year).eq(1)
    region_rows["전년대비변화"] = region_rows["측정값"].sub(previous_value).where(consecutive)
    region_rows = _add_iqr_outlier_flag(
        region_rows,
        group_cols=["세부지표", "연도"],
        change_col="전년대비변화",
        flag_col="급등락후보",
    )

    return long.merge(
        region_rows[["지역", "연도", "세부지표", "전년대비변화", "급등락후보"]],
        on=["지역", "연도", "세부지표"],
        how="left",
        validate="one_to_one",
    )


def prepare_budget_trends(
    budget_panel: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int] = tuple(range(2016, 2025)),
) -> pd.DataFrame:
    """#53 기초패널에 기본계획 기간·증감률·급등락 후보를 추가한다."""

    required = {"지역", "연도", "당해계획예산_백만원", "원자료_누락주의"}
    missing = sorted(required - set(budget_panel.columns))
    if missing:
        raise ValueError(f"계획예산 기초패널 필수 컬럼 누락: {missing}")

    result = budget_panel.copy()
    if result["지역"].eq("전국").any():
        raise ValueError("계획예산 지역 패널에 전국 행이 포함되어 있습니다.")

    duplicated = result.duplicated(["지역", "연도"], keep=False)
    if duplicated.any():
        keys = result.loc[duplicated, ["지역", "연도"]].drop_duplicates()
        raise ValueError(f"계획예산 지역×연도 키 중복: {keys.to_dict('records')}")

    actual_keys = set(result[["지역", "연도"]].itertuples(index=False, name=None))
    expected_keys = {(region, year) for region in expected_regions for year in expected_years}
    if actual_keys != expected_keys:
        raise ValueError(
            "계획예산 지역×연도 조합 불일치: "
            f"누락={sorted(expected_keys - actual_keys)[:10]}, "
            f"예상외={sorted(actual_keys - expected_keys)[:10]}"
        )

    result = result.sort_values(["지역", "연도"]).reset_index(drop=True)
    result["기본계획기간"] = result["연도"].map(classify_basic_plan_period)
    grouped = result.groupby("지역", sort=False)
    previous_year = grouped["연도"].shift(1)
    previous_budget = grouped["당해계획예산_백만원"].shift(1)
    consecutive = result["연도"].sub(previous_year).eq(1)
    result["전년대비증감률_pct"] = (
        result["당해계획예산_백만원"].div(previous_budget).sub(1).mul(100).where(consecutive)
    )
    return _add_iqr_outlier_flag(
        result,
        group_cols=["연도"],
        change_col="전년대비증감률_pct",
        flag_col="급등락후보",
    )


def _representative_years(indicator_df: pd.DataFrame) -> tuple[int, int]:
    coverage = indicator_df.groupby("연도")["측정값"].count()
    fullest_years = coverage.index[coverage.eq(coverage.max())]
    return int(fullest_years.min()), int(fullest_years.max())


def _directional_result(
    base_value: float | None,
    reference_value: float | None,
    direction: str,
) -> str:
    if pd.isna(base_value) or pd.isna(reference_value):
        return "비교 불가"
    if direction == "3점에 가까울수록 양호":
        change = abs(reference_value - 3) - abs(base_value - 3)
    elif direction == "낮을수록 양호":
        change = base_value - reference_value
    else:
        change = reference_value - base_value
    if abs(change) < 1e-12:
        return "변화 없음"
    return "개선" if change > 0 else "악화"


def build_structural_region_summary(
    structural_long: pd.DataFrame,
    *,
    region_order: Sequence[str],
    directions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """지표별 대표 최초·최신 연도의 17개 시도 결과와 방향성 순위를 요약한다."""

    required = {
        "지역",
        "연도",
        "대영역",
        "세부영역",
        "세부지표",
        "측정값",
        "실측여부",
        "급등락후보",
    }
    missing = sorted(required - set(structural_long.columns))
    if missing:
        raise ValueError(f"구조환경지표 지역 요약 필수 컬럼 누락: {missing}")

    direction_map = directions or STRUCTURAL_INDICATOR_DIRECTIONS
    indicators = structural_long["세부지표"].drop_duplicates().tolist()
    missing_directions = sorted(set(indicators) - set(direction_map))
    if missing_directions:
        raise ValueError(f"방향성 정의가 없는 구조환경지표: {missing_directions}")

    region_data = structural_long.loc[structural_long["지역"].isin(region_order)].copy()
    summaries = []
    for indicator, indicator_df in region_data.groupby("세부지표", sort=False):
        base_year, reference_year = _representative_years(indicator_df)
        direction = direction_map[indicator]
        metadata = indicator_df[["대영역", "세부영역"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"{indicator} 대영역·세부영역 값이 하나가 아닙니다.")

        values = indicator_df.pivot(index="지역", columns="연도", values="측정값")
        base_values = values.get(base_year, pd.Series(dtype="float64"))
        reference_values = values.get(reference_year, pd.Series(dtype="float64"))
        if direction == "3점에 가까울수록 양호":
            ranking_values = reference_values.sub(3).abs()
            ranks = ranking_values.rank(method="min", ascending=True)
        else:
            ranks = reference_values.rank(
                method="min",
                ascending=direction == "낮을수록 양호",
            )

        for region in region_order:
            region_rows = indicator_df.loc[indicator_df["지역"].eq(region)]
            base_value = base_values.get(region, pd.NA)
            reference_value = reference_values.get(region, pd.NA)
            outlier_years = sorted(
                region_rows.loc[region_rows["급등락후보"].fillna(False), "연도"].tolist()
            )
            summaries.append(
                {
                    "대영역": metadata.iloc[0]["대영역"],
                    "세부영역": metadata.iloc[0]["세부영역"],
                    "세부지표": indicator,
                    "방향성": direction,
                    "비교시작연도": base_year,
                    "비교기준연도": reference_year,
                    "지역": region,
                    "시작값": base_value,
                    "기준값": reference_value,
                    "변화량": (
                        reference_value - base_value
                        if pd.notna(base_value) and pd.notna(reference_value)
                        else pd.NA
                    ),
                    "방향성기준결과": _directional_result(
                        base_value,
                        reference_value,
                        direction,
                    ),
                    "기준연도순위": (
                        int(ranks.get(region)) if pd.notna(ranks.get(region, pd.NA)) else pd.NA
                    ),
                    "실측연도수": int(region_rows["실측여부"].sum()),
                    "결측연도수": int((~region_rows["실측여부"]).sum()),
                    "급등락후보연도": ", ".join(map(str, outlier_years)) or "-",
                }
            )

    return pd.DataFrame(summaries)


def _format_report_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def render_structural_region_report(summary: pd.DataFrame) -> str:
    """17개 시도 구조환경지표 요약표를 Markdown 상세 보고서로 렌더링한다."""

    lines = [
        "# 구조환경지표별 17개 시도 상세 결과",
        "",
        "이 부록은 검증된 28개 구조환경지표의 지역별 결과를 비교한다. "
        "각 지표에서 17개 시도 관측 수가 가장 많은 연도 중 최초·최신 연도를 사용했다.",
        "",
        "- `개선·악화`는 지표의 방향성에 따른 기술적 변화이며 정책 효과를 뜻하지 않는다.",
        "- 순위는 비교기준연도의 관측 가능한 시도만 대상으로 하며 동순위는 최소순위 방식이다.",
        "- 급등락 후보는 전년 대비 변화의 IQR 기준으로 표시한 검토 대상이며 오류 판정이 아니다.",
        "- 조사 주기가 비연간인 지표는 결측연도 수가 많을 수 있다.",
        "",
    ]
    for indicator, group in summary.groupby("세부지표", sort=False):
        first = group.iloc[0]
        observed = group.dropna(subset=["기준값"]).sort_values("기준연도순위")
        top_regions = ", ".join(
            f"{row.지역}({_format_report_value(row.기준값)})"
            for row in observed.head(3).itertuples()
        )
        bottom_regions = ", ".join(
            f"{row.지역}({_format_report_value(row.기준값)})"
            for row in observed.tail(3).sort_values("기준연도순위", ascending=False).itertuples()
        )
        lines.extend(
            [
                f"## {indicator}",
                "",
                f"- 영역: {first['대영역']} > {first['세부영역']}",
                f"- 방향성: {first['방향성']}",
                f"- 비교: {first['비교시작연도']}년 → {first['비교기준연도']}년",
                f"- 기준연도 양호 상위: {top_regions or '-'}",
                f"- 기준연도 하위: {bottom_regions or '-'}",
                "",
                "| 지역 | 시작값 | 기준값 | 변화량 | 방향성 기준 | 순위 | 실측/결측 연도 | 급등락 후보 연도 |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.지역,
                        _format_report_value(row.시작값),
                        _format_report_value(row.기준값),
                        _format_report_value(row.변화량),
                        row.방향성기준결과,
                        _format_report_value(row.기준연도순위),
                        f"{row.실측연도수}/{row.결측연도수}",
                        row.급등락후보연도,
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "STRUCTURAL_INDICATOR_DIRECTIONS",
    "build_structural_region_summary",
    "classify_basic_plan_period",
    "prepare_budget_trends",
    "render_structural_region_report",
    "reshape_structural_indicators",
]
