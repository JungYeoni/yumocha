"""#80 기준 패널의 결측 정책 매핑과 QA 산출물을 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

KEY_COLUMNS = ["지역", "지표_id", "연도"]
REQUIRED_PANEL_COLUMNS = [*KEY_COLUMNS, "지표명", "측정값", "원본행존재"]
MAPPING_COLUMNS = [
    *KEY_COLUMNS,
    "지표명",
    "missing_position",
    "nearest_observed_year",
    "distance_from_observation",
    "carry_direction",
    "consecutive_missing_years",
    "long_range_backcast",
    "sensitivity_required",
    "policy_risk_level",
    "missing_cause",
    "cause_status",
    "imputation_policy",
    "block_imputation",
    "analysis_included_after_imputation",
    "original_row_exists",
    "auxiliary_scenario_policy",
    "auxiliary_boundary_year",
    "rule_id",
    "evidence",
    "policy_note",
]


def load_policy_config(path: Path) -> dict[str, Any]:
    """YAML 정책 규칙을 읽고 기본 구조를 검증한다."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    required = {"panel", "allowed_values", "expected_counts", "rules"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"정책 설정 필수 항목이 없습니다: {sorted(missing)}")
    if not isinstance(config["rules"], list) or not config["rules"]:
        raise ValueError("정책 규칙은 하나 이상이어야 합니다.")

    rule_ids = [rule.get("rule_id") for rule in config["rules"]]
    if any(not rule_id for rule_id in rule_ids):
        raise ValueError("모든 정책 규칙에 rule_id가 필요합니다.")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("정책 규칙의 rule_id가 중복됩니다.")

    allowed = config["allowed_values"]
    for rule in config["rules"]:
        for field in ("missing_cause", "cause_status", "imputation_policy"):
            if rule.get(field) not in allowed[field]:
                raise ValueError(f"{rule['rule_id']}: 허용되지 않은 {field}={rule.get(field)!r}")
        for field in ("block_imputation", "analysis_included"):
            if not isinstance(rule.get(field), bool):
                raise ValueError(f"{rule['rule_id']}: {field}는 boolean이어야 합니다.")
        auxiliary = rule.get("auxiliary_scenario_policy")
        if auxiliary is not None and auxiliary != "auxiliary_boundary_interpolation":
            raise ValueError(f"{rule['rule_id']}: 허용되지 않은 보조 시나리오입니다.")

    return config


def normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """정책 매핑용 패널 복사본의 키와 자료형을 검증한다."""
    missing_columns = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열이 없습니다: {sorted(missing_columns)}")

    normalized = panel.copy()
    normalized["연도"] = pd.to_numeric(normalized["연도"], errors="raise").astype(int)
    if normalized.duplicated(KEY_COLUMNS).any():
        duplicates = normalized.loc[
            normalized.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS
        ].drop_duplicates()
        raise ValueError(f"패널 키가 중복됩니다: {duplicates.to_dict('records')}")
    if not pd.api.types.is_bool_dtype(normalized["원본행존재"]):
        raise ValueError("원본행존재 열은 boolean이어야 합니다.")
    return normalized


def _add_missing_positions(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    output["_missing_position"] = ""
    output["_nearest_observed_year"] = pd.NA
    output["_distance_from_observation"] = pd.NA
    output["_carry_direction"] = ""
    output["_consecutive_missing_years"] = pd.NA
    for _, group in output.groupby(["지표_id", "지역"], sort=False):
        observed = group["측정값"].notna()
        missing = ~observed
        if not observed.any():
            output.loc[group.index[missing], "_missing_position"] = "all_period_unobserved"
            continue

        first_year = int(group.loc[observed, "연도"].min())
        last_year = int(group.loc[observed, "연도"].max())
        leading = missing & group["연도"].lt(first_year)
        trailing = missing & group["연도"].gt(last_year)
        leading_index = group.index[leading]
        trailing_index = group.index[trailing]

        output.loc[leading_index, "_missing_position"] = "leading_missing"
        output.loc[leading_index, "_nearest_observed_year"] = first_year
        output.loc[leading_index, "_distance_from_observation"] = (
            first_year - group.loc[leading, "연도"]
        )
        output.loc[leading_index, "_carry_direction"] = "backward"
        output.loc[leading_index, "_consecutive_missing_years"] = int(leading.sum())

        output.loc[trailing_index, "_missing_position"] = "trailing_missing"
        output.loc[trailing_index, "_nearest_observed_year"] = last_year
        output.loc[trailing_index, "_distance_from_observation"] = (
            group.loc[trailing, "연도"] - last_year
        )
        output.loc[trailing_index, "_carry_direction"] = "forward"
        output.loc[trailing_index, "_consecutive_missing_years"] = int(trailing.sum())

        interior = missing & group["연도"].between(first_year, last_year)
        output.loc[group.index[interior], "_missing_position"] = "intermediate_missing"
    return output


def _rule_mask(panel: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    mask = panel["지표_id"].isin(rule["indicator_ids"]) & panel["연도"].isin(rule["years"])
    regions = rule["regions"]
    if regions != "all":
        mask &= panel["지역"].isin(regions)
    return mask


def build_policy_mapping(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """정책 규칙을 결측 지역·지표·연도 키 단위로 전개한다."""
    normalized = _add_missing_positions(normalize_panel(panel))
    frames: list[pd.DataFrame] = []

    for rule in config["rules"]:
        selected = normalized.loc[_rule_mask(normalized, rule)].copy()
        if selected.empty:
            raise ValueError(f"{rule['rule_id']}: 선택된 패널 행이 없습니다.")
        if selected["측정값"].notna().any():
            observed_keys = selected.loc[selected["측정값"].notna(), KEY_COLUMNS]
            raise ValueError(
                f"{rule['rule_id']}: 실측 행을 선택했습니다: {observed_keys.to_dict('records')}"
            )

        mapped = selected[
            [
                *KEY_COLUMNS,
                "지표명",
                "_missing_position",
                "_nearest_observed_year",
                "_distance_from_observation",
                "_carry_direction",
                "_consecutive_missing_years",
                "원본행존재",
            ]
        ].copy()
        mapped = mapped.rename(
            columns={
                "_missing_position": "missing_position",
                "_nearest_observed_year": "nearest_observed_year",
                "_distance_from_observation": "distance_from_observation",
                "_carry_direction": "carry_direction",
                "_consecutive_missing_years": "consecutive_missing_years",
                "원본행존재": "original_row_exists",
            }
        )
        for field in (
            "missing_cause",
            "cause_status",
            "imputation_policy",
            "block_imputation",
        ):
            mapped[field] = rule[field]
        mapped["analysis_included_after_imputation"] = rule["analysis_included"]

        boundary = mapped["imputation_policy"].eq("boundary_carry")
        mapped.loc[~boundary, "nearest_observed_year"] = pd.NA
        mapped.loc[~boundary, "distance_from_observation"] = pd.NA
        mapped.loc[~boundary, "carry_direction"] = ""
        mapped.loc[~boundary, "consecutive_missing_years"] = pd.NA
        mapped["long_range_backcast"] = (
            boundary
            & mapped["carry_direction"].eq("backward")
            & mapped["consecutive_missing_years"].ge(2)
        )
        mapped["sensitivity_required"] = boundary
        mapped["policy_risk_level"] = "not_applicable"
        mapped.loc[boundary, "policy_risk_level"] = "medium"
        mapped.loc[mapped["long_range_backcast"], "policy_risk_level"] = "high"
        mapped["auxiliary_scenario_policy"] = rule.get("auxiliary_scenario_policy", "")
        mapped["auxiliary_boundary_year"] = rule.get("auxiliary_boundary_year", pd.NA)
        mapped["rule_id"] = rule["rule_id"]
        mapped["evidence"] = rule["evidence"]
        mapped["policy_note"] = rule["note"]
        frames.append(mapped[MAPPING_COLUMNS])

    mapping = pd.concat(frames, ignore_index=True)
    for field in (
        "nearest_observed_year",
        "distance_from_observation",
        "consecutive_missing_years",
        "auxiliary_boundary_year",
    ):
        mapping[field] = mapping[field].astype("Int64")
    duplicates = mapping.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        duplicate_rules = mapping.loc[duplicates, [*KEY_COLUMNS, "rule_id"]]
        raise ValueError(
            f"여러 규칙에 중복 매핑된 키가 있습니다: {duplicate_rules.to_dict('records')}"
        )

    missing_keys = pd.MultiIndex.from_frame(
        normalized.loc[normalized["측정값"].isna(), KEY_COLUMNS]
    )
    mapping_keys = pd.MultiIndex.from_frame(mapping[KEY_COLUMNS])
    unmapped = missing_keys.difference(mapping_keys)
    extra = mapping_keys.difference(missing_keys)
    if len(unmapped) or len(extra):
        raise ValueError(
            f"결측 정책 키 불일치: 미매핑={list(unmapped)[:10]}, 초과={list(extra)[:10]}"
        )

    return mapping.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _count_dict(series: pd.Series) -> dict[Any, int]:
    return {key: int(value) for key, value in series.value_counts(dropna=False).items()}


def validate_policy_mapping(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """전수 매핑, 미확정 보호, 원본 불변성과 개념 독립성을 검증한다."""
    normalized = normalize_panel(panel)
    checks: list[dict[str, Any]] = []

    def check(item: str, expected: Any, actual: Any, note: str = "") -> None:
        checks.append(
            {
                "검사항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "PASS" if actual == expected else "FAIL",
                "비고": note,
            }
        )

    panel_config = config["panel"]
    missing = normalized["측정값"].isna()
    observed = ~missing
    check("패널 행 수", panel_config["expected_rows"], len(normalized))
    check(
        "고유 키 수",
        panel_config["expected_rows"],
        normalized[KEY_COLUMNS].drop_duplicates().shape[0],
    )
    check("실측 행 수", panel_config["expected_observed"], int(observed.sum()))
    check("결측 행 수", panel_config["expected_missing"], int(missing.sum()))
    check("정책 매핑 행 수", panel_config["expected_missing"], len(mapping))
    check("정책 매핑 중복 키", 0, int(mapping.duplicated(KEY_COLUMNS).sum()))

    missing_keys = pd.MultiIndex.from_frame(normalized.loc[missing, KEY_COLUMNS])
    observed_keys = pd.MultiIndex.from_frame(normalized.loc[observed, KEY_COLUMNS])
    mapping_keys = pd.MultiIndex.from_frame(mapping[KEY_COLUMNS])
    check("미매핑 결측 키", 0, len(missing_keys.difference(mapping_keys)))
    check("결측이 아닌 매핑 키", 0, len(mapping_keys.intersection(observed_keys)))

    allowed = config["allowed_values"]
    for field in ("missing_cause", "cause_status", "imputation_policy"):
        invalid = set(mapping[field].dropna()) - set(allowed[field])
        check(f"{field} 허용값 위반", 0, len(invalid), ", ".join(sorted(invalid)))

    expected_counts = config["expected_counts"]
    distribution_fields = (
        ("missing_cause", "missing_cause"),
        ("cause_status", "cause_status"),
        ("imputation_policy", "imputation_policy"),
        ("block_imputation", "block_imputation"),
        ("analysis_included", "analysis_included_after_imputation"),
        ("missing_position", "missing_position"),
    )
    for expected_field, mapping_field in distribution_fields:
        actual_counts = _count_dict(mapping[mapping_field])
        for value, expected in expected_counts[expected_field].items():
            check(f"{mapping_field}={value}", expected, actual_counts.get(value, 0))

    unresolved = mapping["cause_status"].eq("unresolved")
    check(
        "미확정 결측 수",
        expected_counts["cause_status"]["unresolved"],
        int(unresolved.sum()),
    )
    check(
        "미확정 결측의 비-pending 정책",
        0,
        int((unresolved & mapping["imputation_policy"].ne("pending_review")).sum()),
    )
    check(
        "미확정 결측의 대체 차단 해제",
        0,
        int((unresolved & ~mapping["block_imputation"]).sum()),
    )
    check(
        "미확정 결측의 분석 포함",
        0,
        int((unresolved & mapping["analysis_included_after_imputation"]).sum()),
    )

    family_friendly = mapping["지표_id"].eq("family_friendly_certification_rate")
    youth_regular_national = mapping["지표_id"].eq("youth_regular_employment_rate") & mapping[
        "지역"
    ].eq("전국")
    check(
        "가족친화 인증기업 비율 pending_review",
        expected_counts["missing_cause"]["원인 미확정"],
        int((family_friendly & unresolved).sum()),
    )
    check("정규직 비율 전국 pending_review", 9, int((youth_regular_national & unresolved).sum()))

    nonlinear_housework = mapping["지표_id"].eq("housework_gender_equality")
    check(
        "비선형 가사분담 지표 pending_review",
        72,
        int(
            (
                nonlinear_housework
                & mapping["imputation_policy"].eq("pending_review")
                & mapping["block_imputation"]
            ).sum()
        ),
    )

    blocked_policy = mapping["imputation_policy"].isin(["keep_missing", "pending_review"])
    check(
        "정책과 block_imputation 불일치",
        0,
        int(mapping["block_imputation"].ne(blocked_policy).sum()),
    )
    survey_block_values = set(
        mapping.loc[mapping["missing_cause"].eq("조사 비실시 연도"), "block_imputation"]
    )
    check(
        "조사 비실시 원인의 block 값 종류",
        2,
        len(survey_block_values),
        "동일 missing_cause 안에 차단·비차단 정책이 모두 있어야 한다.",
    )

    auxiliary = mapping["auxiliary_scenario_policy"].eq("auxiliary_boundary_interpolation")
    check(
        "보조 경계 시나리오 행 수", expected_counts["auxiliary_scenario_rows"], int(auxiliary.sum())
    )
    check(
        "보조 경계 시나리오의 본계열 정책 위반",
        0,
        int((auxiliary & mapping["imputation_policy"].ne("boundary_carry")).sum()),
    )
    check(
        "보조 경계 시나리오의 본계열 정책 중복 적용",
        0,
        int(
            (auxiliary & mapping["imputation_policy"].eq("auxiliary_boundary_interpolation")).sum()
        ),
        "보조 시나리오는 별도 열에만 기록한다.",
    )
    check(
        "완전격자 원본 미존재 행 수",
        expected_counts["original_grid_rows"],
        int((~mapping["original_row_exists"]).sum()),
        "원본행존재는 missing_cause와 별도 관리한다.",
    )

    boundary = mapping["imputation_policy"].eq("boundary_carry")
    boundary_mapping = mapping.loc[boundary]
    boundary_expected = expected_counts["boundary_review"]
    check("boundary_carry 세분 검토 행 수", boundary_expected["total"], len(boundary_mapping))
    for field in (
        "nearest_observed_year",
        "distance_from_observation",
        "carry_direction",
        "consecutive_missing_years",
        "policy_risk_level",
    ):
        check(f"boundary_carry {field} 결측", 0, int(boundary_mapping[field].isna().sum()))
    check(
        "boundary_carry 허용 위치 위반",
        0,
        int(
            (
                ~boundary_mapping["missing_position"].isin(["leading_missing", "trailing_missing"])
            ).sum()
        ),
    )
    expected_distance = (boundary_mapping["연도"] - boundary_mapping["nearest_observed_year"]).abs()
    check(
        "boundary_carry 관측거리 불일치",
        0,
        int(boundary_mapping["distance_from_observation"].ne(expected_distance).sum()),
    )
    expected_direction = boundary_mapping["missing_position"].map(
        {"leading_missing": "backward", "trailing_missing": "forward"}
    )
    check(
        "boundary_carry 방향 불일치",
        0,
        int(boundary_mapping["carry_direction"].ne(expected_direction).sum()),
    )
    check(
        "boundary_carry 연속 결측연도 비양수",
        0,
        int(boundary_mapping["consecutive_missing_years"].le(0).sum()),
    )

    for field in (
        "carry_direction",
        "consecutive_missing_years",
        "long_range_backcast",
        "sensitivity_required",
        "policy_risk_level",
    ):
        actual_counts = _count_dict(boundary_mapping[field])
        for value, expected in boundary_expected[field].items():
            check(f"boundary_carry {field}={value}", expected, actual_counts.get(value, 0))

    expected_long_range = boundary_mapping["carry_direction"].eq("backward") & boundary_mapping[
        "consecutive_missing_years"
    ].ge(2)
    check(
        "long_range_backcast 판정 불일치",
        0,
        int(boundary_mapping["long_range_backcast"].ne(expected_long_range).sum()),
    )
    check(
        "boundary_carry 민감도 분석 미지정",
        0,
        int((~boundary_mapping["sensitivity_required"]).sum()),
    )

    long_range_targets = {
        "가족실태조사 3종 2016~2019 장기 소급": (
            mapping["지표_id"].isin(
                [
                    "household_chore_equal_participation",
                    "childcare_equal_participation",
                    "socioeconomic_status_perception",
                ]
            )
            & mapping["연도"].le(2019),
            216,
        ),
        "출산 인식 장기 소급": (
            mapping["지표_id"].eq("childbirth_perception") & mapping["연도"].le(2017),
            36,
        ),
        "세종 근로시간 장기 소급": (
            mapping["지표_id"].eq("work_hours") & mapping["지역"].eq("세종"),
            4,
        ),
    }
    for label, (target, expected) in long_range_targets.items():
        check(
            f"{label} high risk",
            expected,
            int(
                (
                    target
                    & mapping["long_range_backcast"]
                    & mapping["policy_risk_level"].eq("high")
                ).sum()
            ),
        )

    source_start_indicators = mapping["지표_id"].isin(
        ["postpartum_center_supply", "postpartum_center_fee", "delivery_bed_supply"]
    )
    check(
        "시계열 시작점 이전 구간(산후조리원 2종·분만실 병상수) keep_missing 적용",
        int(source_start_indicators.sum()),
        int((source_start_indicators & mapping["imputation_policy"].eq("keep_missing")).sum()),
    )
    check(
        "시계열 시작점 이전 구간의 boundary_carry 잔존",
        0,
        int((source_start_indicators & mapping["imputation_policy"].eq("boundary_carry")).sum()),
    )

    issue_82_expected = expected_counts["issue_82_blockers"]
    provinces = mapping["지역"].ne("전국")
    issue_82_blocked = provinces & mapping["block_imputation"]
    family_friendly_provinces = issue_82_blocked & mapping["지표_id"].eq(
        "family_friendly_certification_rate"
    )
    housework_provinces = issue_82_blocked & mapping["지표_id"].eq("housework_gender_equality")
    postpartum_supply_provinces = issue_82_blocked & mapping["지표_id"].eq(
        "postpartum_center_supply"
    )
    postpartum_fee_provinces = issue_82_blocked & mapping["지표_id"].eq("postpartum_center_fee")
    delivery_bed_provinces = issue_82_blocked & mapping["지표_id"].eq("delivery_bed_supply")
    issue_82_known = (
        family_friendly_provinces
        | housework_provinces
        | postpartum_supply_provinces
        | postpartum_fee_provinces
        | delivery_bed_provinces
    )
    check(
        "#82 가족친화 인증기업 비율 차단",
        issue_82_expected["family_friendly_certification_rate"],
        int(family_friendly_provinces.sum()),
    )
    check(
        "#82 가사분담 성평등 인식 차단",
        issue_82_expected["housework_gender_equality"],
        int(housework_provinces.sum()),
    )
    check(
        "#82 산후조리원 보급도 차단(시계열 시작점)",
        issue_82_expected["postpartum_center_supply"],
        int(postpartum_supply_provinces.sum()),
    )
    check(
        "#82 산후조리원 이용 요금 차단(시계열 시작점)",
        issue_82_expected["postpartum_center_fee"],
        int(postpartum_fee_provinces.sum()),
    )
    check(
        "#82 분만실 병상수 차단(시계열 시작점)",
        issue_82_expected["delivery_bed_supply"],
        int(delivery_bed_provinces.sum()),
    )
    check("#82 17개 시도 차단 합계", issue_82_expected["total"], int(issue_82_blocked.sum()))
    check("#82 기타 차단 결측", 0, int((issue_82_blocked & ~issue_82_known).sum()))

    policy_columns = [column for column in MAPPING_COLUMNS if column not in KEY_COLUMNS]
    annotated = normalized.merge(
        mapping[[*KEY_COLUMNS, *policy_columns]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    check("정책 결합 후 행 수 변경", 0, len(annotated) - len(normalized))
    check(
        "정책 결합 후 키 변경",
        0,
        len(
            pd.MultiIndex.from_frame(annotated[KEY_COLUMNS]).symmetric_difference(
                pd.MultiIndex.from_frame(normalized[KEY_COLUMNS])
            )
        ),
    )
    values_unchanged = annotated["측정값"].equals(normalized["측정값"])
    check("실측·결측 측정값 변경", 0, 0 if values_unchanged else 1)
    check(
        "실측 행 정책 열 오염",
        0,
        int(annotated.loc[annotated["측정값"].notna(), "rule_id"].notna().sum()),
    )
    check("처리값 열 생성", 0, int("processed_value" in annotated.columns))

    qa = pd.DataFrame(checks)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"결측 정책 QA 실패: {failures.to_dict('records')}")
    return qa


def _format_years(values: pd.Series) -> str:
    return "·".join(str(value) for value in sorted(values.unique()))


def _format_unique(values: pd.Series) -> str:
    return ", ".join(sorted(str(value) for value in values.unique()))


def render_summary(mapping: pd.DataFrame, qa: pd.DataFrame, config: dict[str, Any]) -> str:
    """지표별 정책과 핵심 검증 결과를 Markdown으로 만든다."""
    lines = [
        "# 구조환경지표 28개 결측 정책 매핑 요약",
        "",
        "## 범위",
        "",
        "- 입력: `data/processed/구조환경지표_28개_보간전_기준패널.csv`",
        f"- 정책 대상: 측정값 결측 {len(mapping):,}건",
        "- 제외 범위: `structural_missing.py` 실행, 처리값 생성, 원본 측정값 변경",
        "- 정책 규칙: `configs/structural_missing_policy.yaml`",
        "- 전수 매핑: `reports/20260804_구조환경지표_결측정책_전수매핑.csv`",
        "",
        "## 정책 원칙",
        "",
        "- `missing_cause`는 결측 원인, `block_imputation`은 실제 대체 차단 여부다.",
        "- 조사 비실시 연도는 자동 차단하지 않고 관측값 사이면 선형보간 후보로 둔다.",
        "- 경계 결측은 본계열 `boundary_carry` 후보로 두되 이번 단계에서는 값을 만들지 않는다.",
        "- `analysis_included_after_imputation`은 현재 포함 상태가 아니라 대체·QA 완료 후 포함 예정 상태다.",
        f"- 원인 미확정 {int(mapping['cause_status'].eq('unresolved').sum()):,}건과 비선형 가사분담 지표는 `pending_review`로 차단한다.",
        "- 완전격자 생성 여부는 `original_row_exists`로 보존하며 더 구체적인 결측 원인을 덮지 않는다.",
        "",
        "## 정책별 건수",
        "",
        "| 구분 | 값 | 건수 |",
        "|---|---|---:|",
    ]
    for field in ("missing_cause", "cause_status", "imputation_policy"):
        expected = config["expected_counts"][field]
        for value in config["allowed_values"][field]:
            lines.append(f"| {field} | {value} | {expected.get(value, 0):,} |")
    lines.extend(
        [
            f"| block_imputation | False | {(~mapping['block_imputation']).sum():,} |",
            f"| block_imputation | True | {mapping['block_imputation'].sum():,} |",
            "| analysis_included_after_imputation | False | "
            f"{(~mapping['analysis_included_after_imputation']).sum():,} |",
            "| analysis_included_after_imputation | True | "
            f"{mapping['analysis_included_after_imputation'].sum():,} |",
            "",
            "`auxiliary_boundary_interpolation`은 본계열 정책 건수에는 포함하지 않는다. ",
            "소득·여가 만족도 2016/2024의 72개 행에만 별도 시나리오로 기록했다.",
            "",
            "## 지표별 요약",
            "",
            "| 지표 | 결측 | 결측연도 | 원인 | 본계열 정책 | 차단 | 대체 후 분석포함 예정 |",
            "|---|---:|---|---|---|---:|---:|",
        ]
    )
    for _, group in mapping.groupby(["지표_id", "지표명"], sort=False):
        lines.append(
            "| {name} (`{indicator}`) | {count:,} | {years} | {causes} | {policies} | "
            "{blocked:,} | {included:,} |".format(
                name=group["지표명"].iloc[0],
                indicator=group["지표_id"].iloc[0],
                count=len(group),
                years=_format_years(group["연도"]),
                causes=_format_unique(group["missing_cause"]),
                policies=_format_unique(group["imputation_policy"]),
                blocked=int(group["block_imputation"].sum()),
                included=int(group["analysis_included_after_imputation"].sum()),
            )
        )

    boundary = mapping.loc[mapping["imputation_policy"].eq("boundary_carry")]
    long_range = boundary.loc[boundary["long_range_backcast"]]
    lines.extend(
        [
            "",
            "## Boundary carry 위험 검토",
            "",
            f"- 대상: {len(boundary):,}건",
            f"- 역방향 carry: {int(boundary['carry_direction'].eq('backward').sum()):,}건",
            f"- 순방향 carry: {int(boundary['carry_direction'].eq('forward').sum()):,}건",
            f"- 장기 소급: {len(long_range):,}건(high), 1년 경계 carry: {len(boundary) - len(long_range):,}건(medium)",
            f"- {len(boundary):,}건 모두 관측구간 밖 외삽 성격이므로 `sensitivity_required=True`다.",
            "- high 정책은 관측값으로 해석할 수 없으며 carry만을 단독 본계열 근거로 삼기에는 타당성이 낮다.",
            "",
            "| 장기 소급 대상 | 행 | 기준 관측연도 | 거리 | 연속 결측 | 평가 |",
            "|---|---:|---:|---:|---:|---|",
            "| 가족실태조사 3종 2016–2019 | 216 | 2020 | 1–4년 | 4년 | high; 2020 조사 수준의 과거 소급은 시점 효과를 제거하므로 민감도 비교 필수 |",
            "| 출산 인식 2016–2017 | 36 | 2018 | 1–2년 | 2년 | high; 문항·조사 가용성 확인 전 과거 수준 확정 불가 |",
            "| 세종 근로시간 2016–2019 | 4 | 2020 | 1–4년 | 4년 | high; 원자료 부재 구간의 추가 장기 소급 사례 |",
            "",
            "산후조리원 보급도·이용요금(2016–2022, 252건)과 분만실 병상수(2016–2017, 36건)는 더 이상 이 "
            "boundary_carry 위험 목록에 없다 — `reports/20260725_구조환경지표_통합파일_결측치_검증_결과.md`가 "
            '이 구간을 "통계 자체가 최근에 신설됨(시계열 시작점 문제)"으로 분류하고 보간(경계값 이월 포함) '
            "불가로 판단했으므로, `keep_missing`으로 정책을 바꿔 결측으로 유지하고 본계열 분석에서 제외한다.",
            "",
            "위 정책은 아직 실행 승인이 아니라 후보 매핑이다. 실제 적용 단계에서는 carry 포함 결과와 ",
            "관측구간 제한 또는 해당 구간 제외 결과를 함께 산출해야 한다.",
            "",
            "## #82 17개 시도 완전 패널 차단",
            "",
            "| 지표 | 차단 결측 | 후속 조건 |",
            "|---|---:|---|",
            "| 가족친화 인증기업 비율 | 51 | 2016·2017·2019 공식 연도별 자료 확보 전 pending_review 유지 |",
            "| 가사분담 성평등 인식 | 68 | 최종 지표 직접 보간 금지, 원산식·구성 응답값 우선 검토 |",
            "| 산후조리원 보급도 | 119 | 시계열 시작점(2023년) 이전 구간, keep_missing 유지 |",
            "| 산후조리원 이용 요금 | 119 | 시계열 시작점(2023년) 이전 구간, keep_missing 유지 |",
            "| 분만실 병상수 보급도 | 34 | 시계열 시작점(2018년 3분기) 이전 구간, keep_missing 유지 |",
            "| 합계 | 391 | 전국 행을 제외한 17개 시도 기준 |",
        ]
    )

    unresolved = mapping.loc[mapping["cause_status"].eq("unresolved")]
    lines.extend(
        [
            "",
            "## 미확정 목록",
            "",
            "| 지표·범위 | 건수 | 상태 | 정책 |",
            "|---|---:|---|---|",
            "| 가족친화 인증기업 비율, 17개 시도 2016·2017·2019 및 전국 2016·2017·2019·2020·2021 | "
            f"{int(unresolved['지표_id'].eq('family_friendly_certification_rate').sum()):,} | "
            "unresolved | pending_review |",
            "| 청년층 정규직 근로자 비율, 전국 2016–2024 | "
            f"{int((unresolved['지표_id'].eq('youth_regular_employment_rate') & unresolved['지역'].eq('전국')).sum()):,} | "
            "unresolved | pending_review |",
            "",
            "비선형 가사분담에 대한 성평등 인식 72건은 원인은 `confirmed`지만 처리 단계가 ",
            "미확정이므로 별도의 `pending_review` 정책으로 차단한다. 최종 변환값을 직접 선형보간하지 ",
            "않고 원산식과 구성 응답값을 확인한 뒤 구성요소 보간 가능성을 먼저 검토한다.",
            "",
            "## QA",
            "",
            f"- QA 항목: {len(qa):,}개, PASS {int(qa['판정'].eq('PASS').sum()):,}개",
            f"- {len(mapping):,}개 결측 키는 중복·누락 없이 정확히 한 번 매핑됨",
            f"- {len(unresolved):,}개 원인 미확정 행은 모두 pending_review·대체 차단·본계열 분석 제외",
            f"- {config['panel']['expected_observed']:,}개 실측값과 전체 {config['panel']['expected_rows']:,}개 키 변경 없음",
            "- 조사 비실시 원인 안에 block_imputation=True/False가 모두 있어 두 개념이 독립됨",
            "- auxiliary_boundary_interpolation 72건은 별도 시나리오 열에만 존재해 본계열 정책과 중복되지 않음",
            "- #82의 17개 시도 차단 결측은 가족친화 51건+가사분담 68건=119건",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/structural_missing_policy.yaml"),
    )
    parser.add_argument("--panel", type=Path)
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("reports/20260804_구조환경지표_결측정책_전수매핑.csv"),
    )
    parser.add_argument(
        "--qa-output",
        type=Path,
        default=Path("reports/20260804_구조환경지표_결측정책_매핑_QA.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/methodology/20260804_구조환경지표_결측정책_요약.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_policy_config(args.config)
    panel_path = args.panel or Path(config["panel"]["path"])
    panel = pd.read_csv(panel_path)
    mapping = build_policy_mapping(panel, config)
    qa = validate_policy_mapping(panel, mapping, config)
    summary = render_summary(mapping, qa, config)

    for path in (args.mapping_output, args.qa_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.mapping_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    args.summary_output.write_text(summary, encoding="utf-8")

    print(f"결측 정책 전수 매핑: {len(mapping):,}건")
    print(f"QA: {len(qa):,}개 항목 모두 PASS")
    print("structural_missing.py 적용 및 처리값 생성: 수행하지 않음")


if __name__ == "__main__":
    main()
