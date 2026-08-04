"""구조환경지수 표준화·AHP 가중치 적용 유틸."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd

from src.evaluation.structural_validation import (
    require_columns,
    to_numeric_strict,
)

VALID_DIRECTIONS = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    # explicit center label preserved; mapping/validation is handled
    # in the standardization step where indicator-specific rules apply
    "center_or_positive": "center_or_positive",
}

DEFAULT_WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass
class StructuralIndexResult:
    indicator_scores: pd.DataFrame
    subcategory_scores: pd.DataFrame
    category_scores: pd.DataFrame
    final_index: pd.DataFrame


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load structural index YAML configs. Install pyyaml and retry."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 파일이 올바른 매핑 형식이 아닙니다: {path}")
    return data


def _normalize_direction(direction: str, *, source_name: str) -> str:
    if not isinstance(direction, str):
        raise ValueError(f"{source_name}: direction 값은 문자열이어야 합니다: {direction}")
    key = direction.strip().lower()
    if key in VALID_DIRECTIONS:
        return VALID_DIRECTIONS[key]
    raise ValueError(
        f"{source_name}: 방향성 값이 올바르지 않습니다. 허용값: positive/negative, center_or_positive, +/-. 실제: {direction}"
    )


def _ensure_non_negative_weights(weights: pd.Series, *, source_name: str) -> None:
    if (weights < 0).any():
        neg = weights[weights < 0]
        raise ValueError(f"{source_name}: 음수 가중치가 있습니다: {neg.to_dict()}")


def load_structural_indicator_manifest(repo_root: Path) -> pd.DataFrame:
    config_path = repo_root / "configs" / "structural_indicators_verification.yaml"
    content = _load_yaml(config_path)
    indicators = content.get("indicators")
    if not isinstance(indicators, list):
        raise ValueError(
            "structural_indicators_verification.yaml: indicators 목록을 찾을 수 없습니다."
        )

    records = []
    for item in indicators:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "id": item["id"],
                "name": item.get("name"),
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "direction": _normalize_direction(
                    item.get("direction", ""), source_name=f"manifest 지표 {item.get('id')}"
                ),
            }
        )
    return pd.DataFrame(records)


def load_structural_index_weights(repo_root: Path) -> pd.DataFrame:
    config_path = repo_root / "configs" / "structural_index_weights.yaml"
    content = _load_yaml(config_path)
    weights_map = content.get("weights")
    if not isinstance(weights_map, dict):
        raise ValueError("structural_index_weights.yaml: weights 매핑을 찾을 수 없습니다.")

    records = []
    for indicator_id, payload in weights_map.items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"structural_index_weights.yaml: {indicator_id}의 값은 매핑이어야 합니다."
            )
        records.append(
            {
                "id": indicator_id,
                "code": payload.get("code"),
                "name": payload.get("name"),
                "weight": float(payload.get("weight")),
            }
        )
    df = pd.DataFrame(records)
    if df["id"].duplicated().any():
        duplicates = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"structural_index_weights.yaml: 중복 지표 ID가 있습니다: {duplicates}")
    _ensure_non_negative_weights(df["weight"], source_name="structural_index_weights.yaml")
    return df


def validate_structural_index_weights(
    weights: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_WEIGHT_SUM_TOLERANCE,
) -> None:
    required_cols = ["id", "weight"]
    require_columns(weights, required_cols, source_name="AHP 가중치 검증")
    require_columns(manifest, ["id"], source_name="지표 매니페스트 검증")

    indicator_ids = set(manifest["id"].astype(str))
    weight_ids = set(weights["id"].astype(str))

    missing = sorted(indicator_ids - weight_ids)
    extra = sorted(weight_ids - indicator_ids)
    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"매니페스트에 있지만 가중치에 없는 지표: {missing}")
        if extra:
            msg_parts.append(f"가중치에 있지만 매니페스트에 없는 지표: {extra}")
        raise ValueError(" / ".join(msg_parts))

    total_weight = float(weights["weight"].sum())
    if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
        raise ValueError(f"AHP 가중치 합계가 1이 아닙니다: {total_weight} (허용오차 {tolerance})")
    # ensure optional metadata present
    if "code" not in weights.columns or "name" not in weights.columns:
        raise ValueError("AHP 가중치는 'code'와 'name' 필드를 포함해야 합니다.")
    valid_codes = weights["code"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    valid_names = weights["name"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    missing_meta = weights[~valid_codes | ~valid_names]
    if not missing_meta.empty:
        raise ValueError(
            f"AHP 가중치에 code/name 누락이 있습니다: {missing_meta.to_dict(orient='records')}"
        )


def prepare_structural_index_input(
    df: pd.DataFrame,
    *,
    region_col: str,
    year_col: str,
    indicator_id_col: str,
    value_col: str,
    category_col: str,
    subcategory_col: str,
    direction_col: str,
) -> pd.DataFrame:
    rename_map = {
        region_col: "region",
        year_col: "year",
        indicator_id_col: "indicator_id",
        value_col: "value",
        category_col: "category",
        subcategory_col: "subcategory",
        direction_col: "direction",
    }
    require_columns(df, rename_map.keys(), source_name="입력 어댑터")
    result = df.rename(columns=rename_map).copy()
    return result


def validate_structural_index_input(
    df: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
    *,
    expected_regions: Iterable[str] | None = None,
    expected_years: Iterable[int] | None = None,
    expected_indicator_ids: Iterable[str] | None = None,
    nationwide_label: str = "전국",
    allow_missing_values: bool = False,
    strict_grid: bool = True,
) -> pd.DataFrame:
    """
    Validate input frame for structural index computation.

    Contract: `value` MUST be the final measured indicator value after
    applying the indicator-specific formula, not a raw survey response.
    `housework_gender_equality` MUST receive a value produced by its
    `official_formula` or `recommended_formula`. Numeric ranges alone cannot
    distinguish raw responses from final values, so compliance with this
    contract is verified when the #80/#70 panels are connected.

    `allow_missing_values=True` only supports diagnostic input inspection.
    It does not allow missing values in final standardization or index
    computation.

    Columns required: `region, year, indicator_id, category, subcategory,
    direction, value`.
    """

    required_columns = [
        "region",
        "year",
        "indicator_id",
        "category",
        "subcategory",
        "direction",
        "value",
    ]
    require_columns(df, required_columns, source_name="구조환경지수 입력 검증")

    if df.duplicated(subset=["region", "year", "indicator_id"]).any():
        dupes = df[df.duplicated(subset=["region", "year", "indicator_id"], keep=False)]
        raise ValueError(
            f"지역+연도+지표 ID 중복이 있습니다: {dupes[['region', 'year', 'indicator_id']].drop_duplicates().to_dict(orient='records')}"
        )

    values = to_numeric_strict(df["value"], allowed_missing_tokens=frozenset())
    df = df.copy()
    df["value"] = values

    nonnumeric_mask = ~df["value"].isna() & ~np.isfinite(df["value"])
    if nonnumeric_mask.any():
        bad_rows = df.loc[nonnumeric_mask]
        raise ValueError(
            f"무한값 또는 비수치 값이 있습니다: {bad_rows[['region', 'year', 'indicator_id', 'value']].to_dict(orient='records')}"
        )

    missing_values = df["value"].isna().sum()
    if missing_values > 0 and not allow_missing_values:
        raise ValueError(f"측정값 결측이 있습니다: {missing_values}건")

    if expected_indicator_ids is None and manifest is not None:
        expected_indicator_ids = manifest["id"].astype(str).tolist()

    if expected_indicator_ids is not None:
        expected_set = set(str(i) for i in expected_indicator_ids)
        actual_set = set(df["indicator_id"].astype(str))
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"누락 지표 ID: {missing}")
            if extra:
                parts.append(f"등록되지 않은 지표 ID: {extra}")
            raise ValueError(" / ".join(parts))

    if expected_regions is not None and expected_years is not None and strict_grid:
        expected_regions = list(expected_regions)
        expected_years = list(expected_years)
        expected_combinations = {
            (region, year, indicator_id)
            for region in expected_regions
            for year in expected_years
            for indicator_id in df["indicator_id"].unique()
        }
        actual_combinations = set(
            df.loc[df["region"] != nationwide_label, ["region", "year", "indicator_id"]].itertuples(
                index=False, name=None
            )
        )
        missing = sorted(expected_combinations - actual_combinations)
        if missing:
            raise ValueError(
                f"완전격자에 필요한 지역+연도+지표 조합이 누락되었습니다: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    if not (df["region"].astype(str) == nationwide_label).any():
        raise ValueError(f"전국 행이 포함되어 있지 않습니다(지역='{nationwide_label}').")

    for direction in df["direction"].dropna().unique():
        _normalize_direction(str(direction), source_name="입력 방향성")

    # Enforce that center_or_positive is only present where explicitly
    # allowed by the manifest and that center_or_negative is not supported.
    centers = [d for d in df["direction"].dropna().unique() if str(d) == "center_or_positive"]
    if centers:
        # Only the specific indicator is permitted to use center-based direction
        bad = df.loc[df["direction"].astype(str) == "center_or_positive", "indicator_id"].unique()
        allowed = {"housework_gender_equality"}
        badset = set(map(str, bad)) - allowed
        if badset:
            raise ValueError(
                f"center_or_positive 라벨은 허용되지 않는 지표에 사용되었습니다: {sorted(badset)}.\n"
                f"현재 허용된 지표: {sorted(allowed)}. 입력값은 지표별 산식(official/recommended)을 적용한 최종값이어야 합니다."
            )

    return df


def standardize_structural_indicators(
    df: pd.DataFrame,
    *,
    method: str = "pooled",
    nationwide_label: str = "전국",
    expected_regions: Iterable[str] | None = None,
    expected_years: Iterable[int] | None = None,
    expected_indicator_ids: Iterable[str] | None = None,
    allow_missing_values: bool = False,
) -> pd.DataFrame:
    """Standardize final indicator values to a 0-100 scale.

    `allow_missing_values` only relaxes the initial diagnostic validation.
    Missing values are always rejected before standardization output is
    produced and therefore cannot flow into final index computation.
    """

    # Implementation uses pooled or yearly Min-Max.
    allowed_methods = {"pooled", "yearly"}

    if method not in allowed_methods:
        raise ValueError(f"method는 {allowed_methods} 중 하나여야 합니다. 실제: {method}")

    df = validate_structural_index_input(
        df,
        manifest=pd.DataFrame(),
        expected_regions=expected_regions,
        expected_years=expected_years,
        expected_indicator_ids=expected_indicator_ids,
        nationwide_label=nationwide_label,
        allow_missing_values=allow_missing_values,
    )
    # Do not allow missing values to propagate into final standardization.
    # `allow_missing_values` is diagnostic only and will not permit final
    # aggregation with missing numbers.
    if df["value"].isna().any():
        raise ValueError(
            "입력에 결측값이 포함되어 있습니다. allow_missing_values는 진단용이며 최종 지수 계산에서는 허용되지 않습니다."
        )
    data = df.loc[df["region"] != nationwide_label].copy()
    if data.empty:
        raise ValueError("전국을 제외한 지역 데이터가 존재하지 않습니다.")

    if method == "pooled":
        grouping = ["indicator_id"]
    else:
        grouping = ["indicator_id", "year"]

    records = []
    for group_keys, group in data.groupby(grouping, sort=False):
        values = group["value"].astype(float)
        if values.isna().all():
            raise ValueError(f"그룹에 유효한 값이 없습니다: {group_keys}")

        # keep z-score for diagnostics but final score uses Min-Max on raw values
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        group = group.copy()
        group["z_score"] = (
            0.0
            if (std == 0.0 or math.isclose(std, 0.0, abs_tol=0.0))
            else (group["value"] - mean) / std
        )

        vmin = float(values.min())
        vmax = float(values.max())

        if method == "pooled":
            # pooled Min-Max across all years/regions for indicator
            if math.isclose(vmax, vmin, abs_tol=0.0):
                raise ValueError(f"pooled 표준화에서 값의 범위가 0인 지표가 있습니다: {group_keys}")
            group["score_0_100"] = 100.0 * (group["value"] - vmin) / (vmax - vmin)
            zmin = float(group["z_score"].min())
            zmax = float(group["z_score"].max())
            zero_variance = False
        else:
            # yearly Min-Max: if year's values are identical, set midpoint 50
            if math.isclose(vmax, vmin, abs_tol=0.0):
                group["score_0_100"] = 50.0
                zero_variance = True
            else:
                group["score_0_100"] = 100.0 * (group["value"] - vmin) / (vmax - vmin)
                zero_variance = False
            zmin = float(group["z_score"].min())
            zmax = float(group["z_score"].max())

        category = group["category"].iloc[0]
        subcategory = group["subcategory"].iloc[0]
        direction_strings = {str(value) for value in group["direction"].dropna().unique()}
        if len(direction_strings) != 1:
            raise ValueError(
                f"동일 지표 그룹에 서로 다른 방향성이 있습니다: {group_keys} / directions={direction_strings}"
            )
        raw_direction = str(group["direction"].iloc[0])
        direction = _normalize_direction(
            raw_direction, source_name="standardize_structural_indicators"
        )
        # center handling: only allow for whitelisted indicators and treat as positive
        if direction == "center_or_positive":
            if (
                group["indicator_id"].nunique() != 1
                or group["indicator_id"].iloc[0] != "housework_gender_equality"
            ):
                raise ValueError(
                    f"center_or_positive label used for non-supported indicator: {group_keys} / indicator_ids={group['indicator_id'].unique().tolist()}"
                )
            # treat as positive under the contract that `value` is the post-processed metric
            direction = "positive"

        if (
            not zero_variance
            and ((group["score_0_100"] < -1e-8) | (group["score_0_100"] > 100.0 + 1e-8)).any()
        ):
            raise ValueError(
                f"0-100 환산값 범위 이탈: {group_keys} / min={float(group['score_0_100'].min())} max={float(group['score_0_100'].max())}"
            )

        group["score_0_100"] = group["score_0_100"].clip(0.0, 100.0)
        if direction == "negative":
            group["directional_score"] = 100.0 - group["score_0_100"]
        else:
            group["directional_score"] = group["score_0_100"]

        group["group_mean"] = mean
        group["group_std"] = std
        group["group_min_z"] = zmin
        group["group_max_z"] = zmax
        group["method"] = method
        group["zero_variance"] = bool(zero_variance)

        for _, row in group.iterrows():
            records.append(
                {
                    "region": row["region"],
                    "year": int(row["year"]),
                    "indicator_id": row["indicator_id"],
                    "category": category,
                    "subcategory": subcategory,
                    "direction": direction,
                    "value": float(row["value"]),
                    "method": method,
                    "z_score": float(row["z_score"]),
                    "group_mean": float(row["group_mean"]),
                    "group_std": float(row["group_std"]),
                    "group_min_z": float(row["group_min_z"]),
                    "group_max_z": float(row["group_max_z"]),
                    "score_0_100": float(row["score_0_100"]),
                    "directional_score": float(row["directional_score"]),
                    "zero_variance": bool(row["zero_variance"]),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("표준화 결과가 비어 있습니다.")
    return result


def compute_structural_index(
    indicator_scores: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    nationwide_label: str = "전국",
    indicator_id_col: str = "indicator_id",
    region_col: str = "region",
    year_col: str = "year",
    category_col: str = "category",
    subcategory_col: str = "subcategory",
    score_col: str = "directional_score",
    weight_col: str = "weight",
) -> StructuralIndexResult:
    """Compute a final index from complete, finite directional scores.

    Diagnostic missing-value allowances do not apply here. Any missing,
    non-numeric, or non-finite score raises before weighted contributions are
    calculated, preventing a partial index from being produced.
    """

    require_columns(
        indicator_scores,
        [region_col, year_col, indicator_id_col, category_col, subcategory_col, score_col],
        source_name="지수 산출 입력",
    )
    require_columns(weights, ["id", weight_col], source_name="AHP 가중치")

    numeric_scores = pd.to_numeric(indicator_scores[score_col], errors="coerce")
    score_values = numeric_scores.to_numpy(dtype=float, na_value=np.nan)
    invalid_score_mask = ~np.isfinite(score_values)
    if invalid_score_mask.any():
        invalid_count = int(invalid_score_mask.sum())
        diagnostics = (
            indicator_scores.loc[
                invalid_score_mask,
                [region_col, year_col, indicator_id_col],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{score_col}에 결측값 또는 비유한값이 {invalid_count}개 있습니다. "
            f"부분 지수는 계산하지 않습니다. 예시: {diagnostics}"
        )

    indicator_scores = indicator_scores.copy()
    indicator_scores[score_col] = numeric_scores.astype(float)

    merged = indicator_scores.merge(
        weights.rename(columns={"id": indicator_id_col}),
        on=indicator_id_col,
        how="left",
        validate="many_to_one",
    )
    if merged[weight_col].isna().any():
        missing = sorted(merged.loc[merged[weight_col].isna(), indicator_id_col].unique().tolist())
        raise ValueError(f"가중치가 없는 지표가 있습니다: {missing}")

    merged["indicator_weighted_contribution"] = merged[score_col] * merged[weight_col]

    subcategory_totals = (
        merged[[indicator_id_col, subcategory_col, weight_col]]
        .drop_duplicates()
        .groupby(subcategory_col, sort=False, observed=True)[weight_col]
        .sum()
    )
    category_totals = (
        merged[[indicator_id_col, category_col, weight_col]]
        .drop_duplicates()
        .groupby(category_col, sort=False, observed=True)[weight_col]
        .sum()
    )

    subcat_scores = (
        merged.groupby(
            [region_col, year_col, category_col, subcategory_col], sort=False, observed=True
        )["indicator_weighted_contribution"]
        .sum()
        .reset_index()
    )
    subcat_scores["subcategory_weight_total"] = subcat_scores[subcategory_col].map(
        subcategory_totals
    )
    subcat_scores["subcategory_score"] = (
        subcat_scores["indicator_weighted_contribution"] / subcat_scores["subcategory_weight_total"]
    )
    subcat_scores = subcat_scores.rename(
        columns={"indicator_weighted_contribution": "subcategory_contribution"}
    )

    category_scores = (
        subcat_scores.groupby([region_col, year_col, category_col], sort=False, observed=True)
        .agg(category_contribution=("subcategory_contribution", "sum"))
        .reset_index()
    )
    category_scores["category_weight_total"] = category_scores[category_col].map(category_totals)
    category_scores["category_score"] = (
        category_scores["category_contribution"] / category_scores["category_weight_total"]
    )

    final_index = (
        merged.groupby([region_col, year_col], sort=False, observed=True)
        .agg(
            final_index=("indicator_weighted_contribution", "sum"),
            missing_indicator_count=(score_col, lambda s: int(s.isna().sum())),
            indicator_count=(indicator_id_col, "nunique"),
        )
        .reset_index()
    )
    final_index["has_missing_indicators"] = final_index["missing_indicator_count"] > 0

    non_nationwide = final_index[final_index[region_col] != nationwide_label].copy()
    non_nationwide["rank"] = (
        non_nationwide.groupby(year_col)["final_index"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    rank_map = non_nationwide.set_index([region_col, year_col])["rank"]
    final_index["rank"] = (
        final_index.set_index([region_col, year_col]).index.map(rank_map).astype("Int64")
    )

    return StructuralIndexResult(
        indicator_scores=merged,
        subcategory_scores=subcat_scores,
        category_scores=category_scores,
        final_index=final_index,
    )


__all__ = [
    "StructuralIndexResult",
    "load_structural_indicator_manifest",
    "load_structural_index_weights",
    "validate_structural_index_weights",
    "prepare_structural_index_input",
    "validate_structural_index_input",
    "standardize_structural_indicators",
    "compute_structural_index",
]
