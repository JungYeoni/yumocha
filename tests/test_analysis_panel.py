"""지역×연도 계획예산·합계출산율 패널 유틸 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.analysis_panel import (
    add_fiscal_index_features,
    add_fiscal_response_features,
    add_total_expenditure_ratio,
    build_budget_fertility_panel,
    load_budget_qa_panel,
    load_current_budget_panel,
    load_fertility_panel,
    validate_budget_totals_against_sources,
)


REGIONS = ["서울", "부산"]
YEARS = [2020, 2021]


def _write_budget_file(
    base: Path,
    *,
    region: str,
    year: int,
    current_values: list[float | None],
    previous_values: list[float],
) -> None:
    region_dir = base / region
    region_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (current, previous) in enumerate(zip(current_values, previous_values, strict=True)):
        rows.extend(
            [
                {
                    "지역": region,
                    "연도": year,
                    "세부사업명": f"사업{index}",
                    "예산구분": "당해예산",
                    "예산액": current,
                },
                {
                    "지역": region,
                    "연도": year - 1,
                    "세부사업명": f"사업{index}",
                    "예산구분": "전년도예산",
                    "예산액": previous,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(
        region_dir / f"{year}_{region}_세부사업_정제_long.csv",
        index=False,
    )


def test_load_current_budget_panel_uses_only_current_and_tracks_missing(tmp_path):
    for year in YEARS:
        _write_budget_file(
            tmp_path,
            region="서울",
            year=year,
            current_values=[10.0, None],
            previous_values=[1000.0, 2000.0],
        )
        _write_budget_file(
            tmp_path,
            region="부산",
            year=year,
            current_values=[20.0, -1.0],
            previous_values=[3000.0, 4000.0],
        )

    panel = load_current_budget_panel(
        tmp_path,
        expected_regions=REGIONS,
        expected_years=YEARS,
    )

    seoul_2020 = panel.query("지역 == '서울' and 연도 == 2020").iloc[0]
    busan_2020 = panel.query("지역 == '부산' and 연도 == 2020").iloc[0]
    assert seoul_2020["당해계획예산_백만원"] == 10.0
    assert seoul_2020["예산결측_사업수"] == 1
    assert busan_2020["당해계획예산_백만원"] == 19.0
    assert busan_2020["음수예산_사업수"] == 1


def test_load_current_budget_panel_rejects_wrong_current_year(tmp_path):
    for region in REGIONS:
        _write_budget_file(
            tmp_path,
            region=region,
            year=2020,
            current_values=[10.0],
            previous_values=[9.0],
        )
    path = tmp_path / "서울" / "2020_서울_세부사업_정제_long.csv"
    df = pd.read_csv(path)
    df.loc[df["예산구분"].eq("당해예산"), "연도"] = 2019
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="당해예산 연도 불일치"):
        load_current_budget_panel(
            tmp_path,
            expected_regions=REGIONS,
            expected_years=[2020],
        )


def test_load_fertility_panel_maps_regions_and_excludes_nationwide(tmp_path):
    fertility_path = tmp_path / "fertility.csv"
    mapping_path = tmp_path / "mapping.csv"
    columns = pd.MultiIndex.from_tuples(
        [
            ("시군구별", "시군구별"),
            ("합계출산율", "2020"),
            ("합계출산율", "2021"),
            ("합계출산율", "2025 p)"),
        ]
    )
    pd.DataFrame(
        [
            ["전국", 0.84, 0.81, 0.75],
            ["서울특별시", 0.64, 0.63, 0.60],
            ["부산광역시", 0.75, 0.73, 0.70],
        ],
        columns=columns,
    ).to_csv(fertility_path, index=False, encoding="cp949")
    pd.DataFrame(
        {
            "지역": REGIONS,
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(mapping_path, index=False)

    panel, nationwide = load_fertility_panel(
        fertility_path,
        mapping_path,
        expected_years=YEARS,
    )

    assert len(panel) == 4
    assert set(panel["지역"]) == set(REGIONS)
    assert set(panel["연도"]) == set(YEARS)
    assert nationwide["합계출산율"].tolist() == [0.84, 0.81]


def test_load_fertility_panel_rejects_out_of_range_value(tmp_path):
    fertility_path = tmp_path / "fertility.csv"
    mapping_path = tmp_path / "mapping.csv"
    columns = pd.MultiIndex.from_tuples(
        [
            ("시군구별", "시군구별"),
            ("합계출산율", "2020"),
            ("합계출산율", "2021"),
        ]
    )
    pd.DataFrame(
        [
            ["전국", 0.84, 0.81],
            ["서울특별시", 0.64, 5.1],
            ["부산광역시", 0.75, 0.73],
        ],
        columns=columns,
    ).to_csv(fertility_path, index=False, encoding="cp949")
    pd.DataFrame(
        {
            "지역": REGIONS,
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match=r"허용범위\(0~5\) 이탈"):
        load_fertility_panel(
            fertility_path,
            mapping_path,
            expected_years=YEARS,
        )


def test_load_fertility_panel_rejects_duplicate_year_mapping(tmp_path):
    fertility_path = tmp_path / "fertility.csv"
    mapping_path = tmp_path / "mapping.csv"
    columns = pd.MultiIndex.from_tuples(
        [
            ("시군구별", "시군구별"),
            ("합계출산율", "2020"),
            ("합계출산율", "2020 p)"),
            ("합계출산율", "2021"),
        ]
    )
    pd.DataFrame(
        [
            ["전국", 0.84, 0.83, 0.81],
            ["서울특별시", 0.64, 0.63, 0.62],
            ["부산광역시", 0.75, 0.74, 0.73],
        ],
        columns=columns,
    ).to_csv(fertility_path, index=False, encoding="cp949")
    pd.DataFrame(
        {
            "지역": REGIONS,
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="연도 컬럼 중복 매핑"):
        load_fertility_panel(
            fertility_path,
            mapping_path,
            expected_years=YEARS,
        )


def test_validate_budget_totals_against_sources(tmp_path):
    source_paths = []
    for region, values in {"서울": [10.0, None], "부산": [20.0, -1.0]}.items():
        _write_budget_file(
            tmp_path,
            region=region,
            year=2020,
            current_values=values,
            previous_values=[100.0, 200.0],
        )
        source_paths.append(tmp_path / region / f"2020_{region}_세부사업_정제_long.csv")

    panel = load_current_budget_panel(
        tmp_path,
        expected_regions=REGIONS,
        expected_years=[2020],
    )
    comparison = validate_budget_totals_against_sources(panel, source_paths)

    assert comparison["집계차이_백만원"].eq(0).all()

    panel.loc[panel["지역"].eq("서울"), "당해계획예산_백만원"] += 1
    with pytest.raises(ValueError, match="당해예산 합계 불일치"):
        validate_budget_totals_against_sources(panel, source_paths)

    panel.loc[panel["지역"].eq("서울"), "당해계획예산_백만원"] = float("nan")
    with pytest.raises(ValueError, match="당해예산 합계 불일치"):
        validate_budget_totals_against_sources(panel, source_paths)


def test_validate_budget_totals_against_sources_rejects_non_numeric_source_value(tmp_path):
    """역대조 경로가 build_current_budget_panel과 같은 엄격한 숫자 변환 정책을 써야 한다.

    coerce로 조용히 NaN을 만들면 sum(min_count=1)이 그 값을 건너뛰어서, 원본이
    오염돼 있어도 역대조가 차이=0으로 통과해버린다.
    """
    region_dir = tmp_path / "서울"
    region_dir.mkdir(parents=True, exist_ok=True)
    source = region_dir / "2020_서울_세부사업_정제_long.csv"
    pd.DataFrame(
        [
            {
                "지역": "서울",
                "연도": 2020,
                "세부사업명": "사업0",
                "예산구분": "당해예산",
                "예산액": "미정",
            },
            {
                "지역": "서울",
                "연도": 2020,
                "세부사업명": "사업1",
                "예산구분": "당해예산",
                "예산액": 10.0,
            },
        ]
    ).to_csv(source, index=False)

    panel = pd.DataFrame([{"지역": "서울", "연도": 2020, "당해계획예산_백만원": 10.0}])

    with pytest.raises(ValueError, match="역대조 예산액 숫자 변환 실패"):
        validate_budget_totals_against_sources(panel, [source])


def test_load_budget_qa_panel_handles_error_rate_schema_variants(tmp_path):
    rows = {
        2020: pd.DataFrame(
            {
                "지역": ["서울", "서울"],
                "결과": ["일치", "불일치"],
                "허용기준결과": ["이내", "초과"],
                "절대오차율(%)": [1.0, 4.0],
            }
        ),
        2021: pd.DataFrame(
            {
                "지역": ["서울"],
                "결과": ["불일치"],
                "허용기준결과": ["초과"],
                "오차율(%)": [-3.0],
            }
        ),
        2022: pd.DataFrame(
            {
                "지역": ["서울"],
                "결과": ["판정불가"],
                "허용기준결과": ["판정불가"],
            }
        ),
    }
    for year, frame in rows.items():
        report_dir = tmp_path / "yearly" / str(year)
        report_dir.mkdir(parents=True)
        frame.to_csv(report_dir / f"{year}_전국_QA_검증결과.csv", index=False)

    panel = load_budget_qa_panel(tmp_path, expected_years=[2020, 2021, 2022])

    row_2020 = panel.loc[panel["연도"].eq(2020)].iloc[0]
    row_2021 = panel.loc[panel["연도"].eq(2021)].iloc[0]
    row_2022 = panel.loc[panel["연도"].eq(2022)].iloc[0]
    assert row_2020["예산_QA_그룹수"] == 2
    assert row_2020["예산_QA_허용초과건수"] == 1
    assert row_2020["예산_QA_최대절대오차율"] == 4.0
    assert row_2021["예산_QA_최대절대오차율"] == 3.0
    assert pd.isna(row_2022["예산_QA_최대절대오차율"])


def test_load_budget_qa_panel_rejects_missing_required_column(tmp_path):
    report_dir = tmp_path / "yearly" / "2020"
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "지역": ["서울"],
            "결과": ["일치"],
        }
    ).to_csv(report_dir / "2020_전국_QA_검증결과.csv", index=False)

    with pytest.raises(ValueError, match="필수 컬럼 누락"):
        load_budget_qa_panel(tmp_path, expected_years=[2020])


def test_build_panel_requires_one_to_one_complete_match():
    budget = pd.DataFrame(
        {
            "지역": ["서울", "부산"],
            "연도": [2020, 2020],
            "당해계획예산_백만원": [10.0, 20.0],
        }
    )
    fertility = pd.DataFrame(
        {
            "지역": ["서울"],
            "연도": [2020],
            "합계출산율": [0.64],
        }
    )

    with pytest.raises(ValueError, match="예산·출산율 미매칭"):
        build_budget_fertility_panel(
            budget,
            fertility,
            expected_regions=REGIONS,
            expected_years=[2020],
        )


def test_add_fiscal_response_features_uses_only_prior_tfr():
    panel = pd.DataFrame(
        {
            "지역": ["서울"] * 3,
            "연도": [2020, 2021, 2022],
            "합계출산율": [0.70, 0.60, 0.55],
        }
    )

    result = add_fiscal_response_features(panel)

    row_2022 = result.loc[result["연도"].eq(2022)].iloc[0]
    assert row_2022["전년도_합계출산율"] == 0.60
    assert row_2022["전전년도_합계출산율"] == 0.70
    assert row_2022["직전1년_출산율하락도"] == pytest.approx(0.10)


def test_add_fiscal_index_features_calculates_real_per_capita_and_scores():
    panel = pd.DataFrame(
        {
            "지역": ["부산", "서울", "부산", "서울"],
            "연도": [2021, 2020, 2020, 2021],
            "당해계획예산_백만원": [240.0, 100.0, 180.0, 220.0],
            "소비자물가지수": [120.0, 100.0, 90.0, 110.0],
            "20_39세_인구_명": [20_000, 10_000, 20_000, 10_000],
        }
    )

    result = add_fiscal_index_features(panel)

    seoul_2020 = result.query("지역 == '서울' and 연도 == 2020").iloc[0]
    seoul_2021 = result.query("지역 == '서울' and 연도 == 2021").iloc[0]
    assert seoul_2020["실질계획예산_2020년가격_백만원"] == pytest.approx(100.0)
    assert seoul_2020["20_39세_1인당_실질예산_원"] == pytest.approx(10_000.0)
    assert seoul_2021["실질계획예산_2020년가격_백만원"] == pytest.approx(200.0)
    assert seoul_2021["실질예산_전년증감액_백만원"] == pytest.approx(100.0)
    assert seoul_2021["실질예산_전년증감률"] == pytest.approx(1.0)
    assert result["재정대응지수_z"].mean() == pytest.approx(0.0, abs=1e-12)
    assert result["재정대응지수_z"].std(ddof=0) == pytest.approx(1.0)
    assert result["재정대응점수_0_100"].min() == pytest.approx(0.0)
    assert result["재정대응점수_0_100"].max() == pytest.approx(100.0)
    assert result.groupby("지역")["실질예산_전년증감액_백만원"].first().notna().all()


def test_add_fiscal_index_features_names_real_column_after_given_price_label():
    """실질계획예산 컬럼명이 실제 넘긴 CPI 기준연도와 일치해야 한다.

    2024=100 CPI를 넣고도 컬럼명이 "2020년가격"으로 하드코딩돼 있으면
    계산은 맞아도 이름이 거짓 정보를 표시하게 된다(#62에서 CPI를
    2024=100으로 바꾸며 발견).
    """
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2020, 2021],
            "당해계획예산_백만원": [100.0, 120.0],
            "소비자물가지수": [100.0, 105.0],
            "20_39세_인구_명": [10_000, 9_000],
        }
    )

    result = add_fiscal_index_features(panel, real_price_label="2024년가격")

    assert "실질계획예산_2024년가격_백만원" in result.columns
    assert "실질계획예산_2020년가격_백만원" not in result.columns
    assert result.loc[result["연도"].eq(2020), "실질계획예산_2024년가격_백만원"].iloc[0] == (
        pytest.approx(100.0)
    )


def test_add_total_expenditure_ratio_requires_complete_one_to_one_match():
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2020, 2021],
            "당해계획예산_백만원": [100.0, 120.0],
        }
    )
    denominator = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2020, 2021],
            "예산단계": ["당초예산", "당초예산"],
            "포함회계": ["일반회계+기타특별회계+공기업특별회계"] * 2,
            "자치단체수": [26, 26],
            "세출예산순계액_원": [1_000_000_000, 2_000_000_000],
            "세출예산순계액_백만원": [1_000.0, 2_000.0],
            "출처": ["지방재정365"] * 2,
        }
    )

    result = add_total_expenditure_ratio(panel, denominator)

    assert result["계획예산_총세출비율_pct"].tolist() == pytest.approx([10.0, 6.0])
    assert result["분모대안"].eq("지역통합_3개회계_순계").all()
    assert result["분자분모_단위"].eq("백만원").all()


def test_add_total_expenditure_ratio_rejects_unmatched_keys():
    panel = pd.DataFrame({"지역": ["서울"], "연도": [2020], "당해계획예산_백만원": [100.0]})
    denominator = pd.DataFrame(
        {
            "지역": ["서울"],
            "연도": [2021],
            "예산단계": ["당초예산"],
            "포함회계": ["일반회계+기타특별회계+공기업특별회계"],
            "자치단체수": [26],
            "세출예산순계액_원": [1_000_000_000],
            "세출예산순계액_백만원": [1_000.0],
            "출처": ["지방재정365"],
        }
    )

    with pytest.raises(ValueError, match="키 불일치"):
        add_total_expenditure_ratio(panel, denominator)


def test_add_total_expenditure_ratio_rejects_nonpositive_denominator():
    panel = pd.DataFrame({"지역": ["서울"], "연도": [2020], "당해계획예산_백만원": [100.0]})
    denominator = pd.DataFrame(
        {
            "지역": ["서울"],
            "연도": [2020],
            "예산단계": ["당초예산"],
            "포함회계": ["일반회계+기타특별회계+공기업특별회계"],
            "자치단체수": [26],
            "세출예산순계액_원": [0],
            "세출예산순계액_백만원": [0.0],
            "출처": ["지방재정365"],
        }
    )

    with pytest.raises(ValueError, match="총세출 분모는 0보다"):
        add_total_expenditure_ratio(panel, denominator)


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("당해계획예산_백만원", -1.0, "계획예산은 0 이상"),
        ("소비자물가지수", 0.0, "소비자물가지수는 0보다"),
        ("20_39세_인구_명", 0.0, "대상인구는 0보다"),
        ("소비자물가지수", None, "결측 또는 비수치"),
    ],
)
def test_add_fiscal_index_features_rejects_invalid_inputs(column, invalid_value, message):
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2020, 2021],
            "당해계획예산_백만원": [100.0, 120.0],
            "소비자물가지수": [100.0, 105.0],
            "20_39세_인구_명": [10_000, 9_000],
        }
    )
    panel.loc[1, column] = invalid_value

    with pytest.raises(ValueError, match=message):
        add_fiscal_index_features(panel)


def test_fiscal_feature_functions_reject_non_contiguous_years():
    panel = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2020, 2022],
            "당해계획예산_백만원": [100.0, 120.0],
            "소비자물가지수": [100.0, 105.0],
            "20_39세_인구_명": [10_000, 9_000],
            "합계출산율": [0.70, 0.60],
        }
    )

    with pytest.raises(ValueError, match="연도가 연속적이지"):
        add_fiscal_index_features(panel)
    with pytest.raises(ValueError, match="연도가 연속적이지"):
        add_fiscal_response_features(panel)
