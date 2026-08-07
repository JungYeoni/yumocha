from scripts.build_fiscal_fertility_panel_from_provisional import (
    DEFAULT_BUDGET_PANEL,
    QUALITY_NOTES,
    YEARS,
    parse_args,
)


def test_parse_args_defaults_point_to_provisional_budget_panel():
    args = parse_args([])
    assert args.budget_panel == DEFAULT_BUDGET_PANEL
    assert args.budget_panel.name == "provisional_budget_panel.csv"


def test_years_covers_2016_through_2024():
    assert YEARS == list(range(2016, 2025))


def test_quality_notes_has_required_columns_and_known_rows():
    assert set(QUALITY_NOTES.columns) == {"지역", "연도", "원자료_누락주의"}
    assert set(zip(QUALITY_NOTES["지역"], QUALITY_NOTES["연도"], strict=False)) == {
        ("강원", 2018),
        ("전남", 2018),
    }
