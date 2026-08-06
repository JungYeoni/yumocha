from pathlib import Path

import pytest

from scripts.scrape_ffsb_cert_companies import COLUMNS, parse_rows, parse_total_pages

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_parse_total_pages_reads_last_page_link():
    html = _read_fixture("ffsb_cert_2017_page1.html")

    assert parse_total_pages(html) == 54


def test_parse_total_pages_defaults_to_one_without_pagination():
    assert parse_total_pages("<html><body>no results</body></html>") == 1


def test_parse_rows_extracts_expected_columns():
    html = _read_fixture("ffsb_cert_2017_page1.html")

    rows = parse_rows(html)

    assert rows, "2017년 1페이지에는 결과가 있어야 한다"
    assert set(rows[0].keys()) == set(COLUMNS)
    assert rows[0]["신규인증년도"] == "2017"
    assert rows[0]["지역"] == "인천"


def test_parse_rows_strips_label_prefix_not_just_year_page():
    html = _read_fixture("ffsb_cert_all_page2.html")

    rows = parse_rows(html)

    names = [row["기업(관)명"] for row in rows]
    assert "(주)정식품" in names
    assert all(not name.startswith("기업(관)명") for name in names)


@pytest.mark.parametrize("column", COLUMNS)
def test_parse_rows_never_leaks_raw_label_text(column):
    html = _read_fixture("ffsb_cert_2017_page1.html")

    rows = parse_rows(html)

    assert all(":" not in row[column] for row in rows)
