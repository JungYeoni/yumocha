"""2021년 영역분류 라벨 취합 스크립트 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.consolidate_2021_area_labels import (
    MAJOR_BY_SUBCATEGORY,
    normalize_text,
    region_from_filename,
)


def test_normalize_text_handles_missing_unicode_and_whitespace():
    assert normalize_text(pd.NA) == ""
    assert normalize_text("  주요내용\n 정제  ") == "주요내용 정제"
    assert normalize_text("가") == "가"


def test_region_from_filename_finds_exact_region():
    path = Path("(완료) 라벨링_2021_서울_세부사업_정제.xlsx")
    assert region_from_filename(path) == "서울"


def test_region_from_filename_rejects_unknown_region():
    with pytest.raises(ValueError, match="지역을 하나로 판별"):
        region_from_filename(Path("라벨링_2021_전국.xlsx"))


def test_every_subcategory_maps_to_one_canonical_major_category():
    assert len(MAJOR_BY_SUBCATEGORY) == 12
    assert MAJOR_BY_SUBCATEGORY["2-1. 돌봄 여건"] == "2. 가족·생활"
    assert MAJOR_BY_SUBCATEGORY["4-1. 일·가정 양립 여건"] == "4. 사회·문화"
    assert MAJOR_BY_SUBCATEGORY["지표체계 외"] == "지표체계 외"
