import pytest

from scripts.apply_2020_semantic_corrections import strip_source_label


def test_strip_source_label_keeps_only_cleaned_body():
    assert strip_source_label("사업명 원문: 실제 정제문") == "실제 정제문"


def test_strip_source_label_requires_marker():
    with pytest.raises(ValueError, match="원문"):
        strip_source_label("메타 라벨 없는 정제문")
