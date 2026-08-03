from scripts.apply_2024_semantic_corrections import (
    CorrectionRule,
    corrected_text,
)


def test_corrected_text_restores_original():
    rule = CorrectionRule("서울", "1", "restore_original", "요약 복구")

    assert corrected_text(rule, original="원문 전체", cleaned="요약") == "원문 전체"


def test_corrected_text_strips_added_source_label():
    rule = CorrectionRule("서울", "1", "strip_source_label", "라벨 제거")

    assert (
        corrected_text(
            rule,
            original="오타 원문",
            cleaned="사업명 원문: 오타를 고친 정제문",
        )
        == "오타를 고친 정제문"
    )


def test_corrected_text_is_idempotent_after_source_label_removed():
    rule = CorrectionRule("서울", "1", "strip_source_label", "라벨 제거")

    assert corrected_text(rule, original="원문", cleaned="라벨 없는 정제문") == "라벨 없는 정제문"
