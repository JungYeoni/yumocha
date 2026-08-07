import pandas as pd

from src.provisional.loader import _merge_continuation_rows


def test_merges_2022_seoul_row_28_into_row_24():
    """서울 원본행 28("위한 안전망 구축")을 24행("학대피해아동 보호를")에 합친다.

    #73 감사 wide 파일과 대조해 발견한 실제 버그를 재현한다 — 기존 코드는
    28행을 leaf에서 제외(헤더반복)만 하고 텍스트 병합을 빠뜨려 24행
    사업명이 "학대피해아동 보호를"로 잘려 있었다.
    """
    frame = pd.DataFrame(
        [
            {"원본행": 24, "세부사업명": "학대피해아동  보호를", "주요내용": pd.NA},
            {"원본행": 28, "세부사업명": "위한 안전망 구축", "주요내용": pd.NA},
        ]
    )
    result = _merge_continuation_rows(frame, {28: 24}, label="2022")
    row24 = result.loc[result["원본행"].eq(24)].iloc[0]
    assert row24["세부사업명"] == "학대피해아동 보호를 위한 안전망 구축"


def test_merge_continuation_rows_is_noop_when_targets_absent():
    frame = pd.DataFrame([{"원본행": 1, "세부사업명": "무관한 사업", "주요내용": pd.NA}])
    result = _merge_continuation_rows(frame, {28: 24}, label="2022")
    pd.testing.assert_frame_equal(result, frame)
