import pandas as pd

from scripts.apply_2023_blank_input_recovery import resolve_checkpoint_indices


def test_resolve_checkpoint_indices_uses_neighbor_anchors_for_duplicate_value():
    wide = pd.DataFrame(
        [
            {"지역": "서울", "주요내용_정제": "앞 앵커"},
            {"지역": "서울", "주요내용_정제": "중복 사업명"},
            {"지역": "서울", "주요내용_정제": "뒤 앵커"},
            {"지역": "부산", "주요내용_정제": "중복 사업명"},
        ]
    )
    checkpoint = pd.DataFrame(
        {"주요내용_정제": ["앞 앵커", "중복 사업명", "뒤 앵커", "중복 사업명"]},
        index=[10, 11, 12, 20],
    )
    targets = wide.loc[[1]]

    result = resolve_checkpoint_indices(wide, checkpoint, targets)

    assert result == {1: 11}
