from pathlib import Path

import pandas as pd

from scripts.verify_provisional_roundtrip import compare_frame_to_saved_csv


def test_compare_frame_to_saved_csv_passes_for_faithful_save(tmp_path: Path):
    memory = pd.DataFrame({"지역": ["서울", "부산"], "연도": [2021, 2022], "예산": [1.5, 2.5]})
    csv_path = tmp_path / "panel.csv"
    memory.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = compare_frame_to_saved_csv(memory, csv_path)

    assert result["판정"] == "PASS"
    assert result["값_불일치_셀_수"] == 0
    assert result["shape_일치"]
    assert result["열_순서_일치"]


def test_compare_frame_to_saved_csv_detects_value_drift(tmp_path: Path):
    memory = pd.DataFrame({"지역": ["서울"], "연도": [2021], "예산": [1.5]})
    tampered = pd.DataFrame({"지역": ["서울"], "연도": [2021], "예산": [1.6]})
    csv_path = tmp_path / "panel.csv"
    tampered.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = compare_frame_to_saved_csv(memory, csv_path)

    assert result["판정"] == "FAIL"
    assert result["값_불일치_셀_수"] == 1


def test_compare_frame_to_saved_csv_detects_shape_mismatch(tmp_path: Path):
    memory = pd.DataFrame({"지역": ["서울", "부산"], "연도": [2021, 2022]})
    tampered = pd.DataFrame({"지역": ["서울"], "연도": [2021]})
    csv_path = tmp_path / "panel.csv"
    tampered.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = compare_frame_to_saved_csv(memory, csv_path)

    assert result["판정"] == "FAIL"
    assert not result["shape_일치"]
