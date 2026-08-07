import pandas as pd

import pytest

from scripts.build_youth_regular_employment_national_candidates import (
    INDICATOR_ID,
    apply_national_candidates_to_panel,
    build_candidates,
    calculate_national_rate,
    index_raw_files,
    remove_resolved_rows_from_mapping,
)


def _sample_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "지역": "전국",
                "지표_id": INDICATOR_ID,
                "연도": year,
                "측정값": 60.0 + year - 2016,
                "QA_상태": "PASS",
                "반영유형": "원자료 마이크로데이터 직접계산",
                "관측상태": "관측",
            }
            for year in range(2016, 2025)
        ]
    )


def _write_half_file(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="cp949")


def test_calculate_national_rate_ignores_region_and_uses_correct_masks(tmp_path):
    path = tmp_path / "sample.csv"
    _write_half_file(
        path,
        [
            # 서울, 19-34세, 취업, 임금근로자(상용), 전일제 -> 분자+분모
            {
                "만연령": 25,
                "행정구역시도코드": 11,
                "경제활동인구상태코드": 1,
                "종사상지위코드": 1,
                "주업부업총계시간구분코드": 3,
                "시도전국가중값": 10.0,
            },
            # 부산, 19-34세, 취업, 임금근로자(임시) -> 분모만
            {
                "만연령": 30,
                "행정구역시도코드": 21,
                "경제활동인구상태코드": 1,
                "종사상지위코드": 2,
                "주업부업총계시간구분코드": 1,
                "시도전국가중값": 20.0,
            },
            # 35세 -> 청년 연령 범위 밖, 분모·분자 모두 제외
            {
                "만연령": 40,
                "행정구역시도코드": 11,
                "경제활동인구상태코드": 1,
                "종사상지위코드": 1,
                "주업부업총계시간구분코드": 3,
                "시도전국가중값": 999.0,
            },
        ],
    )

    rate = calculate_national_rate(path)

    # 분자 가중치 10, 분모 가중치 10+20=30 -> 33.333...%
    assert rate == pytest.approx(10.0 / 30.0 * 100, rel=1e-9)


def test_index_raw_files_requires_exactly_18_files(tmp_path):
    for year in range(2016, 2025):
        for half in ("상반기", "하반기"):
            (tmp_path / f"{year}_{half}(C형_시도_중분류_부가항목)_test.csv").write_text("")

    file_index = index_raw_files(tmp_path)

    assert len(file_index) == 18
    assert set(file_index["연도"]) == set(range(2016, 2025))


def test_index_raw_files_rejects_wrong_count(tmp_path):
    (tmp_path / "2016_상반기(C형_시도_중분류_부가항목)_test.csv").write_text("")

    with pytest.raises(ValueError, match="18개가 아닙니다"):
        index_raw_files(tmp_path)


def test_build_candidates_produces_nine_rows_with_sanity_bounds(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    regions = {11: "서울", 21: "부산"}
    for year in range(2016, 2025):
        for half in ("상반기", "하반기"):
            path = raw_dir / f"{year}_{half}(C형_시도_중분류_부가항목)_test.csv"
            _write_half_file(
                path,
                [
                    {
                        "만연령": 25,
                        "행정구역시도코드": code,
                        "경제활동인구상태코드": 1,
                        "종사상지위코드": 1,
                        "주업부업총계시간구분코드": 3,
                        "시도전국가중값": 10.0,
                    }
                    for code in regions
                ],
            )

    province_annual_path = tmp_path / "province_annual.csv"
    pd.DataFrame(
        {
            "시도": list(regions.values()),
            **{str(year): [100.0, 100.0] for year in range(2016, 2025)},
        }
    ).to_csv(province_annual_path, index=False, encoding="utf-8-sig")

    candidates, qa = build_candidates(raw_dir, province_annual_path)

    assert len(candidates) == 9
    assert candidates["지역"].eq("전국").all()
    assert qa["판정"].eq("PASS").all()


def _sample_panel() -> pd.DataFrame:
    rows = [
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": year,
            "측정값": float("nan"),
            "원본행존재": False,
        }
        for year in range(2016, 2025)
    ]
    rows.append(
        {"지역": "서울", "지표_id": INDICATOR_ID, "연도": 2018, "측정값": 61.0, "원본행존재": True}
    )
    return pd.DataFrame(rows)


def test_apply_national_candidates_to_panel_fills_only_the_nine_missing_cells():
    panel = _sample_panel()
    candidates = _sample_candidates()

    updated, audit = apply_national_candidates_to_panel(panel, candidates)

    assert len(audit) == 9
    assert audit["QA_상태"].eq("PASS").all()
    assert audit["반영전값"].isna().all()
    for year in range(2016, 2025):
        row = updated.loc[updated["지역"].eq("전국") & updated["연도"].eq(year)].iloc[0]
        expected = candidates.set_index("연도").loc[year, "측정값"]
        assert row["측정값"] == expected
        assert bool(row["원본행존재"]) is True

    seoul_row = updated.loc[updated["지역"].eq("서울")].iloc[0]
    assert seoul_row["측정값"] == 61.0


def test_apply_national_candidates_to_panel_rejects_existing_value():
    panel = _sample_panel()
    panel.loc[panel["연도"].eq(2016), "측정값"] = 999.0
    candidates = _sample_candidates()

    with pytest.raises(ValueError, match="이미 존재"):
        apply_national_candidates_to_panel(panel, candidates)


def _sample_mapping() -> pd.DataFrame:
    rows = [
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": year,
            "block_imputation": True,
            "imputation_policy": "pending_review",
        }
        for year in range(2016, 2025)
    ]
    rows.append(
        {
            "지역": "세종",
            "지표_id": INDICATOR_ID,
            "연도": 2016,
            "block_imputation": False,
            "imputation_policy": "boundary_carry",
        }
    )
    return pd.DataFrame(rows)


def test_remove_resolved_rows_from_mapping_drops_only_national_nine():
    mapping = _sample_mapping()
    candidates = _sample_candidates()

    updated, removed = remove_resolved_rows_from_mapping(mapping, candidates)

    assert len(removed) == 9
    assert len(updated) == 1
    assert updated.iloc[0]["지역"] == "세종"
