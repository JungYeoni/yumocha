"""재정팀 검토 라벨과 최신 정제본을 결합한 전체 연도 검토 마스터를 만든다."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scripts.consolidate_2021_area_labels import consolidate_labels
from src.features.review_keys import KEY_COLUMNS, normalize_review_keys

REVIEW_COLUMNS = [
    "예측_대영역",
    "예측_세부영역",
    "예측_신뢰도",
    "저신뢰_검토대상",
    "검토_대영역",
    "검토_세부영역",
    "검토상태",
    "명칭-내용 불일치 / 원본PDF 동일 여부",
    "검토메모",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        dtype={"원본행": "string"},
    )


def changed_cleaned_flags(
    previous: pd.Series,
    latest: pd.Series,
    new_rows: pd.Series,
) -> pd.Series:
    """기존 검토본에 있던 행 중 실제 정제문이 달라진 행만 표시한다."""
    return ~new_rows & previous.fillna("").ne(latest.fillna(""))


def read_review_workbook(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """검토본의 수작업 컬럼을 읽고 세부영역-대영역 매핑을 반환한다."""
    review = pd.read_excel(
        path,
        sheet_name="영역분류검토",
        dtype={"원본행": "string"},
        engine="openpyxl",
    )
    labels = pd.read_excel(path, sheet_name="라벨목록", engine="openpyxl")
    review = normalize_review_keys(review)
    label_map = dict(zip(labels["세부영역"], labels["대영역"], strict=False))
    review["검토_대영역"] = review["검토_세부영역"].map(label_map)
    if review["검토_대영역"].isna().any():
        invalid = review.loc[review["검토_대영역"].isna(), "검토_세부영역"].unique()
        raise ValueError(f"검토 세부영역의 대영역 매핑 실패: {invalid.tolist()}")
    if review.duplicated(KEY_COLUMNS).any():
        raise ValueError("재정팀 검토본에 연도·지역·원본행 중복이 있습니다.")
    return review, label_map


def build_master(
    review: pd.DataFrame,
    labels_2021: pd.DataFrame,
    prediction_targets: pd.DataFrame,
    semantic_impact: pd.DataFrame,
) -> pd.DataFrame:
    """최신 정제 57,979건에 사람 검토 라벨을 키 기준으로 결합한다."""
    review = normalize_review_keys(review)
    labels_2021 = normalize_review_keys(labels_2021)
    prediction_targets = normalize_review_keys(prediction_targets)
    semantic_impact = normalize_review_keys(semantic_impact)

    base_columns = [
        "연도",
        "지역",
        "대분류",
        "중분류",
        "사업분류재정구분",
        "세부사업명",
        "주요내용",
        "주요내용_정제",
        "당해예산",
        "전년도예산",
        "증감액",
        "증감율",
        "원본행",
        "지원대상",
        "지원내용_상세",
    ]
    latest = pd.concat(
        [prediction_targets[base_columns], labels_2021[base_columns]],
        ignore_index=True,
    )
    if len(latest) != 57_979:
        raise ValueError(f"전체 최신 정제본은 57,979건이어야 합니다: {len(latest)}")
    if latest.duplicated(KEY_COLUMNS).any():
        raise ValueError("전체 최신 정제본에 연도·지역·원본행 중복이 있습니다.")

    old_text = review[KEY_COLUMNS + ["주요내용_정제"]].rename(
        columns={"주요내용_정제": "기존검토본_주요내용_정제"}
    )
    review = review.copy()
    review["_기존검토순서"] = range(len(review))
    metadata = review[KEY_COLUMNS + REVIEW_COLUMNS + ["_기존검토순서"]]
    master = latest.merge(metadata, on=KEY_COLUMNS, how="left", validate="one_to_one")
    matched_review_rows = int(master["_기존검토순서"].notna().sum())
    if matched_review_rows != len(review):
        latest_keys = set(latest[KEY_COLUMNS].itertuples(index=False, name=None))
        missing = review.loc[
            [
                key not in latest_keys
                for key in review[KEY_COLUMNS].itertuples(index=False, name=None)
            ],
            KEY_COLUMNS,
        ]
        raise ValueError(
            "기존 검토본 키가 최신 정제본에 없습니다: "
            f"matched={matched_review_rows}/{len(review)}, "
            f"missing={missing.head(10).to_dict('records')}"
        )
    master = master.merge(old_text, on=KEY_COLUMNS, how="left", validate="one_to_one")

    rows_2021 = master["연도"].eq(2021)
    new_rows = master["_기존검토순서"].isna()
    if not new_rows.eq(rows_2021).all():
        unexpected = master.loc[new_rows.ne(rows_2021), KEY_COLUMNS]
        raise ValueError(
            "기존 검토본과 최신 정제본의 차이가 2021년 행 집합과 다릅니다: "
            f"{unexpected.head(10).to_dict('records')}"
        )
    label_columns = labels_2021[KEY_COLUMNS + ["대영역", "세부영역"]].rename(
        columns={"대영역": "2021_대영역", "세부영역": "2021_세부영역"}
    )
    master = master.merge(label_columns, on=KEY_COLUMNS, how="left", validate="one_to_one")
    master.loc[rows_2021, "검토_대영역"] = master.loc[rows_2021, "2021_대영역"]
    master.loc[rows_2021, "검토_세부영역"] = master.loc[rows_2021, "2021_세부영역"]
    master.loc[rows_2021, "검토상태"] = "확정"
    master.loc[rows_2021, "저신뢰_검토대상"] = False

    for column in ("예측_대영역", "예측_세부영역", "예측_신뢰도"):
        master.loc[rows_2021, column] = pd.NA
    master["라벨출처"] = "재정팀 전체연도 검토본"
    master.loc[rows_2021, "라벨출처"] = "2021 시도별 수작업 라벨"

    master["최신정제문_갱신여부"] = changed_cleaned_flags(
        master["기존검토본_주요내용_정제"],
        master["주요내용_정제"],
        new_rows,
    )
    impact_keys = set(semantic_impact[KEY_COLUMNS].itertuples(index=False, name=None))
    master["#72_정제변경"] = [
        key in impact_keys for key in master[KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    master["#72_재확인대상"] = (
        master["#72_정제변경"] & ~rows_2021 & master["검토상태"].isin(["확정", "수정", "보류"])
    )
    master = master.drop(columns=["2021_대영역", "2021_세부영역"])

    if master[KEY_COLUMNS].isna().any().any():
        raise ValueError("통합 마스터에 키 결측이 있습니다.")
    if master.loc[rows_2021, ["검토_대영역", "검토_세부영역"]].isna().any().any():
        raise ValueError("2021년 수작업 라벨 연결 누락이 있습니다.")
    if master["#72_정제변경"].sum() != 79:
        raise ValueError(f"#72 정제변경 표시는 79건이어야 합니다: {master['#72_정제변경'].sum()}")
    master = master.sort_values("_기존검토순서", na_position="last", kind="stable")
    return master.drop(columns="_기존검토순서").reset_index(drop=True)


def normalize_business_name(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value)).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def add_similarity_groups(master: pd.DataFrame, threshold: float = 0.82) -> pd.DataFrame:
    """세부사업명 문자 TF-IDF 최근접 유사도를 이용해 검토용 그룹을 만든다."""
    output = master.copy()
    normalized = output["세부사업명"].map(normalize_business_name)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
    matrix = vectorizer.fit_transform(normalized)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)

    union_find = _UnionFind(len(output))
    nearest_similarity = 1 - distances[:, 1]
    nearest_index = indices[:, 1]
    for index, (other, similarity) in enumerate(
        zip(nearest_index, nearest_similarity, strict=True)
    ):
        if similarity >= threshold:
            union_find.union(index, int(other))

    roots = [union_find.find(index) for index in range(len(output))]
    root_order = {root: order + 1 for order, root in enumerate(dict.fromkeys(roots))}
    output["유사사업그룹ID"] = [f"G{root_order[root]:05d}" for root in roots]
    output["최근접사업명"] = output.iloc[nearest_index]["세부사업명"].to_numpy()
    output["최근접유사도"] = nearest_similarity.round(6)
    group_sizes = output.groupby("유사사업그룹ID")["유사사업그룹ID"].transform("size")
    output["유사그룹행수"] = group_sizes
    output["유사검토대상"] = group_sizes.gt(1)
    return output.sort_values(
        ["유사검토대상", "유사사업그룹ID", "최근접유사도", "연도", "지역", "원본행"],
        ascending=[False, True, False, True, True, True],
    ).reset_index(drop=True)


def build_qa(master: pd.DataFrame, labels_changed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"검사항목": "전체 행", "결과": len(master), "기대값": 57_979},
            {
                "검사항목": "연도·지역·원본행 중복",
                "결과": int(master.duplicated(KEY_COLUMNS).sum()),
                "기대값": 0,
            },
            {"검사항목": "2021 행", "결과": int(master["연도"].eq(2021).sum()), "기대값": 7_301},
            {
                "검사항목": "2021 신규 라벨 변경",
                "결과": labels_changed,
                "기대값": labels_changed,
            },
            {
                "검사항목": "#72 정제변경 표시",
                "결과": int(master["#72_정제변경"].sum()),
                "기대값": 79,
            },
            {
                "검사항목": "2021 라벨 결측",
                "결과": int(
                    master.loc[master["연도"].eq(2021), ["검토_대영역", "검토_세부영역"]]
                    .isna()
                    .any(axis=1)
                    .sum()
                ),
                "기대값": 0,
            },
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-workbook",
        type=Path,
        default=Path(
            "data/interim/영역분류_라벨링/재정팀_검토본/현재검토본/"
            "2016_2020_2022_2024_TFIDF_영역분류_검토_v2_2016복구반영.xlsx"
        ),
    )
    parser.add_argument(
        "--labels-2021-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/재정팀_검토본/2021_시도별_라벨원본"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/재정팀_검토본/통합작업"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review, _ = read_review_workbook(args.review_workbook)
    labels_2021, _ = consolidate_labels(args.labels_2021_dir, args.data_root)
    targets = _read_csv(
        args.data_root / "영역분류_라벨링/TFIDF_분류대상/"
        "2016_2020_2022_2024_TFIDF_분류대상_통합.csv"
    )
    semantic_impact = _read_csv(args.reports_root / "20260731_LLM_정제변경_라벨예산_영향검토.csv")

    old_labels = _read_csv(
        args.data_root / "영역분류_라벨링/2021/통합/2021_17개시도_영역분류_라벨_통합.csv"
    )
    old_labels = normalize_review_keys(old_labels)
    new_labels = normalize_review_keys(labels_2021)
    comparison = old_labels[KEY_COLUMNS + ["대영역", "세부영역"]].merge(
        new_labels[KEY_COLUMNS + ["대영역", "세부영역"]],
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        suffixes=("_기존", "_신규"),
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        missing = comparison.loc[comparison["_merge"].ne("both"), [*KEY_COLUMNS, "_merge"]]
        raise ValueError(f"2021 기존·신규 라벨 키 집합 불일치: {missing.to_dict('records')}")
    comparison = comparison.drop(columns="_merge")
    label_changes = comparison.loc[
        comparison["대영역_기존"].ne(comparison["대영역_신규"])
        | comparison["세부영역_기존"].ne(comparison["세부영역_신규"])
    ].copy()

    master = build_master(review, labels_2021, targets, semantic_impact)
    similarity = add_similarity_groups(master)
    qa = build_qa(master, len(label_changes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(
        args.output_dir / "전체_57979건_원본순서.csv",
        index=False,
        encoding="utf-8-sig",
    )
    similarity.to_csv(
        args.output_dir / "전체_57979건_유사사업명_검토용.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_changes.to_csv(
        args.output_dir / "2021_라벨변경_106건.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qa.to_csv(args.output_dir / "통합_QA.csv", index=False, encoding="utf-8-sig")
    print(f"전체 통합: {len(master):,}건")
    print(f"2021 라벨 변경: {len(label_changes):,}건")
    print(f"#72 정제변경: {int(master['#72_정제변경'].sum()):,}건")
    print(f"유사사업명 검토 그룹 행: {int(similarity['유사검토대상'].sum()):,}건")


if __name__ == "__main__":
    main()
