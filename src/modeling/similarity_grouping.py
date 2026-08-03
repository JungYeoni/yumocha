"""전국 세부사업명·주요내용 유사도로 검토 행을 그룹핑한다."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scripts.consolidate_2021_area_labels import REGION_ORDER, normalize_text

DEFAULT_NAME_SIMILARITY_THRESHOLD = 0.45
DEFAULT_CONTENT_SIMILARITY_THRESHOLD = 0.3
UNGROUPED_SIMILARITY_SCORE = -1.0

NAME_GROUP_COLUMN = "사업명유사그룹"
NAME_GROUP_SCORE_COLUMN = "사업명유사그룹_평균유사도"
CONTENT_GROUP_COLUMN = "주요내용유사서브그룹"
CONTENT_GROUP_SCORE_COLUMN = "주요내용유사서브그룹_평균유사도"


_PARALLEL_MIN_UNIQUE_TEXTS = 200


def _cluster_texts(texts: list[str], threshold: float) -> tuple[list[int], list[float]]:
    """빈도순 대표 문장과 유사한 텍스트를 같은 그룹으로 묶는다.

    가장 자주 등장한 텍스트를 대표로 먼저 선택하고, 대표와 임계값 이상으로
    유사한 아직 미배정 텍스트만 같은 그룹에 넣는다. 약한 유사 관계가 사슬처럼
    이어져 무관한 사업 대부분이 하나의 거대 그룹이 되는 것을 방지한다.

    반환값은 (원래 순서에 대응하는 그룹 번호, 대표와의 평균 유사도)이다.

    실제 21~24년 데이터(고유 세부사업명 26,250개)로 측정한 결과, 유사도
    계산 자체(희소 코사인 행렬 연산)는 26,250개를 한 번에 처리해도 2~3초면
    끝난다. 병목은 `assign_similarity_groups`가 이 함수를 사업명 그룹마다
    (실측 6,146회) 반복 호출할 때 매번 `n_jobs=-1`로 병렬 백엔드를 새로
    띄우는 오버헤드였다(작은 호출 1,000회 기준 15초 → `n_jobs=1`이면
    0.25초). 그래서 텍스트 수가 적을 때는 병렬화를 끄고, 실제로 클 때만
    켠다.
    """
    if len(texts) == 1:
        return [0], [UNGROUPED_SIMILARITY_SCORE]
    if not any(texts):
        return list(range(len(texts))), [UNGROUPED_SIMILARITY_SCORE] * len(texts)

    positions_by_text: dict[str, list[int]] = {}
    empty_positions: list[int] = []
    for position, text in enumerate(texts):
        if text:
            positions_by_text.setdefault(text, []).append(position)
        else:
            empty_positions.append(position)

    unique_texts = list(positions_by_text)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform(unique_texts)
    n_jobs = -1 if len(unique_texts) >= _PARALLEL_MIN_UNIQUE_TEXTS else 1
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=n_jobs).fit(matrix)
    distances, indices = neighbors.radius_neighbors(
        matrix,
        radius=1 - threshold,
        return_distance=True,
        sort_results=True,
    )

    anchor_order = sorted(
        range(len(unique_texts)),
        key=lambda index: (-len(positions_by_text[unique_texts[index]]), unique_texts[index]),
    )
    unique_group_ids = [-1] * len(unique_texts)
    group_member_scores: dict[int, list[float]] = {}
    next_group_id = 0
    for anchor in anchor_order:
        if unique_group_ids[anchor] >= 0:
            continue
        group_id = next_group_id
        next_group_id += 1
        unique_group_ids[anchor] = group_id
        scores_for_group = [1.0] * len(positions_by_text[unique_texts[anchor]])
        for distance, neighbor in zip(distances[anchor], indices[anchor], strict=True):
            neighbor = int(neighbor)
            if unique_group_ids[neighbor] >= 0:
                continue
            unique_group_ids[neighbor] = group_id
            scores_for_group.extend(
                [float(1 - distance)] * len(positions_by_text[unique_texts[neighbor]])
            )
        group_member_scores[group_id] = scores_for_group

    group_ids = [-1] * len(texts)
    for unique_index, text in enumerate(unique_texts):
        for position in positions_by_text[text]:
            group_ids[position] = unique_group_ids[unique_index]
    for position in empty_positions:
        group_ids[position] = next_group_id
        group_member_scores[next_group_id] = [1.0]
        next_group_id += 1

    group_counts = Counter(group_ids)
    group_scores = {
        group_id: (
            float(np.mean(member_scores))
            if group_counts[group_id] > 1
            else UNGROUPED_SIMILARITY_SCORE
        )
        for group_id, member_scores in group_member_scores.items()
    }
    scores = [group_scores[group_id] for group_id in group_ids]
    return group_ids, scores


def assign_similarity_groups(
    frame: pd.DataFrame,
    *,
    region_column: str = "지역",
    name_column: str = "세부사업명",
    content_column: str = "주요내용_정제",
    name_similarity_threshold: float = DEFAULT_NAME_SIMILARITY_THRESHOLD,
    content_similarity_threshold: float = DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """전국 세부사업명 유사도(1차)·주요내용 유사도(2차)로 그룹을 부여하고 정렬한다.

    반환되는 프레임은 1차 그룹(직접 연결 유사도 평균이 높은 순) →
    지역(REGION_ORDER 순) → 연도 → 원본행 순으로 정렬돼 있어
    ``create_review_workbook(..., preserve_order=True)``에 바로 넘길 수 있다.
    주요내용 서브그룹은 같은 사업명 그룹 안의 보조 검토 정보로 함께 제공한다.

    임계값은 검증된 값이 아니라 이번 작업을 위해 새로 정한 기본값이다.
    분류기 저신뢰 임계값(0.5, 확률)과는 척도가 다른 코사인 유사도이므로
    그대로 재사용하지 않았다. 실제 데이터로 결과를 본 뒤 조정이 필요할 수 있다.
    """
    required = {region_column, name_column, content_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"유사도 그룹핑 필수 열 누락: {missing}")
    if frame.empty:
        raise ValueError("그룹핑할 데이터가 비어 있습니다.")
    if not 0 < name_similarity_threshold <= 1:
        raise ValueError("사업명 유사도 임계값은 0과 1 사이여야 합니다.")
    if not 0 < content_similarity_threshold <= 1:
        raise ValueError("주요내용 유사도 임계값은 0과 1 사이여야 합니다.")

    normalized_name = frame[name_column].map(normalize_text)
    if normalized_name.eq("").any():
        raise ValueError("세부사업명이 비어 있는 행이 있습니다.")

    working = frame.copy()
    working["_사업명정규화"] = normalized_name
    working["_주요내용정규화"] = frame[content_column].map(normalize_text)

    unknown_regions = sorted(set(working[region_column]) - set(REGION_ORDER))
    if unknown_regions:
        raise ValueError(f"REGION_ORDER에 없는 지역이 있습니다: {unknown_regions}")

    working = working.reset_index(drop=True)
    name_group_ids, name_scores = _cluster_texts(
        working["_사업명정규화"].tolist(), name_similarity_threshold
    )
    working[NAME_GROUP_COLUMN] = [f"N{group_id:05d}" for group_id in name_group_ids]
    working[NAME_GROUP_SCORE_COLUMN] = name_scores

    content_group_labels: list[str] = [""] * len(working)
    content_scores: list[float] = [UNGROUPED_SIMILARITY_SCORE] * len(working)
    for name_group, group_positions in working.groupby(NAME_GROUP_COLUMN).groups.items():
        positions = list(group_positions)
        group_content_ids, group_content_scores = _cluster_texts(
            working.loc[positions, "_주요내용정규화"].tolist(),
            content_similarity_threshold,
        )
        for offset, position in enumerate(positions):
            content_group_labels[position] = f"{name_group}-C{group_content_ids[offset]:03d}"
            content_scores[position] = group_content_scores[offset]

    working[CONTENT_GROUP_COLUMN] = content_group_labels
    working[CONTENT_GROUP_SCORE_COLUMN] = content_scores
    region_rank = {region: rank for rank, region in enumerate(REGION_ORDER)}
    working["_지역순서"] = working[region_column].map(region_rank)

    ordered = working.sort_values(
        [
            NAME_GROUP_SCORE_COLUMN,
            NAME_GROUP_COLUMN,
            "_지역순서",
            "연도",
            "원본행",
            CONTENT_GROUP_SCORE_COLUMN,
            CONTENT_GROUP_COLUMN,
        ],
        ascending=[False, True, True, True, True, False, True],
        kind="stable",
    )
    return ordered.drop(columns=["_사업명정규화", "_주요내용정규화", "_지역순서"]).reset_index(
        drop=True
    )
