"""#103 구조환경 4대 영역의 2016~2024년 평균으로 17개 시도를 유형화한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "data/processed/structural_index/structural_index_pooled_category_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/analysis"
EXPECTED_CATEGORIES = ["가족·생활", "경제·고용·주거", "보건·안전", "사회·문화"]


def build_region_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """완전 패널을 검증하고 시도별 9개년 대영역 평균을 만든다."""
    required = {"region", "year", "category", "category_score"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"군집 입력 필수 컬럼 누락: {missing}")
    if data[list(required)].isna().any().any():
        raise ValueError("군집 입력 키 또는 점수에 결측이 있습니다.")
    if data.duplicated(["region", "year", "category"]).any():
        raise ValueError("군집 입력 지역·연도·대영역 키 중복")
    if set(data["category"]) != set(EXPECTED_CATEGORIES):
        raise ValueError("구조환경 대영역 구성이 예상 4개와 다릅니다.")
    years = sorted(data["year"].unique())
    if years != list(range(2016, 2025)):
        raise ValueError(f"군집 입력 연도 불일치: {years}")
    counts = data.groupby(["region", "category"]).size()
    if counts.nunique() != 1 or counts.iloc[0] != len(years):
        raise ValueError("지역·대영역별 연도 관측 수가 완전하지 않습니다.")
    profiles = (
        data.groupby(["region", "category"], as_index=False)["category_score"]
        .mean()
        .pivot(index="region", columns="category", values="category_score")
        .reindex(columns=EXPECTED_CATEGORIES)
    )
    if len(profiles) != 17 or profiles.isna().any().any():
        raise ValueError(f"시도 프로파일은 17개 완전행이어야 합니다: {profiles.shape}")
    return profiles


def _canonicalize(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """가족·생활 중심값 오름차순으로 임의 군집번호를 고정한다."""
    order = np.argsort(centers[:, 0])
    mapping = {old: new + 1 for new, old in enumerate(order)}
    canonical = np.array([mapping[label] for label in labels], dtype=int)
    return canonical, centers[order]


def fit_cluster_solution(
    profiles: pd.DataFrame, cluster_count: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """표준화 K-means 군집과 안정성 지표를 반환한다."""
    scaler = StandardScaler().fit(profiles)
    values = scaler.transform(profiles)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=100).fit(values)
    labels, ordered_scaled_centers = _canonicalize(model.labels_, model.cluster_centers_)
    centers = scaler.inverse_transform(ordered_scaled_centers)

    result = profiles.reset_index().copy()
    result[f"군집_{cluster_count}개"] = labels
    for category in EXPECTED_CATEGORIES:
        result[f"{category}_z"] = values[:, profiles.columns.get_loc(category)]

    hierarchical = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward").fit_predict(
        values
    )
    seed_aris = []
    for seed in range(50):
        candidate = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20).fit_predict(
            values
        )
        seed_aris.append(adjusted_rand_score(model.labels_, candidate))
    loo_aris = []
    for index in range(len(profiles)):
        keep = np.arange(len(profiles)) != index
        candidate = KMeans(n_clusters=cluster_count, random_state=42, n_init=100).fit_predict(
            values[keep]
        )
        loo_aris.append(adjusted_rand_score(model.labels_[keep], candidate))

    metrics = {
        "군집수": float(cluster_count),
        "실루엣점수": float(silhouette_score(values, model.labels_)),
        "초기값_ARI_최소": float(min(seed_aris)),
        "초기값_ARI_평균": float(np.mean(seed_aris)),
        "계층군집_ARI": float(adjusted_rand_score(model.labels_, hierarchical)),
        "시도제외_ARI_최소": float(min(loo_aris)),
        "시도제외_ARI_평균": float(np.mean(loo_aris)),
        "최소군집크기": float(pd.Series(labels).value_counts().min()),
        "최대군집크기": float(pd.Series(labels).value_counts().max()),
    }
    center_rows = []
    for cluster_index, center in enumerate(centers, start=1):
        center_rows.append(
            {
                "군집수": cluster_count,
                "군집": cluster_index,
                **dict(zip(EXPECTED_CATEGORIES, center, strict=True)),
            }
        )
    result.attrs["centers"] = pd.DataFrame(center_rows)
    return result, metrics


def build_elbow_diagnostics(profiles: pd.DataFrame, max_clusters: int = 6) -> pd.DataFrame:
    """k=1~max_clusters의 WCSS와 실루엣 점수를 산출한다."""
    if not 2 <= max_clusters < len(profiles):
        raise ValueError("최대 군집 수는 2 이상이고 시도 수보다 작아야 합니다.")
    values = StandardScaler().fit_transform(profiles)
    rows = []
    for cluster_count in range(1, max_clusters + 1):
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=100).fit(values)
        rows.append(
            {
                "군집수": cluster_count,
                "WCSS": float(model.inertia_),
                "실루엣점수": (
                    float(silhouette_score(values, model.labels_)) if cluster_count >= 2 else np.nan
                ),
                "최소군집크기": int(pd.Series(model.labels_).value_counts().min()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_region_profiles(pd.read_csv(args.input))
    solutions = []
    metrics = []
    centers = []
    for cluster_count in (2, 3):
        solution, metric = fit_cluster_solution(profiles, cluster_count)
        center_table = solution.attrs["centers"]
        solution.attrs.clear()
        solutions.append(solution.set_index("region"))
        metrics.append(metric)
        centers.append(center_table)
    combined = solutions[0].join(solutions[1][["군집_3개"]]).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        args.output_dir / "2016-2024_시도별_구조환경_군집.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metrics).to_csv(
        args.output_dir / "2016-2024_구조환경_군집_안정성.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(centers, ignore_index=True).to_csv(
        args.output_dir / "2016-2024_구조환경_군집중심.csv", index=False, encoding="utf-8-sig"
    )
    elbow = build_elbow_diagnostics(profiles)
    elbow.to_csv(
        args.output_dir / "2016-2024_구조환경_군집수_진단.csv", index=False, encoding="utf-8-sig"
    )
    print(combined[["region", "군집_2개", "군집_3개"]].to_string(index=False))
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print(elbow.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
