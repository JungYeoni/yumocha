from pathlib import Path

import pandas as pd
import pytest

from src.visualization.structural_index import (
    build_index_summary,
    load_structural_index_artifacts,
    plot_region_component_comparison,
    plot_region_contribution_heatmap,
    validate_structural_index_artifacts,
)


def test_structural_index_artifacts_are_complete_and_nationwide_is_excluded():
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = load_structural_index_artifacts(repo_root)

    validation = validate_structural_index_artifacts(artifacts)
    summary = build_index_summary(artifacts)

    assert validation["final_index_rows"] == 153
    assert len(summary) == 153
    assert "전국" not in set(summary["region"])
    assert summary[["pooled_index", "yearly_index"]].notna().all().all()


def test_structural_index_validation_rejects_nationwide_final_row():
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = load_structural_index_artifacts(repo_root)
    artifacts["pooled_final"] = pd.concat(
        [
            artifacts["pooled_final"],
            artifacts["pooled_final"].iloc[[0]].assign(region="전국"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="전국 행"):
        validate_structural_index_artifacts(artifacts)


def test_region_component_comparison_rejects_unknown_region():
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = load_structural_index_artifacts(repo_root)

    with pytest.raises(ValueError, match="구성점수에 지역"):
        plot_region_component_comparison(artifacts, region="없는 지역")


def test_region_contribution_heatmap_rejects_unknown_region():
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = load_structural_index_artifacts(repo_root)

    with pytest.raises(ValueError, match="구성점수에 지역"):
        plot_region_contribution_heatmap(artifacts, region="없는 지역")
