from pathlib import Path
from shutil import copy2

import pandas as pd

import pilot
from stage2_boundary import run


ROOT = Path("artifacts/experiment")
ORIGINAL = ROOT / "stage4_original10"
UNIFIED = ROOT / "stage4_unified"


def combine():
    UNIFIED.mkdir(parents=True, exist_ok=True)
    score_parts = [
        pd.read_csv(ORIGINAL / "model_scores.csv"),
        pd.read_csv(ROOT / "stage2_v2_holdout" / "model_scores.csv"),
    ]
    selection_parts = [
        pd.read_csv(ORIGINAL / "selection_results.csv"),
        pd.read_csv(ROOT / "stage2_v2_holdout" / "selection_results.csv"),
    ]
    scores = pd.concat(score_parts, ignore_index=True)
    selection = pd.concat(selection_parts, ignore_index=True)
    score_key = ["dataset", "seed", "model", "split", "target_alpha"]
    selection_key = ["dataset", "seed", "target_alpha"]
    assert len(scores) == 3840
    assert len(selection) == 840
    assert scores.duplicated(score_key).sum() == 0
    assert selection.duplicated(selection_key).sum() == 0
    assert scores.dataset.nunique() == 12
    assert set(scores.seed) == set(range(10))
    scores.to_csv(UNIFIED / "model_scores.csv", index=False)
    selection.to_csv(UNIFIED / "selection_results.csv", index=False)
    summary = (
        selection.assign(winner_flip=selection.logloss_selected_model != selection.logloss_oracle)
        .groupby("target_alpha", as_index=False)
        .agg(
            median_logloss_regret=("logloss_regret", "median"),
            median_rank_spearman=("logloss_rank_spearman", "median"),
            winner_flip_rate=("winner_flip", "mean"),
            median_auc_regret=("auc_regret", "median"),
        )
        .sort_values("target_alpha", ascending=False)
    )
    summary.to_csv(UNIFIED / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    score_path = ORIGINAL / "model_scores.csv"
    if not score_path.exists():
        copy2(ROOT / "stage2_v1" / "model_scores.csv", score_path)
    run(list(pilot.DATASETS), list(range(10)), ORIGINAL)
    combine()

