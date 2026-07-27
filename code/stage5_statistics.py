from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path("artifacts/experiment")
SOURCE = ROOT / "stage4_unified"
OUTPUT = ROOT / "stage5_statistics"


def check_equal(left, right, keys):
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def bootstrap_interval(cell, rng, repeats=5000):
    datasets = cell.dataset.unique()
    values = []
    groups = {name: part.logloss_regret.to_numpy() for name, part in cell.groupby("dataset")}
    for _ in range(repeats):
        sampled = rng.choice(datasets, len(datasets), replace=True)
        dataset_medians = [np.median(rng.choice(groups[name], len(groups[name]), replace=True)) for name in sampled]
        values.append(np.median(dataset_medians))
    return np.quantile(values, [0.025, 0.975])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SOURCE / "model_scores.csv")
    selection = pd.read_csv(SOURCE / "selection_results.csv")
    score_key = ["dataset", "seed", "model", "split", "target_alpha"]
    selection_key = ["dataset", "seed", "target_alpha"]
    assert len(scores) == 3840
    assert len(selection) == 840
    assert scores.dataset.nunique() == selection.dataset.nunique() == 12
    assert set(scores.seed) == set(selection.seed) == set(range(10))
    assert scores.duplicated(score_key).sum() == 0
    assert selection.duplicated(selection_key).sum() == 0
    assert set(scores.groupby(["dataset", "seed"]).size()) == {32}
    assert set(selection.groupby(["dataset", "seed"]).size()) == {7}
    assert scores.log_loss.notna().all() and selection.logloss_regret.notna().all()

    original = pd.read_csv(ROOT / "stage4_original10" / "model_scores.csv")
    check_equal(
        original[original.seed < 3],
        pd.read_csv(ROOT / "stage2_v1" / "model_scores.csv"),
        score_key,
    )
    heldout_names = set(pd.read_csv(ROOT / "stage2_v2_holdout" / "model_scores.csv").dataset)
    check_equal(
        scores[scores.dataset.isin(heldout_names)],
        pd.read_csv(ROOT / "stage2_v2_holdout" / "model_scores.csv"),
        score_key,
    )

    dataset_alpha = (
        selection.assign(winner_flip=selection.logloss_selected_model != selection.logloss_oracle)
        .groupby(["dataset", "target_alpha"], as_index=False)
        .agg(
            median_logloss_regret=("logloss_regret", "median"),
            median_rank_spearman=("logloss_rank_spearman", "median"),
            winner_flip_rate=("winner_flip", "mean"),
            median_auc_regret=("auc_regret", "median"),
        )
    )
    dataset_alpha.to_csv(OUTPUT / "dataset_alpha_summary.csv", index=False)

    baseline = dataset_alpha[dataset_alpha.target_alpha == 1.5].set_index("dataset").median_logloss_regret
    rng = np.random.default_rng(20260723)
    rows = []
    for alpha, cell in selection.groupby("target_alpha"):
        dataset_cell = dataset_alpha[dataset_alpha.target_alpha == alpha].set_index("dataset")
        paired = dataset_cell.median_logloss_regret - baseline
        p = 1.0 if np.allclose(paired, 0) else wilcoxon(paired, alternative="greater").pvalue
        low, high = bootstrap_interval(cell, rng)
        rows.append(
            {
                "target_alpha": alpha,
                "median_dataset_regret": dataset_cell.median_logloss_regret.median(),
                "bootstrap_low": low,
                "bootstrap_high": high,
                "datasets_regret_ge_0.02": int((dataset_cell.median_logloss_regret >= 0.02).sum()),
                "median_dataset_rank_spearman": dataset_cell.median_rank_spearman.median(),
                "mean_dataset_winner_flip_rate": dataset_cell.winner_flip_rate.mean(),
                "wilcoxon_vs_alpha_1.5_p": p,
            }
        )
    main_summary = pd.DataFrame(rows).sort_values("target_alpha", ascending=False)
    main_summary.to_csv(OUTPUT / "main_summary.csv", index=False)

    selected_frequency = (
        selection.groupby(["target_alpha", "logloss_selected_model"]).size().rename("cells").reset_index()
    )
    oracle_frequency = selection.groupby(["target_alpha", "logloss_oracle"]).size().rename("cells").reset_index()
    selected_frequency.to_csv(OUTPUT / "selected_model_frequency.csv", index=False)
    oracle_frequency.to_csv(OUTPUT / "oracle_model_frequency.csv", index=False)
    (OUTPUT / "integrity_passed.txt").write_text(
        "All frozen row-count, uniqueness, completeness and source-copy checks passed.\n",
        encoding="utf-8",
    )
    print(main_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
