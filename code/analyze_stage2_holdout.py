from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage2_v2_holdout")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert len(scores) == 2240
assert len(selection) == 490
assert not scores.isna().any().any()
assert not selection.isna().any().any()
assert selection[["logloss_regret", "auc_regret"]].min().min() >= -1e-10

selection["winner_flip"] = selection.logloss_selected_model != selection.logloss_oracle
dataset_summary = (
    selection.groupby(["target_alpha", "dataset"], as_index=False)
    .agg(
        median_logloss_regret=("logloss_regret", "median"),
        median_rank_spearman=("logloss_rank_spearman", "median"),
        winner_flip_rate=("winner_flip", "mean"),
        median_auc_regret=("auc_regret", "median"),
    )
)
dataset_summary.to_csv(root / "dataset_summary.csv", index=False)

zero = dataset_summary[dataset_summary.target_alpha == 0.0]
negative = dataset_summary[dataset_summary.target_alpha == -0.5]
primary_median = zero.median_logloss_regret.median()
primary_count = int((zero.median_logloss_regret >= 0.02).sum())
secondary_count = int((negative.median_logloss_regret >= 0.02).sum())
primary_pass = primary_median >= 0.02 and primary_count >= 4
secondary_pass = secondary_count >= 5
decision = "GO" if primary_pass and secondary_pass else "STOP_OR_REFRAME"

summary = (
    selection.groupby("target_alpha", as_index=False)
    .agg(
        median_logloss_regret=("logloss_regret", "median"),
        median_rank_spearman=("logloss_rank_spearman", "median"),
        winner_flip_rate=("winner_flip", "mean"),
        median_auc_regret=("auc_regret", "median"),
    )
    .sort_values("target_alpha", ascending=False)
)
summary.to_csv(root / "summary.csv", index=False)

lines = [
    "# Stage 2-v2 held-out decision",
    "",
    f"**Decision: {decision}**",
    "",
    f"Primary alpha=0 dataset-median regret: {primary_median:.6f}",
    f"Primary datasets >=0.02: {primary_count}/7",
    f"Secondary alpha=-0.5 datasets >=0.02: {secondary_count}/7",
    "",
    "```text",
    summary.to_string(index=False),
    "```",
    "",
]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
