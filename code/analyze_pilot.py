from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/pilot_v1")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")

assert len(scores) == 540
assert len(selection) == 45
assert not scores.isna().any().any()
assert not selection.isna().any().any()
assert selection[["iid_test_regret", "shifted_test_regret"]].min().min() >= -1e-10


def rank_correlation(a, b):
    return float("nan") if a.nunique() < 2 or b.nunique() < 2 else a.corr(b, method="spearman")

flips = selection.assign(winner_flip=selection.selected_model != selection.shifted_test_oracle)
dataset_summary = (
    flips.groupby(["scenario", "dataset"], as_index=False)
    .agg(
        median_shifted_regret=("shifted_test_regret", "median"),
        median_excess_regret=("excess_regret", "median"),
        median_rank_spearman=("shifted_test_rank_spearman", "median"),
        winner_flip_rate=("winner_flip", "mean"),
    )
)
dataset_summary.to_csv(root / "dataset_summary.csv", index=False)

scenario_summary = (
    selection.assign(winner_flip=selection.selected_model != selection.shifted_test_oracle)
    .groupby("scenario", as_index=False)
    .agg(
        median_iid_regret=("iid_test_regret", "median"),
        median_shifted_regret=("shifted_test_regret", "median"),
        median_excess_regret=("excess_regret", "median"),
        median_rank_spearman=("shifted_test_rank_spearman", "median"),
        winner_flip_rate=("winner_flip", "mean"),
    )
)
counts = dataset_summary.groupby("scenario").median_shifted_regret.apply(lambda x: int((x >= 0.02).sum()))
scenario_summary["datasets_regret_ge_0.02"] = scenario_summary.scenario.map(counts)
scenario_summary["gate_pass"] = (scenario_summary["datasets_regret_ge_0.02"] >= 3) | (
    (scenario_summary.median_rank_spearman < 0.5) & (scenario_summary.winner_flip_rate >= 0.5)
)
scenario_summary.to_csv(root / "scenario_summary.csv", index=False)

secondary = []
for keys, cell in scores.groupby(["dataset", "seed", "scenario"]):
    validation = cell[cell.split == "validation"].set_index("model")
    target = cell[cell.split == "shifted_test"].set_index("model")
    selected_model = validation.auc.idxmax()
    secondary.append(
        {
            "dataset": keys[0],
            "seed": keys[1],
            "scenario": keys[2],
            "selected_model": selected_model,
            "target_oracle": target.auc.idxmax(),
            "auc_regret": target.auc.max() - target.loc[selected_model, "auc"],
            "rank_spearman": rank_correlation(validation.auc, target.auc),
        }
    )
secondary = pd.DataFrame(secondary)
secondary.to_csv(root / "secondary_auc_selection.csv", index=False)
secondary_summary = (
    secondary.assign(winner_flip=secondary.selected_model != secondary.target_oracle)
    .groupby("scenario", as_index=False)
    .agg(
        median_auc_regret=("auc_regret", "median"),
        median_rank_spearman=("rank_spearman", "median"),
        undefined_rank_cells=("rank_spearman", lambda x: int(x.isna().sum())),
        winner_flip_rate=("winner_flip", "mean"),
    )
)
secondary_summary.to_csv(root / "secondary_auc_summary.csv", index=False)

decision = "GO" if scenario_summary.gate_pass.any() else "STOP_OR_REDESIGN"
lines = [
    "# Pilot v1 decision",
    "",
    f"**Decision: {decision}**",
    "",
    "## Primary log-loss selection",
    "",
    "```text",
    scenario_summary.to_string(index=False),
    "```",
    "",
    "## Secondary AUROC selection",
    "",
    "```text",
    secondary_summary.to_string(index=False),
    "```",
    "",
]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
