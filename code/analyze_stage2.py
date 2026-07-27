from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage2_v1")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert len(scores) == 480
assert len(selection) == 105
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

summary = pd.read_csv(root / "summary.csv")
gate_rows = []
for alpha in [0.0, -0.5]:
    per_dataset = dataset_summary[dataset_summary.target_alpha == alpha]
    aggregate = summary[summary.target_alpha == alpha].iloc[0]
    count = int((per_dataset.median_logloss_regret >= 0.02).sum())
    gate_rows.append(
        {
            "target_alpha": alpha,
            "datasets_regret_ge_0.02": count,
            "winner_flip_rate": aggregate.winner_flip_rate,
            "gate_pass": count >= 3 or aggregate.winner_flip_rate >= 0.5,
        }
    )
gate = pd.DataFrame(gate_rows)
gate.to_csv(root / "gate.csv", index=False)

dose = []
for (dataset, seed), cell in selection.groupby(["dataset", "seed"]):
    severity = 1.5 - cell.target_alpha
    correlation = float("nan") if cell.logloss_regret.nunique() < 2 else severity.corr(cell.logloss_regret, method="spearman")
    dose.append(
        {
            "dataset": dataset,
            "seed": seed,
            "severity_regret_spearman": correlation,
        }
    )
dose = pd.DataFrame(dose)
dose.to_csv(root / "dose_response.csv", index=False)

decision = "GO" if gate.gate_pass.any() else "STOP_OR_REDESIGN"
lines = [
    "# Stage 2-v1 decision",
    "",
    f"**Decision: {decision}**",
    "",
    "## Boundary summary",
    "",
    "```text",
    summary.to_string(index=False),
    "```",
    "",
    "## Preregistered gate",
    "",
    "```text",
    gate.to_string(index=False),
    "```",
    "",
    f"Median severity-regret Spearman across defined cells: {dose.severity_regret_spearman.median():.3f}",
    f"Undefined constant-regret cells: {dose.severity_regret_spearman.isna().sum()}/15",
    "",
]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
