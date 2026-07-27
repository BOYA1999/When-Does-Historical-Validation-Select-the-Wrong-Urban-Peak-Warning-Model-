from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage3_multiplicity_holdout")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert not scores.isna().any().any()
assert not selection.isna().any().any()
assert selection.logloss_regret.min() >= -1e-10
assert selection.groupby(["dataset", "seed"]).ngroups == 35

correlations = []
for (dataset, seed), cell in selection.groupby(["dataset", "seed"]):
    correlation = float("nan") if cell.logloss_regret.nunique() < 2 else cell.linked.corr(cell.logloss_regret, method="spearman")
    correlations.append({"dataset": dataset, "seed": seed, "multiplicity_regret_spearman": correlation})
correlations = pd.DataFrame(correlations)
correlations.to_csv(root / "dose_response.csv", index=False)
defined = correlations.multiplicity_regret_spearman.dropna()
median_correlation = defined.median()
positive_cells = int((defined > 0).sum())
gate_pass = median_correlation >= 0.7 and positive_cells >= 24
decision = "CONFIRMED" if gate_pass else "NOT_CONFIRMED"
summary = selection.groupby("linked", as_index=False).agg(median_logloss_regret=("logloss_regret", "median"), cells=("logloss_regret", "size"))
summary.to_csv(root / "summary.csv", index=False)
lines = ["# Stage 3 multiplicity held-out decision", "", f"**Decision: {decision}**", "", f"Median defined Spearman: {median_correlation:.3f}", f"Positive defined cells: {positive_cells}/{len(defined)}", f"Undefined cells: {correlations.multiplicity_regret_spearman.isna().sum()}/35", "", "```text", summary.to_string(index=False), "```", ""]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
