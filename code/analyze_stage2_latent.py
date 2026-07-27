from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage2_v4_latent")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert len(scores) == 600
assert len(selection) == 75
assert not scores.isna().any().any()
assert not selection.isna().any().any()
assert selection.logloss_regret.min() >= -1e-10

dataset_summary = selection.groupby(["source_rho", "dataset"], as_index=False).agg(median_logloss_regret=("logloss_regret", "median"))
dataset_summary.to_csv(root / "dataset_summary.csv", index=False)
counts = dataset_summary.groupby("source_rho").median_logloss_regret.apply(lambda x: int((x >= 0.02).sum()))
mild = any(counts.get(rho, 0) >= 3 for rho in [0.25, 0.5])
strong = any(counts.get(rho, 0) >= 3 for rho in [0.75, 1.0])
decision = "MILD_BOUNDARY" if mild else "STRONG_ONLY_BOUNDARY" if strong else "NO_BOUNDARY"
gate = pd.DataFrame({"source_rho": counts.index, "datasets_regret_ge_0_02": counts.values})
gate.to_csv(root / "gate.csv", index=False)
summary = pd.read_csv(root / "summary.csv")
lines = ["# Stage 2-v4 latent decision", "", f"**Decision: {decision}**", "", "```text", gate.to_string(index=False), "```", "", "```text", summary.to_string(index=False), "```", ""]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
