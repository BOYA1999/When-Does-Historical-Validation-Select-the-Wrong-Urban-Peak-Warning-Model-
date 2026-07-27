from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage2_v3_proxy")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert len(scores) == 600
assert len(selection) == 120
assert not scores.isna().any().any()
assert not selection.isna().any().any()
assert selection.logloss_regret.min() >= -1e-10

selection["winner_flip"] = selection.selected_model != selection.oracle
dataset_summary = selection.groupby(["family", "target_alpha", "dataset"], as_index=False).agg(median_logloss_regret=("logloss_regret", "median"), median_rank_spearman=("rank_spearman", "median"), winner_flip_rate=("winner_flip", "mean"))
dataset_summary.to_csv(root / "dataset_summary.csv", index=False)

zero = dataset_summary[dataset_summary.target_alpha == 0.0]
gate = zero.groupby("family", as_index=False).agg(dataset_median_regret=("median_logloss_regret", "median"), datasets_regret_ge_0_02=("median_logloss_regret", lambda x: int((x >= 0.02).sum())))
gate["gate_pass"] = (gate.family == "withheld_proxy") & (gate.datasets_regret_ge_0_02 >= 3)
gate.to_csv(root / "gate.csv", index=False)
decision = "GO_HELDOUT_PROXY" if gate.gate_pass.any() else "KEEP_OUTCOME_STRESS_ONLY"

summary = pd.read_csv(root / "summary.csv")
lines = ["# Stage 2-v3 proxy decision", "", f"**Decision: {decision}**", "", "```text", gate.to_string(index=False), "```", "", "```text", summary.to_string(index=False), "```", ""]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
