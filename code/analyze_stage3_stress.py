from pathlib import Path

import pandas as pd


root = Path("artifacts/experiment/stage3_stress_selection")
scores = pd.read_csv(root / "model_scores.csv")
selection = pd.read_csv(root / "selection_results.csv")
assert len(scores) == 420
assert len(selection) == 90
assert not scores.isna().any().any()
assert not selection.isna().any().any()

shifted = selection[selection.target_rho == 0.0]
iid = selection[selection.target_rho == 1.0]
standard_shifted = shifted[shifted.selector == "standard"].set_index(["dataset", "seed"])
standard_iid = iid[iid.selector == "standard"].set_index(["dataset", "seed"])
rows = []
for selector in ["neutral_stress", "worst_case_stress"]:
    candidate_shifted = shifted[shifted.selector == selector].set_index(["dataset", "seed"])
    candidate_iid = iid[iid.selector == selector].set_index(["dataset", "seed"])
    reduction = standard_shifted.selection_regret - candidate_shifted.selection_regret
    iid_cost = candidate_iid.target_logloss - standard_iid.target_logloss
    dataset_reduction = reduction.groupby("dataset").median()
    median_standard = standard_shifted.selection_regret.median()
    median_candidate = candidate_shifted.selection_regret.median()
    relative_reduction = 0.0 if median_standard == 0 else 1 - median_candidate / median_standard
    rows.append({"selector": selector, "median_standard_shifted_regret": median_standard, "median_candidate_shifted_regret": median_candidate, "relative_regret_reduction": relative_reduction, "datasets_improved": int((dataset_reduction > 0).sum()), "median_iid_logloss_cost": iid_cost.median(), "gate_pass": relative_reduction >= 0.5 and int((dataset_reduction > 0).sum()) >= 3 and iid_cost.median() <= 0.02})

gate = pd.DataFrame(rows)
gate.to_csv(root / "gate.csv", index=False)
decision = "GO_HELDOUT_STRESS" if gate.gate_pass.any() else "DESCRIPTIVE_BOUNDARY_ONLY"
summary = pd.read_csv(root / "summary.csv")
lines = ["# Stage 3 stress-selection decision", "", f"**Decision: {decision}**", "", "```text", gate.to_string(index=False), "```", "", "```text", summary.to_string(index=False), "```", ""]
(root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
