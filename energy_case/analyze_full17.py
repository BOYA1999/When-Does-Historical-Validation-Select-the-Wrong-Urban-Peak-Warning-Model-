import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "energy_case" / "gate_17_buildings"


def bootstrap_median(values, seed=20260723, draws=10000):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def main():
    scores = pd.read_csv(DATA / "model_scores.csv")
    diagnostics = pd.read_csv(DATA / "diagnostics.csv")
    selection = pd.read_csv(DATA / "selection.csv")
    buildings = pd.read_csv(DATA / "building_summary.csv")
    assert len(scores) == 17 * 2 * 4 * 3
    assert len(selection) == 17 * 2
    assert len(buildings) == 17
    assert not scores.isna().any().any()
    assert scores.duplicated(["building", "fold", "model", "split"]).sum() == 0
    excess = buildings["excess_regret"].to_numpy()
    summary = {
        "buildings": 17,
        "building_folds": 34,
        "model_score_rows": len(scores),
        "median_excess_regret": float(np.median(excess)),
        "median_excess_regret_bootstrap_95": bootstrap_median(excess),
        "positive_excess_buildings": int((excess > 0).sum()),
        "one_sided_sign_test_p_all_positive": float(0.5 ** len(excess)),
        "shifted_regret_ge_0.02_buildings": int((buildings["shifted_regret"] >= 0.02).sum()),
        "matched_regret_ge_0.02_buildings": int((buildings["matched_regret"] >= 0.02).sum()),
        "median_matched_regret": float(buildings["matched_regret"].median()),
        "median_shifted_regret": float(buildings["shifted_regret"].median()),
        "shifted_winner_flip_rate": float(selection["shifted_winner_flip"].mean()),
        "median_shifted_rank_spearman": float(buildings["shifted_rank_spearman"].median()),
        "median_missing_count_label_corr_matched": float(diagnostics[diagnostics.split == "matched_target"].missing_count_label_corr.median()),
        "median_missing_count_label_corr_shifted": float(diagnostics[diagnostics.split == "shifted_target"].missing_count_label_corr.median()),
        "target_peak_rate_range": [
            float(diagnostics[diagnostics.split == "shifted_target"].peak_rate.min()),
            float(diagnostics[diagnostics.split == "shifted_target"].peak_rate.max()),
        ],
        "selected_model_counts": selection["selected_model"].value_counts().to_dict(),
        "shifted_oracle_counts": selection["shifted_oracle"].value_counts().to_dict(),
    }
    (DATA / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
