import gc
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pilot
from stage2_v4_latent import latent_mask


VALIDATION_RHOS = [1.0, 0.5, 0.0, -0.5, -1.0]
OUTPUT = Path("artifacts/experiment/stage3_stress_selection")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in pilot.DATASETS:
        x, y = pilot.load_data(dataset_name)
        for seed in [0, 1, 2]:
            x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.4, stratify=y, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
            train = latent_mask(x_train, y_train, 1.0, seed + 10)
            validations = {rho: latent_mask(x_val, y_val, rho, seed + 20) for rho in VALIDATION_RHOS}
            targets = {rho: latent_mask(x_test, y_test, rho, seed + 40) for rho in [1.0, 0.0]}

            for model_name, model in pilot.models(seed).items():
                model.fit(train, y_train)
                for rho, features in validations.items():
                    ll, brier, auc = pilot.metrics(y_val, model.predict_proba(features))
                    rows.append({"dataset": dataset_name, "seed": seed, "model": model_name, "split": "validation", "rho": rho, "log_loss": ll, "brier": brier, "auc": auc})
                for rho, features in targets.items():
                    ll, brier, auc = pilot.metrics(y_test, model.predict_proba(features))
                    rows.append({"dataset": dataset_name, "seed": seed, "model": model_name, "split": "target", "rho": rho, "log_loss": ll, "brier": brier, "auc": auc})
            pd.DataFrame(rows).to_csv(OUTPUT / "model_scores.csv", index=False)
            gc.collect()
            torch.cuda.empty_cache()
            print(dataset_name, seed, "complete", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == 420
    selected = []
    for (dataset, seed), cell in scores.groupby(["dataset", "seed"]):
        validation = cell[cell.split == "validation"].pivot(index="model", columns="rho", values="log_loss")
        targets = cell[cell.split == "target"].pivot(index="model", columns="rho", values="log_loss")
        choices = {"standard": validation[1.0].idxmin(), "neutral_stress": validation[0.0].idxmin(), "worst_case_stress": validation.max(axis=1).idxmin()}
        for selector, model in choices.items():
            for rho in [1.0, 0.0]:
                selected.append({"dataset": dataset, "seed": seed, "selector": selector, "target_rho": rho, "selected_model": model, "oracle": targets[rho].idxmin(), "target_logloss": targets.loc[model, rho], "selection_regret": targets.loc[model, rho] - targets[rho].min()})
    selection = pd.DataFrame(selected)
    selection.to_csv(OUTPUT / "selection_results.csv", index=False)
    summary = selection.groupby(["selector", "target_rho"], as_index=False).agg(median_selection_regret=("selection_regret", "median"), median_target_logloss=("target_logloss", "median"))
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
