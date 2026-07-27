import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "citylearn_v2.5.0" / "data" / "datasets" / "citylearn_challenge_2022_phase_all"
OUTPUT_5 = ROOT / "artifacts" / "energy_case" / "gate_5_buildings"
OUTPUT_17 = ROOT / "artifacts" / "energy_case" / "gate_17_buildings"
CALENDAR = ["month", "hour", "day_type", "daylight_savings_status"]
BUILDINGS = [1, 2, 3, 4, 5]


def calibrated_probability(score, rate=0.20):
    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(score + mid)))
        if p.mean() < rate:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(score + (lo + hi) / 2)))


def mask_telemetry(x, y, alpha, uniforms):
    out = x.copy()
    telemetry = np.arange(len(CALENDAR), x.shape[1])
    score = (y - y.mean()) / (y.std() + 1e-8)
    probability = calibrated_probability(alpha * score)
    mask = uniforms < probability[:, None]
    row_index, column_index = np.where(mask)
    out[row_index, telemetry[column_index]] = np.nan
    missing_count = mask.sum(axis=1)
    return out, {
        "missing_rate": float(mask.mean()),
        "missing_count_label_corr": float(np.corrcoef(missing_count, y)[0, 1]),
    }


def load_building(number):
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    name = f"Building_{number}"
    building = pd.read_csv(DATA / schema["buildings"][name]["energy_simulation"])
    weather = pd.read_csv(DATA / schema["buildings"][name]["weather"])
    pv_power = schema["buildings"][name]["pv"]["attributes"]["nominal_power"]
    pv = building["solar_generation"] * pv_power / 1000.0
    net = building["non_shiftable_load"] - pv
    frame = building[CALENDAR].copy()
    frame["net_load_t"] = net
    for lag in (1, 2, 3, 24, 48, 168):
        frame[f"net_load_lag_{lag}"] = net.shift(lag)
    frame["non_shiftable_load_t"] = building["non_shiftable_load"]
    frame["pv_generation_t"] = pv
    for column in (
        "outdoor_dry_bulb_temperature",
        "outdoor_relative_humidity",
        "diffuse_solar_irradiance",
        "direct_solar_irradiance",
        "outdoor_dry_bulb_temperature_predicted_1",
        "outdoor_relative_humidity_predicted_1",
        "diffuse_solar_irradiance_predicted_1",
        "direct_solar_irradiance_predicted_1",
    ):
        frame[column] = weather[column]
    frame["next_net_load"] = net.shift(-1)
    frame = frame.dropna().reset_index(drop=True)
    return frame.drop(columns="next_net_load").to_numpy(dtype="float32"), frame["next_net_load"].to_numpy(dtype="float32"), list(frame.drop(columns="next_net_load").columns)


def models(seed):
    return {
        "logistic_indicator": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
        "random_forest_indicator": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=4, random_state=seed),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "tabpfn_v2": TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=1,
            device="cpu",
            fit_mode="low_memory",
            memory_saving_mode=True,
            ignore_pretraining_limits=True,
            random_state=seed,
            show_progress_bar=False,
        ),
    }


def metrics(y, probability):
    p = np.clip(probability[:, 1], 1e-7, 1 - 1e-7)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }


def fold_slices(n):
    edges = [0, n // 2, 2 * n // 3, 5 * n // 6, n]
    return [
        (slice(edges[0], edges[1]), slice(edges[1], edges[2]), slice(edges[2], edges[3])),
        (slice(edges[0], edges[2]), slice(edges[2], edges[3]), slice(edges[3], edges[4])),
    ]


def run(buildings, fold_count, output, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / ("smoke_model_scores.csv" if smoke else "model_scores.csv")
    diagnostic_path = output / ("smoke_diagnostics.csv" if smoke else "diagnostics.csv")
    rows = pd.read_csv(score_path).to_dict("records") if score_path.exists() and not smoke else []
    diagnostics = pd.read_csv(diagnostic_path).to_dict("records") if diagnostic_path.exists() and not smoke else []
    completed = set()
    if rows:
        counts = pd.DataFrame(rows).groupby(["building", "fold"]).size()
        completed = set(counts[counts == 12].index)
    for building in buildings:
        x, load, feature_names = load_building(building)
        assert feature_names[: len(CALENDAR)] == CALENDAR
        for fold, (train_slice, val_slice, target_slice) in enumerate(fold_slices(len(x))[:fold_count]):
            if (building, fold) in completed:
                continue
            seed = 20260723 + building * 100 + fold
            threshold = float(np.quantile(load[train_slice], 0.90))
            y_train = (load[train_slice] > threshold).astype(int)
            y_val = (load[val_slice] > threshold).astype(int)
            y_target = (load[target_slice] > threshold).astype(int)
            rng = np.random.default_rng(seed)
            telemetry_count = x.shape[1] - len(CALENDAR)
            train, train_diag = mask_telemetry(x[train_slice], y_train, 1.5, rng.random((len(y_train), telemetry_count)))
            val, val_diag = mask_telemetry(x[val_slice], y_val, 1.5, rng.random((len(y_val), telemetry_count)))
            target_uniforms = rng.random((len(y_target), telemetry_count))
            matched, matched_diag = mask_telemetry(x[target_slice], y_target, 1.5, target_uniforms)
            shifted, shifted_diag = mask_telemetry(x[target_slice], y_target, 0.0, target_uniforms)
            for split, labels, diag in (
                ("train", y_train, train_diag),
                ("validation", y_val, val_diag),
                ("matched_target", y_target, matched_diag),
                ("shifted_target", y_target, shifted_diag),
            ):
                diagnostics.append({"building": building, "fold": fold, "split": split, "threshold": threshold, "n": len(labels), "peak_rate": labels.mean(), **diag})
            for model_name, model in models(seed).items():
                model.fit(train, y_train)
                for split, features, labels in (
                    ("validation", val, y_val),
                    ("matched_target", matched, y_target),
                    ("shifted_target", shifted, y_target),
                ):
                    rows.append({"building": building, "fold": fold, "model": model_name, "split": split, **metrics(labels, model.predict_proba(features))})
                del model
                gc.collect()
            print(f"Building_{building} fold {fold} complete", flush=True)
            pd.DataFrame(rows).to_csv(score_path, index=False)
            pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
            if smoke:
                break
        if smoke:
            break

    scores = pd.DataFrame(rows)
    scores.to_csv(score_path, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
    selected = []
    for (building, fold), cell in scores.groupby(["building", "fold"]):
        validation = cell[cell.split == "validation"].set_index("model")
        choice = validation.log_loss.idxmin()
        record = {"building": building, "fold": fold, "selected_model": choice}
        for condition in ("matched_target", "shifted_target"):
            target = cell[cell.split == condition].set_index("model")
            prefix = condition.replace("_target", "")
            record[f"{prefix}_oracle"] = target.log_loss.idxmin()
            record[f"{prefix}_regret"] = target.loc[choice, "log_loss"] - target.log_loss.min()
            record[f"{prefix}_rank_spearman"] = validation.log_loss.corr(target.log_loss, method="spearman")
            record[f"{prefix}_winner_flip"] = choice != target.log_loss.idxmin()
        record["excess_regret"] = record["shifted_regret"] - record["matched_regret"]
        selected.append(record)
    selection = pd.DataFrame(selected)
    selection.to_csv(output / ("smoke_selection.csv" if smoke else "selection.csv"), index=False)
    if not smoke:
        building_summary = selection.groupby("building", as_index=False).agg(
            matched_regret=("matched_regret", "median"),
            shifted_regret=("shifted_regret", "median"),
            excess_regret=("excess_regret", "median"),
            shifted_rank_spearman=("shifted_rank_spearman", "median"),
            shifted_winner_flip_rate=("shifted_winner_flip", "mean"),
        )
        building_summary.to_csv(output / "building_summary.csv", index=False)
        median_excess = float(building_summary.excess_regret.median())
        median_matched = float(building_summary.matched_regret.median())
        positive = int((building_summary.shifted_regret >= 0.02).sum())
        required = 9 if len(buildings) == 17 else 3
        pass_name = "application_case_supported" if len(buildings) == 17 else "continue_to_17"
        verdict = pass_name if median_excess >= 0.02 and positive >= required and median_matched < 0.02 else "stop_or_debug"
        gate = {"verdict": verdict, "median_excess_regret": median_excess, "median_matched_regret": median_matched, "buildings_shifted_regret_ge_0.02": positive, "required_buildings": required}
        (output / "gate_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full17", action="store_true")
    args = parser.parse_args()
    buildings = [1] if args.smoke else (list(range(1, 18)) if args.full17 else BUILDINGS)
    output = OUTPUT_17 if args.full17 else OUTPUT_5
    run(buildings, 1 if args.smoke else 2, output, args.smoke)
