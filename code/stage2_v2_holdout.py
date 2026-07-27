from pathlib import Path

import pilot
from stage2_boundary import run


NEW_DATASETS = {
    "diabetes": 37,
    "sonar": 40,
    "blood_transfusion": 1464,
    "qsar_biodeg": 1494,
    "kc1": 1067,
    "pc1": 1068,
    "hill_valley": 1479,
}

if __name__ == "__main__":
    pilot.DATASETS.update(NEW_DATASETS)
    run(list(NEW_DATASETS), list(range(10)), Path("artifacts/experiment/stage2_v2_holdout"))
