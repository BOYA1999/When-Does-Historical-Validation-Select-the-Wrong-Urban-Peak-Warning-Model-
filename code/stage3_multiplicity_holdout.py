from pathlib import Path

import pilot
from stage2_v2_holdout import NEW_DATASETS
from stage3_multiplicity import run


pilot.DATASETS.update(NEW_DATASETS)
run(list(NEW_DATASETS), list(range(5)), [1, 4, 8], Path("artifacts/experiment/stage3_multiplicity_holdout"))
