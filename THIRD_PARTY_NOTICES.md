# Third-party notices and redistribution boundaries

## CityLearn

Source: <https://github.com/citylearn-project/CityLearn>, release v2.5.0, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`.

The checked source repository includes an MIT license. This code-only package does not redistribute CityLearn data or derived experimental outputs. Users must retrieve the official release directly.

## OpenML and scikit-learn datasets

The package stores only dataset identifiers in source code and documentation. OpenML entries are downloaded by data ID through the OpenML/scikit-learn interface. Users remain responsible for the terms attached to each source dataset.

## TabPFN v2

The code calls `TabPFNClassifier.create_default_for_version(ModelVersion.V2)` and does not redistribute checkpoints. The Prior Labs project states that TabPFN-2 code and weights use the Prior Labs License, derived from Apache 2.0 with an additional attribution requirement. Obtain and accept the current license directly from Prior Labs before downloading weights: <https://github.com/PriorLabs/TabPFN> and <https://priorlabs.ai/tabpfn-license>.

Package versions 6.0 and later changed the default model to later non-commercial checkpoints; this study avoids that ambiguity by explicitly requesting `ModelVersion.V2`.

## Python libraries

NumPy, pandas, SciPy, scikit-learn, OpenML, Matplotlib, PyTorch, and their transitive dependencies retain their own licenses. They are named in the environment files but are not redistributed here.
