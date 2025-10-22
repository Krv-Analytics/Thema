.. _preprocessing:

===================
Data Preprocessing
===================

Planet cleans, encodes, scales, and imputes tabular data, outputting Moon files for downstream analysis.

Basic Usage
-----------

.. code-block:: python

   from thema.multiverse import Planet

   planet = Planet(
       data="/path/to/data.pkl",
       outDir="./outputs/my_run/clean",
       scaler="standard",
       encoding="one_hot",
       imputeColumns="auto",
       imputeMethods="auto",
       numSamples=1,
       seeds="auto"
   )
   planet.fit()

Parameters
----------

**data** : str or Path
    Absolute path to input file. Supported formats: CSV, pickle (``.pkl``), parquet, Excel (``.xlsx``).

**outDir** : str or Path
    Absolute path for output Moon files. Creates directory if missing.

**dropColumns** : list of str, optional
    Column names to remove before preprocessing. Use for IDs, timestamps, or non-predictive features.

**imputeColumns** : list of str or "auto"
    Columns requiring imputation. ``"auto"`` detects all columns with missing values.

**imputeMethods** : list of str or "auto"
    Imputation strategy per column in ``imputeColumns``. Must align in length. Options:

    - ``"mean"`` - Column mean (numeric only)
    - ``"median"`` - Column median (numeric only)
    - ``"mode"`` - Most frequent value
    - ``"sampleNormal"`` - Sample from fitted normal distribution (numeric only)
    - ``"sampleCategorical"`` - Sample from category distribution
    - ``"zeros"`` - Fill with zeros

    ``"auto"`` selects ``sampleNormal`` for numeric columns, ``sampleCategorical`` for categorical.

**scaler** : str or None
    Feature scaling method. Applied after encoding and imputation. Options:

    - ``"standard"`` - Zero mean, unit variance (recommended)
    - ``"minmax"`` - Scale to [0, 1] range
    - ``"robust"`` - Robust to outliers (uses IQR)
    - ``None`` - No scaling

**encoding** : str
    Categorical encoding method:

    - ``"one_hot"`` - Binary columns per category (recommended)
    - ``"label"`` - Integer encoding
    - ``"ordinal"`` - Ordered integer encoding

**numSamples** : int
    Number of imputed datasets per seed. Use >1 only with randomized imputation methods (``sampleNormal``, ``sampleCategorical``) to capture uncertainty.

**seeds** : list of int or "auto"
    Random seeds for reproducibility. Each seed generates ``numSamples`` datasets. ``"auto"`` generates random seeds.

Output
------

Moon files saved as ``moon_<seed>_<sample>.pkl`` in ``outDir``. Each contains:

- ``imputeData``: Preprocessed DataFrame
- ``original_columns``: Column names before encoding
- ``encoding_map``: Mapping for categorical variables

Inspection Methods
------------------

**get_missingData_summary()**
    DataFrame showing missing value counts and percentages per column

**get_na_as_list()**
    List of column names with missing values

**get_recomended_sampling_method()**
    Dict mapping columns to recommended imputation methods

**getParams()**
    Dict of current configuration

**writeParams_toYaml(path)**
    Save configuration to YAML file

Examples
--------

**Manual Configuration**

.. code-block:: python

   planet = Planet(
       data="/data/survey.pkl",
       outDir="./outputs/analysis/clean",
       dropColumns=["id", "timestamp"],
       imputeColumns=["age", "income", "category"],
       imputeMethods=["sampleNormal", "median", "mode"],
       scaler="standard",
       encoding="one_hot",
       seeds=[42, 13, 99],
       numSamples=2
   )
   planet.fit()  # Produces 6 Moon files (3 seeds × 2 samples)

**Auto-Detection**

.. code-block:: python

   planet = Planet(
       data="/data/survey.pkl",
       outDir="./outputs/analysis/clean",
       imputeColumns="auto",
       imputeMethods="auto",
       scaler="standard",
       encoding="one_hot",
       numSamples=1,
       seeds="auto"
   )
   planet.fit()

**YAML Configuration**

In ``params.yaml``:

.. code-block:: yaml

   Planet:
     scaler: standard
     encoding: one_hot
     imputeColumns: auto
     imputeMethods: auto
     numSamples: 1
     seeds: auto

Then:

.. code-block:: python

   planet = Planet(YAML_PATH="params.yaml")
   planet.fit()

Best Practices
--------------

- Use ``"auto"`` for initial runs, then inspect with ``get_missingData_summary()``
- Set ``numSamples=1`` unless using randomized imputation
- Use ``"standard"`` scaling for most ML pipelines
- Drop high-cardinality categoricals before encoding to avoid dimension explosion
- Save raw data as pickle to preserve dtypes across runs
