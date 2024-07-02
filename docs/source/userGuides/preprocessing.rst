.. _preprocessing:

Data Preprocessing
===================

Data preprocessing is a crucial step in the data analysis pipeline. It involves cleaning, transforming, and organizing raw data to prepare it for further analysis. This section outlines the steps involved in preprocessing data using ``Thema``.

Manual Data Exploration
----------------------------

Before diving into automated preprocessing, it's essential to manually explore the data. This step helps you understand the dataset's structure, identify missing values, detect outliers, and gain insights into the data's distribution and relationships. 

``Thema`` provides tools to summarize, clean, scale, encode, and pre-process your data, making it easier to identify patterns and anomalies. These insights can guide your preprocessing decisions and help you choose the right techniques to clean and prepare the data.

Once you've completed the manual data exploration and identified the preprocessing steps required, ``Thema`` supports the creation of a formatted `.yaml` parameter file. This file captures the preprocessing steps and their parameters, making it easy to reproduce the preprocessing pipeline.

The parameter file can include instructions for handling missing values, scaling features, encoding categorical variables, and other preprocessing tasks. By creating this file, you can automate the preprocessing pipeline and apply it consistently to new datasets.

For detailed instructions on creating the `.yaml` parameter file and using it to preprocess your data, refer to the Params YAML Guide. This guide provides examples and best practices for defining preprocessing steps and parameters in ``Thema``.

Walkthrough
----------------------------

Step 1: Data
^^^^^^^^^^^^^^

To demonstrate the data cleaning and preprocessing functionality, let's create a sample DataFrame.

.. ipython:: python

    data = {
        'Name': ['Mercury', 'Venus', 'Earth', 'Jupiter', 'Saturn'],
        'Diameter [1000 km]': [12.7, 121, np.nan, 142.9, 49.5],
        'Has Moon': ['No', 'No', 'Yes', 'Yes', np.nan],
        'Distance from Sun [au]': [0.39, 0.72, 1.00, 5.20, 9.58]}

.. ipython:: python

    import pandas as pd
    df = pd.DataFrame(data)
    df

Step 2: Create a `Planet`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Planet`` class is a core component of the ``Thema`` library, designed for managing tabular data. Its primary purpose is to perturb, label, and navigate existing tabular datasets.

In the context of our demo, the ``Planet`` class serves as the medium for managing data, handling the transition from raw tabular data to scaled, encoded, and complete data. It is particularly useful for datasets with missing values, as it can fill missing values with randomly-sampled data and explore the distribution of possible missing values.

In the following sections, we will explore how to use the ``Planet`` class to preprocess and manage our data effectively.

.. ipython:: python

    from thema.multiverse import Planet

.. ipython:: python

    df.to_pickle("myRawData.pkl")

    data_path = "myRawData.pkl"
    planet = Planet(data = data_path)

    planet


Step 3: Explore Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Planet.get_missingData_summary()`` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_missingData_summary`` function provides a breakdown of missing and complete data in the columns of the 'data' DataFrame. It returns a dictionary containing the following key-value pairs:

- ``'numericMissing'``: Numeric columns with missing values.
- ``'numericComplete'```: Numeric columns without missing values.
- ``'categoricalMissing'``: Categorical columns with missing values.
- ``'categoricalComplete'``: Categorical columns without missing values.

This function is useful for quickly understanding the data quality and identifying columns that may require imputation or other preprocessing steps.

.. ipython:: python

    planet.get_missingData_summary()
    

``get_recommended_sampling_method()`` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_recommended_sampling_method()`` method returns a list of recommended sampling methods for columns with missing values. For numeric columns, "sampleNormal" is recommended, while for non-numeric columns, "sampleCategorical" (most frequent value) is recommended. 

By using these recommended sampling methods, you can effectively impute missing values in your dataset, ensuring that the data remains representative and usable for analysis without the need to drop rows or columns with missing data.

.. ipython:: python

    planet.get_recomended_sampling_method()


``get_na_as_list()`` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_na_as_list`` method returns a list of column names that contain NaN values in the DataFrame. These columns correspond to the columns for which the recommended sampling method should be applied (as returned by ``get_recommended_sampling_method``), providing a convenient way to identify columns that require imputation due to missing values.

By using the `get_na_as_list` method in conjunction with the ``get_recommended_sampling_method`` method, you can efficiently locate and address missing values in your dataset, ensuring that it is properly prepared for further analysis or modeling.

.. ipython:: python

    planet.get_na_as_list()

Step 4: Impute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assign ``imputeMethods`` and ``imputeColumns`` based on the recommended methods for each column ***OR*** define your own lists

Additionally, you can assign ``imputeMethods`` drop, which will drop using Pandas: ``.dropna(axis=0, inplace=True)``

.. ipython:: python

    planet.imputeMethods = planet.get_recomended_sampling_method()
    planet.imputeColumns = planet.get_na_as_list()

**Why Encode and Scale?**

Encoding and scaling are important preprocessing steps in data preparation. Encoding transforms categorical variables into numerical values, making them suitable for machine learning algorithms. Scaling standardizes the range of numerical features, preventing features with large scales from dominating the model.

**Default Configuration**

- **Scaler**: The default scaling method is "standard", which scales features to have a mean of 0 and a variance of 1.
- **Encoding**: The default encoding method is "one_hot" for all categorical variables, which creates binary columns for each category.

**Customization Options**

You can customize the encoding and scaling methods to suit your needs. For scaling, you can choose from an sklearn scaler classes. For encoding, you can select "integer" encoding for ordinal variables or "hash" encoding for high-cardinality categorical variables.

Step 5: Assign an ``outDir``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ``outDir``, or the place to write the imputed, encoded, and scaled data, is where a ``thema.multiverse.Moon`` object (or objects, see guide on Random Imputation) are saved as ``.pkl`` files.

This can also be done in the constructor or params ``.yaml`` file

.. ipython:: python

    planet.outDir = '<YOUR DIRECTORY NAME>'

Step 6: Fit ``Planet``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fitting cleans, encodes, imputes, etc. your data, and then creates and pickles a ``Moon`` object containing your data.

.. note::

    Most objects in Thema have a ``fit()`` method, excecuting a procedure that can be time consuming depending on the size of your dataset.

As the ``Planet.fit()`` method writes a ``thema.multiverse.Moon`` object to a ``.pkl`` file, we will need to read in the file to view the data here. This step is not necessary, but its nice to see what the `fit()` method has done for the purpose of this user guide.

.. code-block:: python

    cleanedData = pd.read_pickle('<YOUR DIRECTORY NAME>/myRawData_standard_one_hot_imputed_0.pkl').imputeData


Note the the ``imputeData`` member variable being used to access the ``Moon``'s cleaned data. See :ref:`moon` documentation for more info on member variables, etc.

Step 7: Get Params
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 

    We recommended using the the ``planet.writeParams_toYaml()`` method to write this dictionary to a ``.yaml`` file, which in turn can be passed to the ``Planet`` constructor as a `YAML_PATH` for quick pipelining. This will eliminate the need to explore, scale, encode, etc. your data again.

.. ipython:: python

    planet.getParams()

Once you have explored your data and determined how to best preprocess, this entire workflow can be streamlined using the following code:

.. code-block:: python

    from thema.multiverse import Planet

    yaml = "<PATH TO params.yaml>"

    planet = Planet(YAML_PATH=yaml)
    planet.fit()
