# THEMA 🔮

_By Krv Analytics._

---

Welcome to **Thema**, our custom _Topological Hyperparameter Evaluation and Mapping Algorithm_!🌟

---

Thema, inspired by the German word "Thema" meaning "subject" or "topic," is your go-to tool for uncovering the most intriguing and significant aspects hidden within your data. By leveraging advanced techniques to understand the distribution of representations that emerge from various preprocessing and hyperparameter choices, Thema brings a new level of insight to your unsupervised tasks. 🧠🔍

Imagine navigating a landscape of endless possibilities, where each preprocessing step and parameter tweak can lead to a new perspective on your data. Thema acts as your guide through this complex terrain, helping you identify the most salient patterns and features and advising you on the most trustworthy representations. It's like having a data scientist with a knack for finding the most interesting and reliable stories your data has to tell. 🗺️✨

Dive into the world of Thema and transform the way you explore and interpret your data. With Thema, the subject of your analysis is always the star of the show! 🌠🚀

---

## Installation

To install the Thema software package, you can use pip, the Python package installer. Follow the steps below to install Thema:

1. Open a terminal.

2. Run the following command:

```bash
pip install thema
```

This command will download and install the latest version of Thema from the Python Package Index (PyPI).

Once the installation is complete, you can verify that Thema is installed correctly by running:

```bash
pip show thema
```

This will display information about the installed package, including its version and location. Now you're ready to start using Thema in your projects!

You can also clone our repository by running:

```bash
git clone https://github.com/Krv-Analytics/Thema.git
```

This will give you access to our [invoke](https://www.pyinvoke.org/) scripts. Once you have `invoke` installed (`pip install invoke`) you can create a conda environment with:

```bash
cd Thema
invoke condaenv
```

or if you prefer [poetry](https://python-poetry.org/):

```
poetry install
```

The details of these environments are in `environment.yml` and `pyproject.toml` respectively.

---

## Usage

Welcome to the **Thema** usage tutorial! This guide will walk you through the process of using Thema to analyze your data, generate embeddings, and visualize the results. Follow the steps below to get started. See `params.yaml.sample` as a template for defining your own representation grid search. Once you've filled this out, follwow the steps below!

### Step 1: Encode, Clean, and Impute Raw Data

First, you'll need to encode, clean, and impute your raw data using the `Planet` class. Make sure you have your parameters defined in a YAML file.

```python
from thema.multiverse import Planet

yaml = "path/to/params.yaml"

# Encode, Clean and Impute Raw Data
planet = Planet(YAML_PATH=yaml)
planet.fit()
```

### Step 2: Generate Low Dimensional Embeddings

Next, use the `Oort` class to generate low-dimensional embeddings from your processed data.

```python
from thema.multiverse import Oort

# Generate Low Dimensional Embeddings
oort = Oort(YAML_PATH=yaml)
oort.fit()
```

### Step 3: Generate Multiscale Graph Clustering Models

Now, create multiscale graph clustering models using the `Galaxy` class.

```python
from thema.multiverse import Galaxy

# Generate Multiscale Graph Clustering Models
galaxy = Galaxy(YAML_PATH=yaml)
galaxy.fit()
```

### Step 4: Cluster Representations and Select Representatives

After generating the clustering models, cluster the representations and select representatives.

```python
# Cluster Representations and Select Representatives
model_representatives = galaxy.collapse()
```

### Step 5: Visualize the Results

Finally, visualize the results using the `Telescope` class. Choose a sample from the model representatives to create a graph.

```python
from thema.probe import Telescope

# Visualize Mode
sample = model_representatives[1]['star']
T = Telescope(star_file=sample)
T.makeGraph()
```

With these steps, you have successfully processed your data, generated embeddings, created clustering models, and visualized the results using Thema. Enjoy exploring the insights and patterns uncovered in your data!
