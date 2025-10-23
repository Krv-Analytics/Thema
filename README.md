# THEMA 🔮

<p align="center">
  <a href="https://github.com/Krv-Analytics/Thema/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Krv-Analytics/Thema?style=flat-square">
  </a>
  <a href="https://github.com/Krv-Analytics/Thema/blob/main/LICENSE.md">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey?style=flat-square">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python">
  </a>
  <a href="https://krv-analytics.github.io/Thema/">
    <img src="https://img.shields.io/badge/docs-available-green?style=flat-square&logo=gitbook">
  </a>
  <br>
  <a href="https://krv.ai">
    <img src="https://img.shields.io/badge/web-krv.ai-black?style=flat-square&logo=vercel">
  </a>
  <a href="https://www.linkedin.com/company/krv-analytics">
    <img src="https://img.shields.io/badge/LinkedIn-Krv%20Analytics-blue?style=flat-square&logo=linkedin">
  </a>
  <a href="mailto:team@krv.ai">
    <img src="https://img.shields.io/badge/Email-team@krv.ai-fe2b27?style=flat-square&logo=gmail">
  </a>
</p>

_By Krv Labs._

---

Welcome to **Thema**, our **Topological Hyperparameter Evaluation and Mapping Algorithm**! 🌟

---

Thema systematically explores hyperparameter spaces for unsupervised learning through topological data analysis. Instead of manually tuning preprocessing and embedding parameters, Thema generates candidate models systematically and uses curvature-based graph distances to identify diverse, high-quality representatives.

By leveraging advanced techniques to understand the distribution of representations that emerge from various preprocessing and hyperparameter choices, Thema brings a new level of insight to your unsupervised tasks. Navigate the complex terrain of hyperparameter optimization with confidence, identifying the most salient patterns and features in your data. 🧠🔍

## Architecture

Thema operates through three distinct modules:

### 🌍 **Multiverse** - Core Data Processing Pipeline

The foundational system that transforms raw data into topological representations:

- **Planet** (Preprocessing): Generates multiple clean data versions with different imputation, scaling, and encoding strategies
- **Oort** (Embeddings): Creates low-dimensional projections across parameter grids (t-SNE, PCA)
- **Galaxy** (Graph Construction): Builds Mapper graphs, computes topological distances, and selects representatives

### 🚀 **Expansion** - Advanced Analytics Extensions

Specialized tools for extended analysis capabilities:

- **Realtor**: Real estate and geographic data analysis tools
- **Utils**: Utility functions for specialized data processing workflows

---

## Installation

Install Thema using pip:

```bash
pip install thema
```

Verify the installation:

```bash
pip show thema
```

---

## Quick Start

Get started with Thema in just a few lines of code! See `params.yaml.sample` as a template for defining your own representation grid search.

```python
import thema
from thema import Thema

# Enable logging to see progress
thema.enable_logging()

# Initialize Thema with your configuration
my_thema = Thema(YAML_PATH='path/to/custom.yaml')

# Run the complete pipeline
my_thema.genesis()

# Access the selected representative model files
print(my_thema.selected_model_files)
```

That's it! Thema will systematically process your data through preprocessing, embedding, and graph construction stages, automatically selecting the most representative models.

---

## Pipeline Components

### Step 1: Preprocessing with Planet 🌍

Clean, encode, and impute your raw data with multiple strategies:

```python
from thema.multiverse import Planet

# Initialize Planet with your configuration
planet = Planet(YAML_PATH='path/to/params.yaml')

# Generate multiple cleaned datasets
planet.fit()
```

**Planet** creates various versions of your cleaned data with different:

- Scaling methods (`standard`, `minmax`, `robust`)
- Encoding strategies (`one_hot`, `label`, `ordinal`)
- Imputation methods (`mean`, `median`, `mode`, `sampleNormal`)
- Random seeds for reproducible sampling

### Step 2: Embedding with Oort ☄️

Generate low-dimensional projections from your cleaned data:

```python
from thema.multiverse import Oort

# Create embeddings across parameter grids
oort = Oort(YAML_PATH='path/to/params.yaml')
oort.fit()
```

**Oort** produces embeddings using:

- **t-SNE**: With various perplexity values and dimensions
- **PCA**: With different dimensionality settings
- Multiple random seeds for robustness

### Step 3: Graph Construction with Galaxy 🌌

Build Mapper graphs and select representatives:

```python
from thema.multiverse import Galaxy

# Generate graph models across hyperparameter space
galaxy = Galaxy(YAML_PATH='path/to/params.yaml')
galaxy.fit()

# Cluster and select representative models
representatives = galaxy.collapse()
```

**Galaxy** creates and analyzes:

- **Mapper graphs**: Using various cover resolutions and overlap parameters
- **Topological distances**: Computing curvature-based similarity metrics
- **Representative selection**: Choosing diverse, high-quality models using clustering

### Coordinate Space Generation

Generate a 2D embedding space of your models for analysis:

```python
# Get 2D coordinates of all models in the galaxy
coordinates = galaxy.get_galaxy_coordinates()

# Access the selected representatives
for cluster_id, info in galaxy.selection.items():
    print(f"Cluster {cluster_id}: {info['star']} ({info['cluster_size']} models)")
```

---

## Key Features

✨ **Systematic Exploration**: Automatically explores preprocessing and embedding parameter combinations

🎯 **Representative Selection**: Uses topological distance metrics to identify diverse, high-quality models

📊 **Robust Analysis**: Generates multiple models per configuration for statistical reliability

🔧 **Flexible Configuration**: YAML-based configuration for easy parameter management

🚀 **Parallel Processing**: Efficient multiprocessing for large parameter grids

📈 **Topological Insights**: Leverage graph topology and curvature for model comparison

---

## Output Structure

Thema organizes outputs hierarchically:

```
{outDir}/{runName}/
├── clean/                  # Preprocessed datasets (Moon files)
│   ├── moon_42_0.pkl
│   ├── moon_42_1.pkl
│   └── ...
├── projections/           # Low-dimensional embeddings (Comet files)
│   ├── tsne_perplexity30_dims2_seed42_moon_42_0.pkl
│   ├── pca_dims2_seed42_moon_42_0.pkl
│   └── ...
└── models/               # Mapper graphs (Star files)
    ├── star_tsne_perplexity30_nCubes10_overlap0.6.pkl
    ├── star_pca_dims2_nCubes10_overlap0.6.pkl
    └── ...
```

---

## When to Use Thema

**✅ Good Use Cases:**

- Exploring preprocessing choices for unsupervised learning
- Comparing embedding methods systematically
- Finding robust data representations across hyperparameter grids
- Identifying diverse graph topologies in your data
- Validating clustering stability across multiple configurations

**❌ Not Ideal For:**

- Supervised learning (Thema focuses on unsupervised tasks)
- Single fixed preprocessing pipeline
- Real-time inference (Thema generates models offline)

---

## Documentation

For comprehensive guides and tutorials, visit our [documentation](https://krv-analytics.github.io/Thema/).

**Quick Links:**

- [Installation Guide](https://krv-analytics.github.io/Thema/userGuides/installation.html)
- [Complete Tutorial](https://krv-analytics.github.io/Thema/userGuides/beginner.html)
- [Programmatic API](https://krv-analytics.github.io/Thema/userGuides/programmatic.html)
- [Parameter Reference](https://krv-analytics.github.io/Thema/configuration.html)
- [Best Practices](https://krv-analytics.github.io/Thema/userGuides/best_practices.html)

---

**Transform the way you explore and interpret your data with Thema - where the topology of your analysis reveals the hidden stories in your data!** 🌠✨
