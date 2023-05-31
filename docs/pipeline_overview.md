# Pipeline Overview & File Descriptions
Here we will give a brief overview of our provided files in the `src` directory, and how to execute our pipeline using `make` commands.


## Processing
This subdirectory handles all data processing. Our project supports interfacing directly with MongoDB to fetch raw data. However, we expect this pipeline to be useful to other folks with entirely different data configurations, cleaning methods, and preprocessing techniques. In the end, our clustering method only requires `pickle` files that store `raw`, `clean`, and `projected` versions of your data; as is common practice, we expect cleaned data to be scaled with no missing values, and have categorical variables encoded. We hope that our workflow for pulling data and cleaning is easily customizable and with minimal effort can support the practices of a many domains. The final stage of our preprocessing is dimensionality reduction, or `projecting`. Our pipeline uses `UMAP`, but again we hope that our architechture makes it easy to incorporate other methods. However, hopefully our functionality for __evaluating__ a hyperparameter grid search for `UMAP` projections will convince you to adpot this method.



### 1: Pulling
We provide a `Mongo` class that handles the interface with MongoDB. If you are have your data stored on MongoDB and have configured your `.env` file (see `README`), you can use to execute `data_generator.py`, generating a local copy of our dataset:

<!-- We should add ability to read env variables for Mongo colletion/database from make  -->
```
make fetch-raw-data
``` 

This creates a local `data` directory and adds a pickle file to `data/raw` by pulling from the specified collection on MongoDB.
If you are not interfacing with Mongo, run:

```
make data-dir
``` 
to generate the local data directory, and deposit a pickled version of your raw data (csv format) to `data/raw`. 

Going forward, to use our `make` commands out of the box, name your file `raw_values.pkl`. 
Otherwise, we invite you to edit the `Makefile` to match your own naming conventions.

### 2: Cleaning
Now that you have a local copy of your `raw` data, you can run the `cleaner.py` driver via:
```
make fetch-processed-data
``` 
to clean your data. This adds a `clean` pickle file to `data/clean`. The functionality for cleaning is contained in `cleaner_helper.py`. Our cleaning consists of filtering columns, dropping `NaNs`, scaling, and encoding categorical variables. Our methods for this study are defaulted in `cleaner.py`, but these can be specified as needed using command line arguments and we hope it is easy to expand functionality if need be by adapting the helper functions.





### 3: Projecting
To produce a single UMAP projection, of your `clean` data run `projector.py`:

```
make projection n=10 d=0
``` 

Notice, the driver is parameterized by 2 inputs `n_neighbors` or and `min_dist` (n,d respectively). We point you to the UMAP paper and documentation for a full description, but the basic idea is that these parameters change your view of "locality" when conducting a manifold based projection. Though UMAP advertises itself as a stucture preserving dimensionality reduction algorithm, the structure that you preserve can change quite significantly based on these two input parameters. Thus, we *reccomend* that you run a grid search over these input parameters, and explore the structure of your data set at various resolutions. These will allow you to generate a rich class of models, and we provide functionality for grouping these models into equivalency classes to make down stream analysis manageable. To run a UMAP grid search, we run `projection_grid_search.sh` in the `scripts` directory with:

```
make projections
```
This executes a script that iteratively calls `projector.py` over a parameter grid to populate `data/projections/UMAP` with pickle files. Once again, our grid is defaulted in the script, but feel free to edit this as you like.
 

## Modeling
Here are the most relevant files for our modeling pipeline:
```
├── jmapper.py
├── model.py
├── model_generator.py
├── model_helper.py
├── model_selector.py
├── model_selector_helper.py
```
We provide two classes `JMapper` and `Model`. `JMapper` adds some new functionality to [scikit-tda's](https://kepler-mapper.scikit-tda.org/en/latest/) `KepplerMapper` . Our `Model` uses the graph structure of the Mapper output as a clustering interpretation: namely each connected component we consider to be a cluster and provide functionality for analyzing a density based description of each cluster. 

### 4. Model Generation
You can generate a specific model by running `model_generator.py`:
```
python ../src/modeling/model_generator.py --raw path/to/raw --clean path/to/clean --projection path/to/projection
make single-model raw= clean= projection= 
```

As with the UMAP projections, we have discovered huge amounts of variability in the models that the Mapper Algorithm can produce. This is expected from the nature of the Mapper's hyperparameters. In particular, `n_cubes` and `perc_overlap` which define a resolution at which to pull out sturcture of your data. However, thanks to recent advances in graph learning, there now exist well principled metrics to compare graphs based on [discrete curvature](https://arxiv.org/abs/2301.12906). Thus, once again, we encourage that you run a grid search over the hyperparameters needed to generate a `Model`. To configure your grid search, edit `model_grid_search.sh` (also in `scripts`), and execute with:

```
make models
``` 

Upon output, the models are grouped into subdirectories based on their number of connected components. We have named these `target_groups` as each connected component should have similarities that can be whittled into a target strategy.  


## Tuning
This step of the pipeline allows you to "tune" a large set of potential models, as generated by `model_grid_search.sh`. In particular, we want to understand equivalency classes of models based on the structure of their graph representation. We also provide functionality for filtering out models that don't cover a sufficient percentage of the dataset; for our project, we want wholistic clusterings of coal plants which inlcude, for example, at least 90% of plants in our final represnetation.


### 5. Visualize Model Distribution
Since we are binnning (and interpreting) models based on their number of connected components, we find it very informative to visualize the results of our grid search by counting the number of models per policy group. To produce this plot, run the following command:

```
python src/modeling/model_selector.py -H --coverage_filter 0.9
``` 

where again, `coverage_filter` can take any value of interest. This bar plot gives insight to the structure of our dataset at various scales, and can point to the most popular policy group resolution. We encourage that you use this information to tweak the model grid search to your liking, and select a frequent `num_policy_groups` that suits your goals for downstream analysis. 

### 6. Generate Metrics
At this step we assume that we have selected a set of models, based on a desirable choice of `num_policy_groups` (e.g. 10) and `coverage_filter` (e.g. 0.9). We can now execute the driver file to generate pairwise distances between each of the models by running:

```
python src/tuning/metrics/metric_generator.py --num_policy_groups 10 --coverage_filter 0.9 -v
``` 

This saves a distance matrix to pickle in the `data` directory. 

### 7. Graph Clustering
Assuming we have pairwise distances between the models' graphs, we can now visualize a dendrogram by fitting an [Agglomerative Clustering Model](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html). This visualization will elucidate the structural equivalencey classes of the graphs. To generate this visualization, run:

```
python src/tuning/graph_clustering/model_clusterer.py --num_policy_groups 10 --distance_threshold 0.5  -p 10 
``` 
to visualize a dendrogram with 10 levels and labels determined by a distance threshold of 0.5. We encourage you to tweak `distance_threshold` until you are satisfied with the courseness of the grouping, which can be checked visually by each group in the dendrogram being colored as you wish. Once an ideal threshold `D` has been found, run:

```
python src/tuning/graph_clustering/model_clusterer.py --num_policy_groups 10 --distance_threshold D -s -v
``` 

to save the information on graph equivalency classes to pickle. 



## 8. Model Selection 
For the time being, this is the final step of our pipeline! Here we provide functionality for choosing a representative model from an equivalency class. So if your dendrogram in Step 7 showed 3 model groups (i.e. 3 different colors) for 10 policy groups, then we return you 3 models (the idea being that each of these is structurally distict although they report the same number of policy groups). Namely, we select the model with the optimal coverage from each group. To execute the final model selection, run:


```
python src/modeling/model_selector.py --num_policy_groups 10 -v
``` 

this populates a pickle file with your equivalence class representatives! 


## 9. Analysis
Now you can read in these models, and analyze the clustering. For example, in a notebook you can read in each of the structurally distinct models using the following code:
```
selection = "root/data/model_analysis/models/11_policy_groups/equivalence_class_candidates_landscape_*.pkl"
with open(selection,"rb") as s:
    models = pickle.load(s)
reference = {}
for i,model in enumerate(models):
    reference[f"model_{i}"] = Model(model)
```

To understand what attributes connect your each policy group, try running this next:

```
example_model = referece["model_0"]
example_model.cluster_descriptions
```

You can also visualize properties of your models using built in methods. For example, try running:

```
example_model.visualize_model()
```
```
example_model.visualize_projection()
```
Have fun analyzing!


# Visualizing
**work in progress**
This subdirectory contains various visualization functions to help inform stages of the pipeline. Most of the functionality for visualizing properties of models are methods of the `Model` class. However, there are other interesting summary visualizations that we hope to support in this section that will populate the corresponding visualizations subdirectory in `data`.



## Scripts 

```
├── mapper_grid_search.sh
└── projection_grid_search.sh
```
With the current implementation, these bash shells *must be run from the scripts* directory.

The parameters over which to conduct the grid search can be set by editing the files. 
* projection_grid_search takes no arguments
* mapper_grid_search.sh takes 3 arguments:
  * path to raw data, normally something like ../data/raw/coal_plant_raw_data.pkl
  * path to clean data, normally something like ../data/clean/clean_data_standard_scaled_integer-encdoding_filtered.pkl
  * path to projections, normally ../data/projections/UMAP/* 
