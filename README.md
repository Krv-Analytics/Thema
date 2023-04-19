# coal_mapper

## Description 
An topolgical clustering pipeline that combines `UMAP` and the `Mapper` algorithm to analyze various energy datasets. 



## Installation

To clone the coal_mapper repo, run in your favorite bash shell (eg. terminal)

```
$ git clone git@github.com:sgathrid/coal_mapper.git
```

### Poetry and Dependencies

It is recommended to use the [`poetry`](https://python-poetry.org) package
manager. Once you have installed `poetry`, you will need to initialize `poetry` by running:

```
$ poetry install
```

We use `poetry` to manage our dependencies. This makes sure you are up to date with the necessary packages without having to update your root dependencies.  `poetry` is able to do this by creating its own virtual environment where it has access to packages specific to this project. You can access this virtual environment (before running any of the provided files) by running 
```
$ poetry shell 
```  
You may still need to install packages into this environment, which can be done within a poetry shell by running 
```
pip install package_name
```
### Local Environment Configuration

#### Creating a .env file

** Coming soon is a setup script that automates the .env configuration process **

In coal_mapper home directory, run the following command to create a .env file:

```
$ cp .env.SAMPLE .env
```

 Using your favorite text editor, you should now populate the fields  `root`, `src`, and (optional) `mongo_client`. You will need to set

```
root = path to this repository 
src = root + "src/"
```

Should you wish to link this repository to a Mongo database, you will need to copy an API access token into the `mongo_client` field. To generate models with our provided data, you will need to request access first. You can do so by emailing __coalmapper.TDA@gmail.com__. 

** coming soon will be HTML request form for read-only MongoDB access **

Note: If you do not configure `mongo_client` in your .env, you will not be able to pull data using the data_generator.py file. 

####  Configuring DVC (optional) 

** Soon to only exist on a development fork of this repository** 

We recommend that you skip the configuration of dvc unless you plan on contributing to this repository. We have set up a Google Drive dvc_store in order to store models and locally generated data for ease of sharing (helpful when wanting to store models requiring signicant computation). Should you wish to obtain access to our dvc_store, you will need to request a google drive access string from __coalmapper.TDA@gmail.com__. You can then create a config file in the .dvc directory by running 
```
cp .dvc/config.SAMPLE .dvc/config
```
and populate the `gdrive_client_id` and `gdrive_client_secret` fields in .dvc/config with your provided google drive access credentials. 

## Work Flow 

Once you have have configured your local environment, you're all set to start using our functionality! As an overview, we provide functionality for loading data, cleaning data, projecting data, modeling data, and comparing models. For each stage of our pipeline, we provide a python driver file to perform the desired operation and save the output in an corresponding `data` directory. 

## File Descriptions 
Here we will give a brief overview of our provided files in the `src` directory. 

### Processing
This subdirectory handles all data processing. Our project interfaces directly with MongoDB, as we currently have our custom `CoalMapper` database stored here. However, we expect this pipeline to be useful to other folks with entirely different data configurations, cleaning methods, and preprocessing techniques. In the end, our clustering technique only requires `pickle` files that store `raw`, `clean`, and `projected` versions of your data; as is common practice, we expect cleaned data to be scaled with no missing values, and have categorical variables encoded. We hope that our workflow for pulling data and cleaning is easily customizable and with minimal effort can support the practices of a many domains. The final stage of our preprocessing is dimensionality reduction, or `projecting`. Our pipeline uses `UMAP`, but again we hope that our architechture makes it easy to incorporate other methods. However, hopefully our functionality for __evaluating__ a hyperparameter grid search for `UMAP` projections will convince you to adpot this method.


#### 1: Pulling 
We provide a `Mongo` class that handles the interface with MongoDB. If you have configured your `.env` file as specified above, you can execute the driver file to generate a local copy of our dataset:

```
$ python src/processing/pulling/data_generator.py -v
```  

This creates a local `data` directory and adds a pickle file to `data/raw` by pulling from the `CoalMapper` database.  

#### 2: Cleaning
Now that you have a local copy of your `raw` data, you can run:
```
$ python src/processing/cleaning/cleaner.py -v
``` 

to clean your data, adding a `clean` pickle file to `data/clean`. The functionality for cleaning is contained in `cleaner_helper.py`. Our cleaning consists of filtering columns, dropping `NaNs`, scaling, and encoding categorical variables. Our methods for this study are defualted in `cleaner.py`, but these can be specified as needed using command line arguments and we hope it is easy to expand functionality if need be by adapting the helper functions.


#### 3: Projecting
To produce a single projection, of your `clean` data run:

```
$ python src/processing/projecting/projector.py -n 10 -d 0
``` 

Notice, the projection is parameterized by 2 inputs which represent `n_neighbors` and `min_dist` (n,d). We point you to the UMAP paper and documentation for a full description, but the basic idea is that these parameters change your view of "locality" when conducting a manifold based projection. Though UMAP advertises itself as a (topological and geometric) stucture preserving dimensionality reduction algorithm, the structure that you preserve can change quite signigicantly based on these two input parameters. Thus, we *reccomend* that you run a grid search over these input parameters, and explore the structure of your data set at various resolutions. These will allow you to generate a rich class of models, and we provide functionality for grouping these models into equivalency classes to make down stream analysis manageable. To run UMAP grid search, `cd` into the `scripts/` directory and run the following command:

```
$ ./projection_grid_search.sh
``` 

to script iteratively calls the `projection.py` over a parameter grid to populate `data/projections/UMAP/` with pickle files. Once again, our grid is defaulted in the script, but feel free to edit this as you like in the bash file.
 

### Tuning


#### Metrics
#### Par


### Visualizing



## Scripts 

## Ouputs


