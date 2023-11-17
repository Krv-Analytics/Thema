# krv_mapper

## Description 
A graph generation pipeline that combines `UMAP`, the `Mapper` algorithm, and `Ollivier Ricci Curvature` to model and analyze sparse, high dimensional data sets.


## Installation

To clone `krv_mapper`, run in your favorite bash shell (eg. terminal)

```
git clone git@github.com:jeremy-wayland/krv_mapper.git
```

## Make
To install, run:
```
cd krv_mapper && make install 
```
### Poetry
We use the [`poetry`](https://python-poetry.org) package manager to handle our dependencies. 
The Makefile will install poetry, and then load all dependencies needed for this project.
You can access this virtual environment (before running any of the provided files) by running 
```
poetry shell 
```  
### Environment Configuration
The Makefile will also write a `.env` file for you, configuring paths and setting environment variables.
If you are using a dataset stored on MongoDB, then please configure the following variables: 
`mongo_client`,`mongo_database`,`mongo_collection`. Then configure `params.yaml` according to the sample provided
to set the parameters required to run our pipeline.


## Work Flow 

Once you have have configured your local environment, you're all set to start using our functionality! As an overview, we provide functionality for loading data, cleaning data, projecting data, modeling data, and comparing models. For each stage of our pipeline, we provide a python driver file to perform the desired operation and save the output in an corresponding `data` directory. See `docs` for a pipeline overview and a description of our outputs.

## Possible Workflow using Makefile:
`make process-data`
`make projections`
`make models`
`make histogram`
`make dendrogram`
`make model-selection`


### File Structure
```
├── modeling
├── processing
│   ├── cleaning
│   ├── projecting
│   └── pulling
└── tuning
│    ├── graph_clustering
│    └── metrics
└── visualizing
```



