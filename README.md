# coal_mapper
An implementation of the Kepler Mapper to analyze various energy datasets



## Installation

To clone the coal_mapper repo, run in a bash shell (eg. terminal)

```
$ git clone git@github.com:sgathrid/coal_mapper.git
```

It is recommended to use the [`poetry`](https://python-poetry.org) package
manager. With `poetry` installed, setting up the repository works like
this:

```
$ poetry install
```

Since `poetry` creates its own virtual environment, it is easiest to
interact with scripts by calling `poetry shell`.


## MongoDB API Connection String Request and Configuration

** coming soon will be HTML request form for read-only MongoDB access **

In coal_mapper home directory, run the following commands:

```
$ cp .env.SAMPLE .env
$ vim .env
```
This creates a .env file and opens that file in a vim editor. You will then copy your API access token into the `mongo_client` field.


## Retrieving Data from Mongo DB 

Once the access token has been set, src/data_processing/data.py may be used to interface with MongoDB to retrieve a maintained dataset. To retrieve the default data set, run:

```
$ poetry run python src/data_processing/data.py
``` 
This will create and populate the **data** directory with the default data *coal_mapper_one_hot_scaled_TSNE.pkl*  
Run the above code with the **--help** flag for specifics of usage.   

TODO:  Should put in a link to a doc.md that specifies the possible datasets ie options for the -d flag 

## Computing Curvature 

Now that a dataset is stored locally in the data directory, we may use *compute_curvature.py* to generate, analyze, and store kepler mapper objects based on our desired parameter inputs. As above, appending the **--help** flag to the following command will provide specifics for usage, but for default values, run 

```
$ poetry run python src/compute_curvature.py -v 
``` 

This generates a mapper object from default parameters and computes the Ollivier-Ricci curvature values for the corresponding graph, populating an instance of TopologyMapper defined in [`utils.py`](https://github.com/sgathrid/coal_mapper/blob/main/src/utils.py)). The TopologyMapper object is then stored in the outputs/curvature/ directory. Note that, in this case, a TopologyMapper object corresponds to a single file. However, when specifying the **--min_intersection** parameter, the number of TopologyMapper objects in a given file is determined by the length of the min intersection list. 

Since there is considerable cause to generate many TopologyMapper objects, one may run (from the src/ directory)

```
$ ./scripts/curvature.sh 
``` 
to generate a distribution of TopologyMapper objects, storing the corresponding files in outputs/curvature/.   

## Comparing TopologyMapper Objects 

TODO: Update with usage of 
