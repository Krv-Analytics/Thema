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
Then run the following script to generate pickled data files:

```
$ cd data_processing/ 
$ python data.py --type="pkl"
```

