# Makefile

all: fetch-raw-data fetch-processed-data 

install: check-poetry 
	@echo " Generating and populating .env file..."
	poetry run python scripts/setup.py
	@echo "Installing necessary dependencies..." 
	poetry run pip install python-dotenv pandas pymongo scikit-learn umap-learn hdbscan kmapper networkx matplotlib seaborn giotto-tda


uninstall: clean
	@echo "Removing Dependencies from Poetry environment..."
	poetry run pip uninstall python-dotenv pandas pymongo scikit-learn umap-learn hdbscan kmapper networkx matplotlib seaborn giotto-tda
	@echo "Removing Poetry"
	sudo pip uninstall poetry 
	rm -f .env 

check-poetry:
	@which poetry || (echo "Poetry is not installed. Installing..."; sudo pip install poetry; poetry install;)


data-dir:
	@mkdir data && cd data && mkdir raw && mkdir clean

fetch: fetch-raw-data fetch-processed-data

fetch-raw-data:
	poetry run python src/processing/pulling/data_generator.py -v

fetch-processed-data:
	poetry run python src/processing/cleaning/cleaner.py -v

#TODO: want to read in -n (n_neighbors) and -d (min_dist)
single-projection:
	poetry run python src/processing/projecting/projector.py 

#TODO: want to read in -n (n_cubes) and -p (perc_overlap) -m (min_intersection)
single-model:


projections: 
	cd scripts && ./projection_grid_search.sh

##########################################################################################################################
# Relative paths to data files from the scripts directory 
# May consider setting these to be absolute paths 
path_to_raw = "../data/raw/raw_data.pkl"
path_to_clean = "../data/clean/clean_data_standard_scaled_integer-encoding_filtered.pkl"
path_to_umap = "../data/projections/UMAP/*"
##########################################################################################################################

models: 
	cd scripts && ./model_grid_search.sh ${path_to_raw} ${path_to_clean} ${path_to_umap}

# Coverage Filter Float read in 
model-histogram: 
	poetry run python src/modeling/model_selector.py -H --coverage_filter ${coverage_filter}

#TODO: Read in cmd line args
model-dendrogram:
	poetry run python src/tuning/graph_clustering/model_clusterer.py --num_policy_groups 10 --distance_threshold 0.5  -p 10
model-equivalency-classes:
	poetry run python src/tuning/graph_clustering/model_clusterer.py --num_policy_groups 10 --distance_threshold D -s -v
model-selection:
	python src/modeling/model_selector.py --num_policy_groups 10 -v


##################################################
clean-raw-data: 
	rm -f data/raw/* 

clean-processed-data: 
	rm -f data/clean/*

clean-projections:
	rm -f -r data/projections/*

clean-models:
	rm -f -r data/models/*

clean: clean-raw-data clean-processed-data clean-projections 
	rm -f -r data/






