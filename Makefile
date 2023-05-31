# Makefile

all: fetch-raw-data fetch-processed-data 

install: check-poetry 
	@echo " Generating and populating .env file..."
	python scripts/setup.py
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

fetch: fetch-raw-data fetch-processed-data

fetch-raw-data:
	python src/processing/pulling/data_generator.py -v

fetch-processed-data:
	python src/processing/cleaning/cleaner.py -v

projections: 
	cd scripts && ./projection_grid_search.sh

##########################################################################################################################
# Relative paths to data files from the scripts directory 
# May consider setting these to be absolute paths 
path_to_raw = "../data/raw/esg_raw2017.pkl"
path_to_clean = "../data/clean/clean_data_standard_scaled_integer-encoding_filtered.pkl"
path_to_umap = "../data/projections/UMAP/*"
##########################################################################################################################

models: 
	cd scripts && ./model_grid_search.sh ${path_to_raw} ${path_to_clean} ${path_to_umap}

clean-raw-data: 
	rm -f data/raw/* 

clean-processed-data: 
	rm -f data/clean/*

clean-projections:
	rm -f -r data/projections/*

clean-models:
	rm -f -r data/models/*

clean-model-analysis:
	rm -f -r data/model_analysis/*

clean: clean-raw-data clean-processed-data clean-projections 
	rm -f -r data/


