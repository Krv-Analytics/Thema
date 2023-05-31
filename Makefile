# Makefile
include .env
PARAMS_JSON := $(strip $(params))

run: process-data 

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

process-data:
	cd scripts && ./cleaning_script.sh

projections: 
	cd scripts && ./projection_grid_search.sh

models: 
	cd scripts && ./model_grid_search.sh 


# Cleaning commands for data fields 

clean-raw-data: 
	rm -f data/raw/* 

clean-process-data: 
	rm -f data/clean/*

clean-projections:
	rm -f -r data/projections/*

clean-models:
	rm -f -r data/models/*

clean: clean-processed-data clean-projections 
	rm -f -r data/

clean_full : clean-raw-data clean-processed-data clean-projections 
	rm -f -r data/



