# Makefile
include .env
PARAMS_JSON := $(strip $(params))



install: check-poetry 
	@echo " Generating and populating .env file..."
	python scripts/python/setup.py
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
	cd scripts/bash && ./cleaner.sh

projections: 
	cd scripts/bash && ./projector.sh

models: 
	cd scripts/bash && ./model_generator.sh 

histogram:
	cd scripts/bash && ./histogram.sh
dendrogram:
	cd scripts/bash && ./dendrogram.sh
model-selection:
	cd scripts/bash && ./model_selector.sh





# Cleaning commands for data fields 

clean-raw-data: 
	rm -f data/raw/* 

clean-processed-data: 
	rm -f data/clean/*

clean-projections:
	rm -f -r data/projections/*

clean-models:
	rm -f -r data/models/*

clean-model-analysis:
	rm -f -r data/model_analysis/

clean: clean-processed-data clean-projections 
	rm -f -r data/

clean_full : clean-raw-data clean-processed-data clean-projections 
	rm -f -r data/



