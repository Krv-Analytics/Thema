# Makefile
include .env
PARAMS_FILE := $(strip $(params))
PARAMS_JSON := $(shell cat $(PARAMS_FILE))
RUN_NAME := $(shell echo '$(PARAMS_JSON)' | jq -r '.Run_Name')
COVERAGE_FILTER := $(shell echo '$(PARAMS_JSON)' | jq -r '.coverage_filter')


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

model-histogram:
	cd scripts/python && python model_histogram.py

curvature-histogram:
	cd scripts/python && python curvature_histogram.py
dendrogram:
	cd scripts/bash && ./dendrogram.sh
model-selection:
	cd scripts/bash && ./model_selector.sh



# Cleaning commands for data fields 

clean-processed-data: 
	rm -f data/$(RUN_NAME)/clean/*

clean-projections:
	rm -f -r data/${RUN_NAME}/projections/*

clean-models:
	rm -f -r data/${RUN_NAME}/models/*

clean-model-analysis:
	rm -f -r data/${RUN_NAME}/model_analysis/

clean-final-models:
	rm -f -r  data/${RUN_NAME}/final_models/

clean: clean-processed-data clean-projections 
	rm -f -r data/${RUN_NAME}/

clean-raw-data: 
	rm -f data/raw/* 

clean-full : clean-raw-data clean-processed-data clean-projections 
	rm -f -r data/





