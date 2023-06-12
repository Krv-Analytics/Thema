# Makefile
include .env
PARAMS_FILE := $(strip $(params))
PARAMS_JSON := $(shell cat $(PARAMS_FILE))
RUN_NAME := $(shell echo '$(PARAMS_JSON)' | jq -r '.Run_Name')
COVERAGE_FILTER := $(shell echo '$(PARAMS_JSON)' | jq -r '.coverage_filter')

all: process-data projections models model-selection 
	@echo "Process complete"

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
	poetry run python src/processing/pulling/data_generator.py -v

process-data:
	cd scripts/bash && ./cleaner.sh

projections: 
	cd scripts/bash && ./projector.sh

summarize-projections:
	poetry run python src/summarizing/projection_summarizer.py

models: 
	cd scripts/bash && ./model_generator.sh 

model-histogram:
	cd scripts/python && poetry run python model_histogram.py

curvature-distances:
	cd scripts/python && poetry run python curvature_distance_generator.py

curvature-histogram: curvature-distances
	cd scripts/python && poetry run python curvature_histogram.py

stability-histogram:
	cd scripts/python && poetry run python stability_histogram.py

dendrogram:
	cd scripts/bash && ./dendrogram.sh

model-clustering: curvature-distances
	cd scripts/python && poetry run python clusterer.py

model-selection: model-clustering
	cd scripts/python && poetry run python selector.py



# Cleaning commands for data fields 

clean: clean-processed-data clean-projections clean-models clean-model-analysis clean-final-models
	rm -f -r data/$(RUN_NAME)/

clean-processed-data: 
	rm -f -r data/$(RUN_NAME)/clean/*

clean-projections:
	rm -f -r data/${RUN_NAME}/projections/*

clean-models:
	rm -f -r data/${RUN_NAME}/models/*

clean-model-analysis:
	rm -f -r data/${RUN_NAME}/model_analysis/

clean-final-models:
	rm -f -r  data/${RUN_NAME}/final_models/

clean-raw-data: 
	rm -f -r data/${RUN_NAME}/

clean-all :
	rm -f -r data/





