# Makefile
include .env
PARAMS_FILE := $(strip $(params))
PARAMS_JSON := $(shell cat $(PARAMS_FILE))
RUN_NAME := $(shell echo '$(PARAMS_JSON)' | jq -r '.Run_Name')
COVERAGE_FILTER := $(shell echo '$(PARAMS_JSON)' | jq -r '.coverage_filter')

all: init process-data projections jmaps jmap-selection 
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

process-data: init
	cd scripts/bash && ./cleaner.sh

projections: init
	cd scripts/bash && ./projector.sh

summarize-projections: init
	poetry run python src/modeling/synopsis/projection_summarizer.py

jmaps: init
	cd scripts/bash && ./jmap_generator.sh 

jmap-histogram: init
	cd scripts/python && poetry run python jmap_histogram.py

curvature-distances:init
	cd scripts/python && poetry run python curvature_distance_generator.py

curvature-histogram: init curvature-distances
	cd scripts/python && poetry run python curvature_histogram.py

stability-histogram:init
	cd scripts/python && poetry run python stability_histogram.py

dendrogram: init
	cd scripts/bash && ./dendrogram.sh

jmap-clustering: init curvature-distances
	cd scripts/python && poetry run python clusterer.py

jmap-selection: init jmap-clustering
	cd scripts/python && poetry run python selector.py


init: 
	cd scripts/bash && ./startup.sh

save: 
	cd scripts/bash && ./dvc_save.sh 

load: 
	cd scripts/bash && ./dvc_load.sh


# Cleaning commands for data fields 

clean: clean-processed-data clean-projections clean-jmaps clean-jmap-analysis clean-final-jmaps
	rm -f -r data/$(RUN_NAME)/

clean-processed-data: 
	rm -f -r data/$(RUN_NAME)/clean/*

clean-projections:
	rm -f -r data/${RUN_NAME}/projections/*

clean-jmaps:
	rm -f -r data/${RUN_NAME}/jmaps/*

clean-jmap-analysis:
	rm -f -r data/${RUN_NAME}/jmap_analysis/

clean-final-jmaps:
	rm -f -r  data/${RUN_NAME}/final_jmaps/

clean-raw-data: 
	rm -f -r data/${RUN_NAME}/

clean-all :
	rm -f -r data/





