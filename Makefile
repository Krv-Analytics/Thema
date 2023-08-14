# Makefile
include .env
PARAMS_FILE := $(strip $(params))
PARAMS_JSON := $(shell cat $(PARAMS_FILE))
RUN_NAME := $(shell echo '$(PARAMS_JSON)' | jq -r '.Run_Name')
COVERAGE_FILTER := $(shell echo '$(PARAMS_JSON)' | jq -r '.coverage_filter')

.PHONY: all 
all: check-params init process-data projections jmaps jmap-selection 
	@echo "Process complete"

.PHONY: install 
install: check-poetry 
	@echo " Generating and populating .env file..."
	python scripts/python/setup.py
	@echo "Installing necessary dependencies..." 
	poetry run pip install python-dotenv pandas pymongo scikit-learn umap-learn hdbscan kmapper networkx matplotlib seaborn giotto-tda

.PHONY: install 
uninstall: clean
	@echo "Removing Dependencies from Poetry environment..."
	poetry run pip uninstall python-dotenv pandas pymongo scikit-learn umap-learn hdbscan kmapper networkx matplotlib seaborn giotto-tda
	@echo "Removing Poetry"
	sudo pip uninstall poetry 
	rm -f .env 

.PHONY: fetch 
fetch: check-params fetch-raw-data fetch-processed-data

.PHONY: fetch-raw-data 
fetch-raw-data: check-params
	poetry run python src/processing/pulling/data_generator.py -v

.PHONY: process-data 
process-data: check-params init
	cd scripts/bash && ./cleaner.sh

.PHONY: projections 
projections: check-params init 
	cd scripts/bash && ./projector.sh

.PHONY: summarize-projections
summarize-projections: check-params init 
	poetry run python src/modeling/synopsis/projection_summarizer.py

.PHONY: jmaps 
jmaps: check-params init 
	cd scripts/bash && ./jmap_generator.sh 

.PHONY: jmap-histogram 
jmap-histogram: check-params  init
	cd scripts/python && poetry run python jmap_histogram.py

.PHONY: curvature-distances
curvature-distances: check-params  init
	cd scripts/python && poetry run python curvature_distance_generator.py

.PHONY: curvature-histogram
curvature-histogram: check-params init curvature-distances
	cd scripts/python && poetry run python curvature_histogram.py

.PHONY: stability-histogram 
stability-histogram: check-params init
	cd scripts/python && poetry run python stability_histogram.py

.PHONY: dendrogram
dendrogram: check-params init
	cd scripts/bash && ./dendrogram.sh

.PHONY: jmap-clustering 
jmap-clustering:check-params init curvature-distances
	cd scripts/python && poetry run python clusterer.py

.PHONY: jmap-selection 
jmap-selection: check-params init jmap-clustering
	cd scripts/python && poetry run python selector.py


init: 
	cd scripts/bash && ./startup.sh

save: 
	cd scripts/bash && ./dvc_save.sh 

load: 
	cd scripts/bash && ./dvc_load.sh


# Cleaning commands for data fields 
.PHONY: clean 
clean: check-params clean-processed-data clean-projections clean-jmaps clean-jmap-analysis clean-final-jmaps
	rm -f -r data/$(RUN_NAME)/

.PHONY: clean-processed-data
clean-processed-data:  check-params
	rm -f -r data/$(RUN_NAME)/clean/*

.PHONY: clean-projections
clean-projections: check-params 
	rm -f -r data/${RUN_NAME}/projections/*

.PHONY: clean-jmaps
clean-jmaps: check-params
	rm -f -r data/${RUN_NAME}/jmaps/*

.PHONY: clean-jmap-analysis
clean-jmap-analysis:  check-params
	rm -f -r data/${RUN_NAME}/jmap_analysis/

.PHONY: clean-final-jmaps
clean-final-jmaps:  check-params
	rm -f -r  data/${RUN_NAME}/final_jmaps/

.PHONY: clean-raw-data
clean-raw-data: check-params
	rm -f -r data/${RUN_NAME}/

.PHONY: clean-all 
clean-all:  
	rm -f -r data/



#  Checks 
.PHONY: check-poetry 
check-poetry:
	@which poetry || (echo "Poetry is not installed. Installing..."; sudo pip install poetry; poetry install;)

.PHONY: check-params 
check-params:
	@if [ ! -f ${PARAMS_FILE} ]; then echo "Error: Paramter file not found!"; exit 1; fi 



