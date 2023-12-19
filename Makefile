# Makefile

.PHONY: all 
all:    init process-data projections jmaps jmap-selection 
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
fetch:    fetch-raw-data fetch-processed-data

.PHONY: fetch-raw-data 
fetch-raw-data:   
	poetry run python src/processing/pulling/data_generator.py -v

.PHONY: process-data 
process-data:    init
	cd scripts/bash && ./cleaner.sh

.PHONY: projections 
projections:    init 
	cd scripts/bash && ./projector.sh

.PHONY: summarize-projections
summarize-projections:    init 
	poetry run python src/modeling/synopsis/projection_summarizer.py

.PHONY: jmaps 
jmaps:    init 
	cd scripts/bash && ./jmap_generator.sh 

.PHONY: jmap-histogram 
jmap-histogram:     init
	cd scripts/python && poetry run python jmap_histogram.py

.PHONY: curvature-distances
curvature-distances:     init
	cd scripts/python && poetry run python curvature_distance_generator.py

.PHONY: curvature-histogram
curvature-histogram:    init curvature-distances
	cd scripts/python && poetry run python curvature_histogram.py

.PHONY: stability-histogram 
stability-histogram:    init
	cd scripts/python && poetry run python stability_histogram.py

.PHONY: dendrogram
dendrogram:    init
	cd scripts/bash && ./dendrogram.sh

.PHONY: jmap-clustering 
jmap-clustering:   init curvature-distances
	cd scripts/python && poetry run python clusterer.py

.PHONY: jmap-selection 
jmap-selection:    init jmap-clustering
	cd scripts/python && poetry run python selector.py


init: 
	cd scripts/bash && ./startup.sh

save: 
	cd scripts/bash && ./dvc_save.sh 

load: 
	cd scripts/bash && ./dvc_load.sh


# Cleaning commands for data fields 


.PHONY: clean 
clean: 
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh 

.PHONY: clean-processed-data
clean-processed-data:    
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d clean

.PHONY: clean-projections
clean-projections:    
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d projections

.PHONY: clean-jmaps
clean-jmaps:   
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d jmaps

.PHONY: clean-jmap-analysis
clean-jmap-analysis:    
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d jmap_analysis

.PHONY: clean-final-jmaps
clean-final-jmaps:    
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d final_jmaps

.PHONY: clean-raw-data
clean-raw-data:   
	cd scripts/bash/makefile_helpers/ && ./sweeper.sh -d raw

#  Checks 
.PHONY: check-poetry 
check-poetry:
	@which poetry || (echo "Poetry is not installed. Installing..."; sudo pip install poetry; poetry install;)

	


	 



