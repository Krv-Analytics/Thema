#!/bin/bash

echo "Please enter a policy_group size:"
read NUM_GROUPS

# Calling cleaning script from params.yaml 
poetry run python ../../src/tuning/graph_clustering/jmap_clusterer.py -n ${NUM_GROUPS} -s
poetry run python ../../src/jmapping/jmap_selector.py -n ${NUM_GROUPS} -v



