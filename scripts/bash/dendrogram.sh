#!/bin/bash


# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    eval "$params" 
fi

echo "Which jmap group size would you like to visualize?"
read NUM_GROUPS

# Calling cleaning script from params.json 
poetry run python ../../src/tuning/metrics/metric_generator.py -n ${NUM_GROUPS}
poetry run python ../../src/tuning/graph_clustering/jmap_clusterer.py -n ${NUM_GROUPS}


