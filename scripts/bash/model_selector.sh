#!/bin/bash


# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    eval "$params" 
fi

echo "Which model group size would you like to select token models?"
read NUM_GROUPS

# Calling cleaning script from params.json 
poetry run python ../../src/tuning/graph_clustering/model_clusterer.py -n ${NUM_GROUPS} -s
poetry run python ../../src/modeling/model_selector.py -n ${NUM_GROUPS}


