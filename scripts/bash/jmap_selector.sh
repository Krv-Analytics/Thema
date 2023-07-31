#!/bin/bash


# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    eval "$params" 
fi

echo "Please enter a policy_group size:"
read NUM_GROUPS

# Calling cleaning script from params.json 
poetry run python ../../src/tuning/graph_clustering/jmap_clusterer.py -n ${NUM_GROUPS} -s
poetry run python ../../src/jmapping/jmap_selector.py -n ${NUM_GROUPS} -v



