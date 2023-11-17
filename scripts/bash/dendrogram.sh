#!/bin/bash


# Load the .env file
if [ -f ../../.env ]; then
    source ../../.env
fi

# Access Parameter YAML file
if [ -n "$YAML_PATH" ]; then
    params=$(yq r "$YAML_PATH" -j | jq -r 'to_entries | .[] | "export \(.key)=\(.value)"')
    eval "$params"
fi

echo "Which jmap group size would you like to visualize?"
read NUM_GROUPS

# Calling cleaning script from params.yaml
poetry run python ../../src/tuning/metrics/metric_generator.py -n ${NUM_GROUPS}
poetry run python ../../src/tuning/graph_clustering/jmap_clusterer.py -n ${NUM_GROUPS}


