#!/usr/bin/env bash
# UMAP GRID SEARCH

#Run from root/scripts/

# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    
    eval "$params" 
fi

poetry run python ../python/projection_grid.py

echo -e
exit 0
