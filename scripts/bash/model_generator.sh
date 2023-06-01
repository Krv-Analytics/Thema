#!/usr/bin/env bash
# MODEL GRID SEARCH

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

if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    
    eval "$params" 
fi

poetry run python ../python/model_grid.py


echo -e
echo "##################################################################################"
echo -e
echo -e
echo "Finished Model grid search!"
echo "See data/models/ to view the generated files, subgrouped by number of clusters."
echo -e
echo -e
echo "##################################################################################"
echo -e
exit 0
