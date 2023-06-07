#!/bin/bash


# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    eval "$params" 
fi


# Calling model_selector script  
poetry run python ../../src/modeling/model_selector.py -H

