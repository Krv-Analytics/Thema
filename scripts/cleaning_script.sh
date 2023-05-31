#!/usr/bin/env bash
# Cleaning script to drive src/processing/cleaning/cleaner.py
# Note: Must be run from scripts/ directory! 


# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    eval "$params" 
fi

if [ ! -d "$root/data/clean" ]; then
    mkdir -p "$root/data/clean"
fi

# Calling cleaning script from params.json 
python ../src/processing/cleaning/cleaner.py  -v  

