#!/usr/bin/env bash
# Cleaning script to drive src/processing/cleaning/cleaner.py
# Note: Must be run from scripts/ directory! 


# Load the .env file
if [ -f ../../.env ]; then
    source ../../.env
fi



if [ ! -d "${root}/data/clean" ]; then
    mkdir "${root}/data/clean"
fi

# Calling cleaning and imputing scripts. Configure options in params.yaml
poetry run python ../../src/processing/cleaning/cleaner.py  -v  
poetry run python ../../src/processing/imputing/imputer.py  -v  

