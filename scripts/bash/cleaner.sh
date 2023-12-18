#!/usr/bin/env bash
# Cleaning script to drive src/processing/cleaning/cleaner.py
# Note: Must be run from scripts/ directory! 

# Calling cleaning and imputing scripts. Configure options in params.yaml
poetry run python ../../src/processing/cleaning/cleaner.py  -v  
poetry run python ../../src/processing/imputing/imputer.py  -v  

