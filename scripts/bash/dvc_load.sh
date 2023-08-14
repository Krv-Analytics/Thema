#!/bin/bash

# Description: Automated retrieval of results from dvc store. 

# Define some color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#####################################################################################
#   Loading .env file 
#####################################################################################

# Load the .env file
if [ -f ../../.env ]; then
    source ../../.env
else
    echo -e "${RED}Unable to locate .env file. ${NC}" 
    exit 1
fi

#####################################################################################
#   Retrieving User Data 
#####################################################################################

echo "Please specifiy the name of the results folder you wish to load from our DVC Store."
read Run_Name 

abs_path=$(find "$dvc_store" -type d -name "$Run_Name" -print -quit)

# Check if the directory was found
if [ -n "$abs_path" ]; then
  target_folder="${abs_path#*/dvc_store/}"
  echo -e "${GREEN}Succesfully located '$Run_Name' directory: $relative_path. ${NC}"
  echo "Initiating load of $Run_Name results from dvc store repository: $dvc_store"
  echo -e "${YELLOW}Warning: continuining will overwrite your params.json file. ${NC}"
  echo "Would you like to proceed? [y/n]" 
  read PROCEED
else
  echo -e "${RED}Directory '$Run_Name' not found in DVC Store.${NC}"
  exit 0
fi


#####################################################################################
#   Pulling from DVC REPO 
#####################################################################################

# pulling clean data 

if [ -f "$abs_path/clean.dvc" ]; then 
    echo "Pulling clean data..."
    cd "$dvc_store" && dvc "pull" "$target_folder/clean/" 
fi 

# pulling projected data 
if [ -f "$abs_path/projections.dvc" ]; then 
    echo "Pulling projections..."
    cd "$dvc_store" && dvc "pull" "$target_folder/projections/" 
fi 

# pulling jmaps 
if [ -f "$abs_path/jmaps.dvc" ]; then 
    echo "Pulling jmaps" 
    cd "$dvc_store" && dvc "pull" "$target_folder/jmaps/" 
fi 

# pulling jmap analysis objects 
if [ -f "$abs_path/jmap_analysis.dvc" ]; then 
    echo "Pulling jmap_analysis" 
    cd "$dvc_store" && dvc "pull" "$target_folder/jmap_analysis/" 
fi 

# pulling logs 
if [ -f "$abs_path/logs.dvc" ]; then 
    echo "Pulling logs" 
    cd "$dvc_store" && dvc "pull" "$target_folder/logs/" 
fi 


#####################################################################################
#   Copying to data/
#####################################################################################
 
rsync -rvm --exclude='*.dvc' "$abs_path" "$root/data/" 


if [ -f "$abs_path/logs/params.json" ]; then
    cp "$abs_path/logs/params.json" "$root/params.json"
else 
    echo -e "${YELLOW}Warning: No params.json file found in logs/. Parameter file has not been updated.${NC}"
fi 


#####################################################################################
#   Cleaning our DVC Store after pulling 
#####################################################################################

if [ -d "$abs_path/clean/" ]; then
    rm -r "$abs_path/clean/"
fi 

if [ -d "$abs_path/projections/" ]; then
    rm -r "$abs_path/projections/"
fi 

if [ -d "$abs_path/jmaps/" ]; then
    rm -r "$abs_path/jmaps/"
fi  

if [ -d "$abs_path/jmap_analysis/" ]; then
    rm -r "$abs_path/jmap_analysis/"
fi 

if [ -d "$abs_path/logs/" ]; then
    rm -r "$abs_path/logs/"
fi 


echo "${GREEN}Successfully loaded results from DVC store. ${NC}" 














