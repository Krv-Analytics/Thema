#!/bin/bash

# Description: Automated save of results to dvc store. 

# Define some color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color


#####################################################################################
#   Checking .env and Params.yaml file paths 
#####################################################################################

# Load the .env file
if [ -f ../../.env ]; then
    source ../../.env
else
    echo -e "${RED}Unable to locate .env file. ${NC}" 
    exit 1
fi

# Read the yaml file and extract the dvc_store and run_Name fields using grep and awk
Run_Name=$(grep -o '"Run_Name": *"[^"]*"' "$params" | awk -F '"' '{print $4}')

#####################################################################################
#   Checking varaible loads from .env and params.yaml files 
#####################################################################################


# Check root variable exists 
if [ -z "$root" ]; then 
    # Does not exist
    echo "${RED}Unsuccessful read of root data field from .env file.${NC}" 
    exit 1 
fi 

# Check that dvc_store variable exists 
if [ -z "$dvc_store" ]; then 
    # Does not exist
    echo "${RED}Unsuccessful read of dvc_store data field from .env file.${NC}"
    exit 1 
fi

# Check that run_Name variable exists  
if [ -z "$Run_Name" ]; then
    # Does not exist 
    echo "${RED}Unsuccessful read of run_Name data field from params.yaml.${NC}" 
    exit 1
fi

#####################################################################################
#   Checking variable loads from .env and params.yaml files 
#####################################################################################


if [ -d "$dvc_store" ]; then
    # Directory exists
    echo ""
    echo "Standing by to save $Run_Name results to: $dvc_store"
    echo "----------------------------------------------------------"
    echo -e "${YELLOW}Warning: continuing will push ${Run_Name} results to the dvc remote: sw_gdrive. ${NC}"
    echo "Would you like to proceed? [y/n]"
    read PROCEED  
    
else
    # Directory does not exist
    echo "$dvc_store"
    echo "Unable to locate your dvc_store repository. Please make sure it exists and that you have 
    specified the correct path in params.yaml. The dvc_store field needs to be set to the relative path 
    from root to the dvc_store repository." 
    exit 1  
fi



if [ $PROCEED == "y" ]; then 

    echo "Please specify the results subfolder" 
    read subfolder

    if [ ! -d "$dvc_store/results/$subfolder" ]; then 
        mkdir $dvc_store/results/$subfolder
    fi

    write_to="$dvc_store/results/$subfolder/"
    echo "----------------------------------------------------------"
    echo "Copying ${Run_Name} results to: $write_to"

    cp -r "$root/data/$Run_Name" "$write_to/"

    if [ -d "$write_to/$Run_Name" ]; then 
        echo -e "${GREEN}Successfully copied Files.${NC}" 
    
    else 
        echo -e "${RED}Error: There was a problem copying your files. ${NC}" 
        exit 1
    fi 

    # Pulling from GitHub to sync changes 

    echo "Pulling from GitHub remote to sync changes" 
    cd "$dvc_store" && git "pull" "origin" "main"

    # Adding Parmeter File and Folders to DVC Store
    echo "$write_to/$Run_Name/clean/" 

    
    if [ -d "$write_to/$Run_Name/clean/" ]; then
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/clean/"
    fi 
    
    if [ -d "$write_to/$Run_Name/projections/" ]; then
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/projections/"
    fi 

    if [ -d "$write_to/$Run_Name/jmaps/" ]; then

    echo "Would you like to save your entire jmap folder? [y/n]" 
    read ans
    if [ ans == "y" ]; then 
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/jmaps/"
    fi
    fi  

    if [ -d "$write_to/$Run_Name/final-jmaps/" ]; then
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/final-jmaps/"
    fi 

    if [ -d "$write_to/$Run_Name/jmap_analysis/" ]; then
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/jmap_analysis/"
    fi 

    if [ -d "$write_to/$Run_Name/logs/" ]; then
    cd "$dvc_store" && dvc "add" "results/$subfolder/$Run_Name/logs/"
    fi 


    echo -e "${GREEN}Successfully added. Beginning push... ${NC}" 


    dvc "push" 


    # Removing Copied Files after push 

    if [ -d "$write_to$Run_Name/clean/" ]; then
        rm -r "$write_to$Run_Name/clean/"
    fi 
    
    if [ -d "$write_to$Run_Name/projections/" ]; then
        rm -r "$write_to$Run_Name/projections/"
    fi 

    if [ -d "$write_to$Run_Name/jmaps/" ]; then
        rm -r "$write_to$Run_Name/jmaps/"
    fi  

    if [ -d "$write_to$Run_Name/jmap_analysis/" ]; then
        rm -r "$write_to$Run_Name/jmap_analysis/"
    fi 
    
    if [ -d "$write_to$Run_Name/final-jmaps/" ]; then
        rm -r "$write_to$Run_Name/final-jmaps/"
    fi 

    if [ -d "$write_to$Run_Name/logs/" ]; then
        rm -r "$write_to$Run_Name/logs/"
    fi 


    echo -e "${GREEN}Successfully began tracking $Run_Name results with DVC. ${NC}"     
    echo ""
    echo "Pushing dvc files to GitHub."
  
    cd "$dvc_store" && git "commit" "-m" "'began tracking $Run_Name results'" 
    cd "$dvc_store" && git "push" "origin" "main"

    echo -e "${GREEN}Success: Your files have been saved. ${NC}"

else
    echo -e "Save cancelled." 
    exit 0 
fi 