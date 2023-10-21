#!/bin/bash

# Description: Inititalization script for an instance THEMA

# Colors 
# Define some color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
ORANGE='\033[0;33;1m'
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


# Read the JSON file and extract the dvc_store and run_Name fields using grep and awk
Run_Name=$(grep -o '"Run_Name": *"[^"]*"' "$params" | awk -F '"' '{print $4}')


results="$root/data/$Run_Name" 
 
 if [ ! -d "$results/" ]; then

# First Run

echo ""
echo "---------------------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
echo -e "${GREEN}                =======================                                             "
echo -e "${GREEN}                =======================                                             " 
echo -e "${GREEN}                          |||                                                       " 
echo -e "${GREEN}                          |||                                                       " 
echo -e "${GREEN}                          |||   |      |    _____                                   " 
echo -e "${GREEN}                          |||   |      |   |        |\    /|      /\                " 
echo -e "${GREEN}                          |||   |______|   |        | \  / |     /  \               " 
echo -e "${GREEN}                          |||   |      |   |----    |  \/  |    /____\              " 
echo -e "${GREEN}                          |||   |      |   |        |      |   /      \                              "
echo -e "${GREEN}                          |||   |      |   |_____   |      |  /        \                             " 
echo -e "${NC}                                                                                                      "
echo "                                                                 by Krv Analytics                              "
echo "                                                                                                               "
echo "---------------------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------"
echo "                                                                                                               "
 
mkdir "$results"
mkdir "$results/logs"
cp "$root/params.json" "$results/logs/params.json"



# else 

# Update Params.json
cp "$root/params.json" "$results/logs/params.json" 


fi 