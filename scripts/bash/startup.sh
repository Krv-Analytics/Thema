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
# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "yq is not installed. Please install it first. (On Mac, 'brew install yq')"
    exit 1
fi

# Extract Run_Name using yq
Run_Name=$(yq eval '.Run_Name' "$params")

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

results="$root$Run_Name"

if [ ! -d "$results/" ]; then
mkdir "$results"
mkdir "$results/logs"
fi

cp "$root/params.yaml" "$results/logs/params.yaml"



# else 

# Update Params.yaml
cp "$root/params.yaml" "$results/logs/params.yaml" 

