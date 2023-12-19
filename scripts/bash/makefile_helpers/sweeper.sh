#!/bin/bash
# Process command-line options
while getopts ":d:" opt; do
  case $opt in
    d)
      dir="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Load the .env file
if [ -f ../../../.env ]; then
    source ../../../.env
else
    echo "${RED}Unable to locate .env file. ${NC}" 
    exit 1
fi

# Check if the params file is defined in .env
if [ -z "$params" ]; then
    echo "Params variable is not defined in .env."
    exit 1
fi

# Check if the params file exists
if [ ! -f "$params" ]; then
    echo "Params file not found: $params."
    exit 1
fi

# Check if the params file exists
if [ -z "$root" ]; then
    echo "Root not found in .env file."
    exit 1
fi

Run_Name=$(grep -o 'Run_Name: *[^ ]*' "$params" | awk -F ': *' '{print $2}' | tr -d '[:space:]')

# Check if Run_Name is empty
if [ -z "$Run_Name" ]; then
    echo -e "${RED}Run_Name is empty in $params.${NC}"
    exit 1
fi

# Define file-path to remove 
file_path="${root}data/${Run_Name}/${dir}" 

if [ -d $file_path ]; then
    rm -rf ${file_path}
fi  

echo "rm -rf ${Run_Name}/${dir}"
