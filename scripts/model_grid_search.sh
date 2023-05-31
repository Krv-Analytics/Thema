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

RAW="$root$raw_data"
CLEAN="$root$clean_data" 
PROJECTIONS="$root$path_to_projections"

# PARAMETER GRID
MIN_CLUSTER_SIZES=(${model_min_cluster_sizes//,/ })
N_CUBES=(${model_nCubes//,/ })
PERC_OVERLAP=(${model_percOverlap//,/ })
MIN_INTERSECTIONS=(${model_minIntersection//,/ })

echo "$CLEAN"
echo "$PROJECTIONS"

# Filtering Command line arguments 
# TODO: There has to be a cleaner way to do this 

MIN_CLUSTER_SIZES=(${MIN_CLUSTER_SIZES[@]//\[/})
MIN_CLUSTER_SIZES=(${MIN_CLUSTER_SIZES[@]//\]/})

N_CUBES=(${N_CUBES[@]//\[/})
N_CUBES=(${N_CUBES[@]//\]/})

PERC_OVERLAP=(${PERC_OVERLAP[@]//\[/})
PERC_OVERLAP=(${PERC_OVERLAP[@]//\]/})

MIN_INTERSECTIONS=(${MIN_INTERSECTIONS[@]//\[/})
MIN_INTERSECTIONS=(${MIN_INTERSECTIONS[@]//\]/})


poetry shell
echo "Initializing Poetry Shell"
# echo "Would you like to clean out existing mapper objects? yes or no"

# read clean

# if [ $clean == "yes" ];then
#     echo "Cleaning..."
#     echo -e 
#     rm -r ../data/models/
# fi

echo "Computing Mapper Parameter Grid Search, over all available projections!"
            echo "--------------------------------------------------------------------------------"
            echo -e "Choices for min_cluster_size: ${MIN_CLUSTER_SIZES[@]}"
            echo -e "Choices for n_cubes: ${N_CUBES[@]}"
            echo -e "Choices for perc_overlap: ${PERC_OVERLAP[@]}"
            echo -e "Choices for min_intersection: ${MIN_INTERSECTIONS[@]}"
            echo "--------------------------------------------------------------------------------"

for MIN_INTERSECTION in $MIN_INTERSECTIONS; do            
    for PROJECTION in "$PROJECTIONS"/*; do
    # for MIN_CLUSTER_SIZE in "${MIN_CLUSTER_SIZES[@]}"; do
    #     for N in "${N_CUBES[@]}"; do
    #         for P in "${PERC_OVERLAP[@]}"; do
    #             echo -e 
    #             echo -e "Computing mapper with $N cubes and $P% overlap."
    #             python ../src/modeling/model_generator.py --projection ${PROJECTION} -n ${N} -p ${P} --min_cluster_size ${MIN_CLUSTER_SIZE} --min_intersection ${MIN_INTERSECTION}                                                 
    #                                 #Using Default min_intersection for now
                
    #             done
    #         done 
    #     done
        echo "$PROJECTION"
    done
done
exit 0
