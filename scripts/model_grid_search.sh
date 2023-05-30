#!/usr/bin/env bash
# MODEL GRID SEARCH

#Run from root/scripts/

#USER INPUTS
RAW=$1
CLEAN=$2
PROJECTIONS=$3


# PARAMETER GRID
MIN_CLUSTER_SIZES=(6)
N_CUBES=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
PERC_OVERLAP=(.3 .325 .35 .375 .4 .425 .45 .475 .5 .525 .55 .575 .6 .625 .65 .675 .7)
MIN_INTERSECTIONS=(1 2 3 4)


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
    for PROJECTION in $PROJECTIONS; do
    for MIN_CLUSTER_SIZE in "${MIN_CLUSTER_SIZES[@]}"; do
        for N in "${N_CUBES[@]}"; do
            for P in "${PERC_OVERLAP[@]}"; do
                # echo -e 
                # echo -e "Computing mapper with $N cubes and $P% overlap."
                python ../src/modeling/model_generator.py                                          \
                                        --raw ${RAW}                                               \
                                        --clean ${CLEAN}                                           \
                                        --projection ${PROJECTION}                                 \
                                        -n ${N}                                                    \
                                        -p ${P}                                                    \
                                        --min_cluster_size ${MIN_CLUSTER_SIZE}                     \
                                        --min_intersection ${MIN_INTERSECTION}                  \
                                        --script                                                   \
                                    #Using Default min_intersection for now
                done
            done 
        done
    done
done
exit 0
