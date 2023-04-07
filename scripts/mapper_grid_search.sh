#!/usr/bin/env bash
# MAPPER GRID SEARCH

#Run from root/scripts/


MIN_CLUSTER_SIZES=(6)
N_CUBES=(4 5 6 7)
PERC_OVERLAP=(0.45 0.55 0.65 0.7)
MIN_INTERSECTION=(1 2 3 4)

PROJECTIONS="../data/projections/UMAP/*.pkl"

poetry shell
echo "Initializing Poetry Shell"
echo "Would you like to clean out existing mapper objects? yes or no"

read clean

if [ $clean == "yes" ];then
    echo "Cleaning..."
    rm -r ../data/mappers/
fi


echo "Computing Mapper Parameter Grid Search, over all available projections!"
            echo "--------------------------------------------------------------------------------"
            echo -e "Choices for min_cluster_size: ${MIN_CLUSTER_SIZES[@]}"
            echo -e "Choices for n_cubes: ${N_CUBES[@]}"
            echo -e "Choices for perc_overlap: ${PERC_OVERLAP[@]}"
            echo -e "Choices for min_intersection: ${MIN_INTERSECTION[@]}"
            echo "--------------------------------------------------------------------------------"
for PROJECTION in $PROJECTIONS; do
for MIN_CLUSTER_SIZE in "${MIN_CLUSTER_SIZES[@]}"; do
    for N in "${N_CUBES[@]}"; do
        for P in "${PERC_OVERLAP[@]}"; do
            echo -e 
            echo -e "Computing mapper with $N cubes and $P% overlap."
            python ../src/modeling/coal_mapper_generator.py                                    \
                                    -n ${N}                                                    \
                                    -p ${P}                                                    \
                                    --min_cluster_size ${MIN_CLUSTER_SIZE}                     \
                                    --projection ${PROJECTION}                                 \
                                    --min_intersection ${MIN_INTERSECTION[@]}                  \
                                   #Using Default min_intersection for now
            done 
        done
    done
done
exit 0
