#!/usr/bin/env bash
# MAPPER GRID SEARCH

#Run from root/scripts/


MIN_CLUSTER_SIZES=(6)
N_CUBES=(3 4 5 6 7 8 9 10 11)
PERC_OVERLAP=(0.45 0.5 0.55)
MIN_INTERSECTION=(1)

PROJECTIONS = # GRAB FILES IN PROJECTIONS_DIR

poetry shell
echo "Initializing Poetry Shell"
#TODO: Need to add umap to Poetry, for now need to run normal python within VE

echo "Computing Mapper Parameter Grid Search!"
            echo "--------------------------------------------------------------------------------"
            echo -e "Choices for n_cubes: ${N_CUBES[@]}"
            echo -e "Choices for perc_overlap: ${PERC_OVERLAP[@]}"
            echo "--------------------------------------------------------------------------------"
for MIN_CLUSTER_SIZE in "${MIN_CLUSTER_SIZES[@]}"; do
    for N in "${N_CUBES[@]}"; do
        for P in "${PERC_OVERLAP[@]}"; do
            echo -e 
            echo -e "Computing mapper with $N cubes and $P% overlap."
            python ../src/modeling/coal_mapper_generator.py                                    \
                                    -n ${N}                                                    \
                                    -p ${P}                                                    \
                                    --min_cluster_size ${MIN_CLUSTER_SIZE}                     \
                                    -v                                                         \
                                    --min_intersection ${MIN_INTERSECTION[@]}                  \
                                   #Using Default min_intersection for now
        done
    done
done
exit 0
