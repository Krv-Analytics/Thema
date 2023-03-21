#!/usr/bin/env bash
# CURVATURE ANALYSIS

#Run from src/ directory


MIN_CLUSTER_SIZES=(10)
N_CUBES=(7 8 9 10 11)
PERC_OVERLAP=(0.5)
MIN_INTERSECTION=(1) 

poetry shell
echo "Initializing Poetry Shell"
#TODO: Need to add umap to Poetry, for now need to run normal python within VE

for MIN_CLUSTER_SIZE in "${MIN_CLUSTER_SIZES[@]}"; do
    for N in "${N_CUBES[@]}"; do
        for P in "${PERC_OVERLAP[@]}"; do
echo "Running curvature analysis on Mapper graphs generated from K = $K_VAL clusters and a cover with $N cubes that overlap by $P %."

python compute_curvature.py                                                         \
                                    -n ${N}                                                    \
                                    -p ${P}                                                    \
                                    --min_cluster_size ${MIN_CLUSTER_SIZE}                                                 \
                                    -v                                                         \
                                    --min_intersection ${MIN_INTERSECTION[@]}                  \
                                   #Using Default min_intersection for now
        done
    done
done

exit 0