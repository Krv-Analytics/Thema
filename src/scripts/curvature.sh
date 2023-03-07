#!/usr/bin/env bash
# CURVATURE ANALYSIS

#Run from src/ directory

K_VALS=(3)
N_CUBES=(8)
PERC_OVERLAP=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
MIN_INTERSECTION=(1) 


for K_VAL in "${K_VALS[@]}"; do
    for N in "${N_CUBES[@]}"; do
        for P in "${PERC_OVERLAP[@]}"; do
echo "Running curvature analysis on Mapper graphs generated from K = $K_VAL clusters and a cover with $N cubes that overlap by $P %."

poetry run python compute_curvature.py                                                         \
                                    -n ${N}                                                    \
                                    -p ${P}                                                    \
                                    -K ${K_VAL}                                                \
                                    -v                                                         \
                                    --min_intersection ${MIN_INTERSECTION[@]}                  \
                                   #Using Default min_intersection for now
        done
    done
done

exit 0