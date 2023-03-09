#!/usr/bin/env bash
# CURVATURE ANALYSIS

#Run from src/ directory

K_VALS=(6 7 8 9 10 11 12 13 14 15 )
N_CUBES=(7 8 9 10 11 12 13 14 15 16 17 18 19)
PERC_OVERLAP=(0.5)
MIN_INTERSECTION=(1 2 3 4 5 6 7 8 9 10) 


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