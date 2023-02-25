#!/usr/bin/env bash
#
# Create data sets for upcoming diffusion condensation publication. Feel
# free to add additional scenarios here, but check that each output name
# is unique---else, the script will overwrite everything.

SEED=2021
N_POINTS= 128 # How many data points fo we have? Will we need to subsample?

DATASET = ../data_processing/data #Replace with txt file 
KERNELS=(box) # alpha box gaussian laplacian)
# Lets try box first as it has the quickest convergence

for KERNEL in "${KERNELS[@]}"; do
echo "Running condensation for $DATASET with '$KERNEL' kernel..."
    poetry run python condensation.py --kernel ${KERNEL}                                         \
                                    --data ${DATASET}                                          \
                                    -s ${SEED}                                                 \
                                    -n ${N_POINTS}                                             \
                                    -c CalculateDiffusionHomology CalculatePersistentHomology  \
                                    -o data/publication/${DATASET}_${KERNEL}_n${N_POINTS}.npz

done