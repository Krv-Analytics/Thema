#!/usr/bin/env bash
# CONDENSATION ANALYSIS
    #Scripts from the PECAN Library: https://github.com/KrishnaswamyLab/PECAN

SEED=2023
KERNELS=(box) # box gaussian laplacian)

echo "Please indicate the path to the dataset you would like to run Condensation on:"
read DATASET

for KERNEL in "${KERNELS[@]}"; do
echo "Running condensation for $N_POINTS samples of $DATASET with '$KERNEL' kernel..."
    poetry run python ./pecan/condensation.py --kernel ${KERNEL}                               \
                                    --data ${DATASET}                                          \
                                    -s ${SEED}                                                 \                                           \
                                    -c CalculateDiffusionHomology CalculatePersistentHomology  \
                                    -o ../outputs/condensation/${KERNEL}_n${N_POINTS}.npz --force

done
