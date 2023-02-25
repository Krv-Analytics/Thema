#!/usr/bin/env bash
# Condensation Analysis
SEED=2023
DATASET=../notebooks/coal_mapper.txt #Replace with txt file 
KERNELS=(box alpha) # box gaussian laplacian)



echo "How many samples of your dataset would you like to use in the condensation algorithm?"

read N_POINTS
for KERNEL in "${KERNELS[@]}"; do
echo "Running condensation for $N_POINTS samples of $DATASET with '$KERNEL' kernel..."
    poetry run python ../coal_mapper/pecan/condensation.py --kernel ${KERNEL}                                         \
                                    --data ${DATASET}                                          \
                                    -s ${SEED}                                                 \
                                    -n ${N_POINTS}                                             \
                                    -c CalculateDiffusionHomology CalculatePersistentHomology  \
                                    -o ../outputs/condensation/${KERNEL}_n${N_POINTS}.npz --force

done
