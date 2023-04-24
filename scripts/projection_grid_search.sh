#!/usr/bin/env bash
# UMAP GRID SEARCH

#Run from root/scripts/


N_NEIGHBORS=(3 6 9 12 15 18 21 24 27 30)
MIN_DISTS=(0 0.01 0.05 0.1 0.5 1)


poetry shell
echo "Initializing Poetry Shell"
echo "Would you like to clean out existing projections? yes or no"

read clean

if [ $clean == "yes" ];then
    echo "Cleaning..."
    echo -e 
    rm -r ../data/projections/UMAP/
fi


echo "Computing UMAP Projection Grid Search!"
            echo "--------------------------------------------------------------------------------"
            echo -e "Choices for n_neighbors: ${N_NEIGHBORS[@]}"
            echo -e "Choices for min_dist: ${MIN_DISTS[@]}"
            echo "--------------------------------------------------------------------------------"
for N in "${N_NEIGHBORS[@]}"; do
    for D in "${MIN_DISTS[@]}"; do
        python ../src/processing/projecting/projector.py                                    \
                                -n ${N}                                                    \
                                -d ${D}                                                    \
    
    done 
done
echo -e
echo "##################################################################################"
echo -e
echo -e
echo "Finished projection grid search!"
echo "See data/projections/UMAP/ to view the projection pickle files."
echo -e
echo -e
echo "##################################################################################"
echo -e
exit 0
