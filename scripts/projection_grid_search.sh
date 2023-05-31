#!/usr/bin/env bash
# UMAP GRID SEARCH

#Run from root/scripts/

# Load the .env file
if [ -f ../.env ]; then
    source ../.env
fi

# Access Parameter Json file 
if [ -n "$JSON_PATH" ]; then 
    params=$(jq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "$JSON_PATH")    
    eval "$params" 
fi



N_NEIGHBORS=(${projector_Nneighbors//,/ })
MIN_DISTS=(${projector_minDists//,/ })

# Remove brackets from N_NEIGHBORS
N_NEIGHBORS=(${N_NEIGHBORS[@]//\[/})
N_NEIGHBORS=(${N_NEIGHBORS[@]//\]/})

# Remove brackets from MIN_DISTS
MIN_DISTS=(${MIN_DISTS[@]//\[/})
MIN_DISTS=(${MIN_DISTS[@]//\]/})

echo "Computing UMAP Projection Grid Search!"
            echo "--------------------------------------------------------------------------------"
            echo -e "Choices for n_neighbors: ${N_NEIGHBORS[@]}"
            echo -e "Choices for min_dist: ${MIN_DISTS[@]}"
            echo "--------------------------------------------------------------------------------"
for N in "${N_NEIGHBORS[@]}"; do
    for D in "${MIN_DISTS[@]}"; do
        python ../src/processing/projecting/projector.py                                 \
                                -n $N                                                    \
                                -d $D                                                
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
