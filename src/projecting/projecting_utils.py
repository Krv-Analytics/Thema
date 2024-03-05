# File: src/projecting/projection_utils.py 
# Last Update: 03-04-24
# Updated by: SW 

def projection_file_name(
    projector,
    impute_method=None,
    impute_id=None,
    **kwargs
):
    """
    This function generates a filename for a projected dataset.

    Parameters:
    -----------
    projector : str
        The projection method used.

    **kwargs 

    TODO: udpate doc string 
    Returns:
    -----------
    str
        The filename for the projected dataset.
    """
    if projector == "UMAP":
        
        output_file = f"{projector}_{kwargs["dimensions"]}D_Nbors{kwargs["nn"]}_minDist_{kwargs["minDist"]}_{kwargs["seed"]}rs_{impute_method}_{impute_id}.pkl"
    
    if projector == "TSNE": 
        output_file = f"{projector}_{kwargs["dimensions"]}D_{kwargs["perplexity"]}perp_{kwargs["seed"]}rs_{impute_method}_{impute_id}.pkl"
    
    if projector == "PCA":
        output_file = f"{projector}_{kwargs["dimensions"]}D_{kwargs["seed"]}rs_{impute_method}_{impute_id}.pkl"
    
    return output_file
