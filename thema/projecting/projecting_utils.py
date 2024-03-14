# File: src/projecting/projection_utils.py 
# Last Update: 03-13-24
# Updated by: SW 

def projection_file_name(
    projector,
    id=None,
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
        
        output_file = f'{projector}_{kwargs["dimensions"]}D_{kwargs["nn"]}nn_{kwargs["minDist"]}minDist_{kwargs["seed"]}rs__{id}.pkl'
    
    if projector == "TSNE": 
        output_file = f'{projector}_{kwargs["dimensions"]}D_{kwargs["perplexity"]}perp_{kwargs["seed"]}rs__{id}.pkl'
    
    if projector == "PCA":
        output_file = f'{projector}_{kwargs["dimensions"]}D_{kwargs["seed"]}rs_{id}.pkl'
    return output_file
