# File: /src/fitting/fitting_utils.py
# Lasted Updated: 03-04-24
# Updated By: SW

def generate_gModel_filename(gen_method, **kwargs):
    """
    Generates a filename based on generation method and corresponding parameters. 
    
    Returns
    -----------
    output_file: str
        A unique filename to identify gModels.

    """

    if gen_method == "jmap":    
        output_file = f"{gen_method}_ncubes{kwargs["nn"]}_percOverlap{int(kwargs["percOverlap"]*100)}_{kwargs["clusterer"]}_UMAP_Nbors_minDist{kwargs["minDist"]}_min_int{kwargs["minIntersection"]}.pkl"

    elif gen_method == "pyball":
        output_file = f"{gen_method}_eps{kwargs["eps"]}" 

    return output_file




def convert_keys_to_alphabet(dictionary):
    """Simple Helper function to make node labels more readable."""
    base = 26  # Number of letters in the alphabet
    new_dict = {}

    keys = list(dictionary.keys())
    for i, key in enumerate(keys):
        # Calculate the position of each letter in the new key
        position = i
        new_key = ""
        while position >= 0:
            new_key = chr(ord('a') + (position % base)) + new_key
            position = (position // base) - 1

        new_dict[new_key] = dictionary[key]

    return new_dict
