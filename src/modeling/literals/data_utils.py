
import pandas as pd 
import numpy as np


####################################################################################
# 
#  node_desription Helper functions
# 
####################################################################################



def get_minimal_std(df: pd.DataFrame, mask: np.array, density_cols=None):
    """Find the column with the minimal standard deviation
    within a subset of a Dataframe.

    Parameters
    -----------
    df: pd.Dataframe
        A cleaned dataframe.

    mask: np.array
        A boolean array indicating which indices of the dataframe
        should be included in the computation.

    Returns
    -----------
    col_label: int
        The index idenitfier for the column in the dataframe with minimal std.

    """
    if density_cols is None:
        density_cols = df.columns
    sub_df = df.iloc[mask][density_cols]
    col_label = sub_df.columns[sub_df.std(axis=0).argmin()]
    return col_label



####################################################################################
# 
#  group_identity Helper functions
# 
####################################################################################



def std_zscore_threshold_filter(col, global_means:dict(), std_threshold = 1, zscore_threshold = 1): 
    """
    TODO: Fill out Doc String
    """
    std = np.std(col)
    zscore = (np.mean(col) - global_means[col.name])/std
    

    if zscore > zscore_threshold and std < std_threshold:
        return 1 
    else:
        return 0



def get_best_std_filter(col, global_means:dict()):
    """
    TODO: Fill out Doc String
    """
    std = np.std(col)    
    return std


def get_best_zscore_filter(col, global_means:dict()):
    """
    TODO: Fill out Doc String
    """
    zscore = (np.mean(col) - global_means[col.name])/np.std(col)

    return zscore


####################################################################################
# 
#  Auxillary functions
# 
####################################################################################


#NOTE: Unneccessary? 
def _define_zscore_df(jmap):
    df_builder = pd.DataFrame()
    dfs = jmap.tupper.clean

    column_to_drop = [col for col in dfs.columns if dfs[col].nunique() == 1]
    dfs = dfs.drop(column_to_drop, axis=1)

    dfs['cluster_IDs']=(list(jmap.cluster_ids))

    #loop through all policy group dataframes
    for group in list(dfs['cluster_IDs'].unique()):
        zscore0 = pd.DataFrame()
        group0 = dfs[dfs['cluster_IDs']==group].drop(columns={'cluster_IDs'})
        #loop through all columns in a policy group dataframe
        for col in group0.columns:
            if col != "cluster_IDs":
                mean = dfs[col].mean()
                std = dfs[col].std()
                zscore0[col] = group0[col].map(lambda x: (x-mean)/std)
        zscore0_temp = zscore0.copy()
        zscore0_temp['cluster_IDs'] = group
        df_builder = pd.concat([df_builder,zscore0_temp])
    return df_builder