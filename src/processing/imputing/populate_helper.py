import pandas as pd
import numpy as np

def add_imputed_flags(df):
    '''
    Add a flag for each value per column that is NA
    '''
    imputationdf = df.copy()

    for column in imputationdf.columns[imputationdf.isna().any()]:
        imputed_column_name = f'imputed_{column}'
        imputationdf[imputed_column_name] = imputationdf[column].isna().astype(int)

    imputed_columns = imputationdf.filter(like='imputed_')

    return pd.concat([df, imputed_columns], axis=1)
    
def fillna_normal_distribution(column):
    '''
    Function to fill in missing data based on sampling from the normal distribution of the column
    '''
    mean = column.mean()
    std = column.std()
    na_mask = column.isna()
    na_count = na_mask.sum()

    if na_count > 0:
        # Generate random samples from a normal distribution
        random_samples = np.random.normal(loc=mean, scale=std, size=na_count)
        # Fill NAs with the generated random samples
        column.loc[na_mask] = random_samples

    return column


def perturbulate_data(df: pd.DataFrame, fill_method=fillna_normal_distribution, **kwargs):
    '''
    Function to fill NaN values in a DataFrame using the specified fill method.
    If fill_method is 'drop', it drops NaN values.

    Inputs
    ----------
    df: <pd.DataFrame>
        - dataframe that we are perturbulating (dealing with missing data)

    fill_method: <function or string>
        - NOTE: current options include `fillna_normal_distribution`, "drop", and `None`

    '''
    if fill_method == 'drop':
        return df.dropna()
    elif fill_method is not None:
        return df.copy().apply(fill_method, axis=0)
    else:
        return df
    
def fillna_average(column):
    '''
    Function to fill in missing data in a column based on its average
    '''
    average = column.mean()
    column.fillna(average, inplace=True)
    return column


def fillna_average(column):
    '''
    Function to fill in missing data in a column based on its average
    '''
    average = column.mean()
    column.fillna(average, inplace=True)
    return column


def fillna_mode(column):
    '''
    Function to fill in missing data in a column based on its mode (most frequent value)
    '''
    mode_value = column.mode().iloc[0]  # Get the first mode if multiple modes exist
    column.fillna(mode_value, inplace=True)
    return column


def fillna_median(column):
    '''
    Function to fill in missing data in a column based on its median value
    '''
    average = column.median()
    column.fillna(average, inplace=True)
    return column

def populate_nas_filename(run_name, fill_method, number=None):
    """
    Generate a filename for the cleaned and preprocessed data.

    Parameters
    ----------
    fill_method : string, optional
        fill_method pulled from the params.yaml file

    number : int, optional
        The encoding type used for categorical variables. Default is "integer".

    Returns
    -------
    str
        A filename for the cleaned and preprocessed data.

    """
    if number is not None:
        number_out = "_"+str(number)
    else:
        number_out = ""

    return f"{run_name}_imputed_data_{fill_method}{number_out}.pkl"