import os


def get_raw_data(
    mongo_object,
    out_dir,
    data_base,
    col,
):
    """
    Pull raw data from MongoDB and create a local file
    storing the specified dataset.

    Parameters
    ----------
    mongo_object : Mongo
        A Mongo object instance connected to a MongoDB database.
    out_dir : str
        The path to the output directory where the file will be saved.
    data_base : str
        The name of the MongoDB database where the dataset is located.
    col : str
        The name of the collection within the specified MongoDB database.

    Returns
    -------
    str
        The name of the saved file.

    Dataset Options
    ---------------
    coal_mapper : pd.DataFrame
        A compiled dataset of all information.
    eGrid_coal : pd.DataFrame
        A compiled dataset of a yearly instance of every
        US coal plant since 2009.
    """
    df = mongo_object.pull(data_base, col)

    file = "raw_values"

    output_path = os.path.join(out_dir, file)
    # Generate Output Files

    df.to_pickle(
        output_path + ".pkl",
    )

    return file
