import os


def get_data(
    mongo_object,
    out_dir,
    data_base,
    col,
):
    """This function creates a local file containing the specified dataset

    database - name of the database you are accessing \n
    col - name of collection within database \n

    DATASET OPTIONS: \n
    – coal_mapper is a complied dataset of all information \n
    – eGrid_coal is a compiled dataset of a yearly instance of every US coal plant since 2009"""

    df = mongo_object.pull(data_base, col)

    file = "coal_plant_data_raw"

    output_path = os.path.join(out_dir, file)
    # Generate Output Files

    df.to_pickle(
        output_path + ".pkl",
    )

    return file
