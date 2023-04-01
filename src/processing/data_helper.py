import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def get_data(
    mongo_object,
    out_dir,
    one_hot=True,
    scaled=True,
):
    """This function creates a local file containing the specified dataset

    database - name of the database you are accessing \n
    col - name of collection within database \n

    DATASET OPTIONS: \n
    – coal_mapper is a complied dataset of all information \n
    – eGrid_coal is a compiled dataset of a yearly instance of every US coal plant since 2009"""

    df = mongo_object.pull()

    file = "coal_plant_data"
    # Encode Categorical Variables
    if one_hot:
        df = pd.get_dummies(df, prefix="One_hot", prefix_sep="_")
        file += "_one_hot"

    # Scale Data using StandardScaler
    if scaled:
        scaler = StandardScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data, columns=list(df.columns))
        file += "_scaled"

    output_path = os.path.join(out_dir, file)
    # Generate Output Files

    df.to_pickle(
        output_path + ".pkl",
    )

    return file
