import pandas as pd
import pymongo
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
from dotenv import load_dotenv

load_dotenv()
my_client = os.getenv("mongo_client")

def df_to_mongodb(client, database: str, col: str, df):
    """client - insert your pymongo.MongoClient token here
    database - name of the database you are accessing
    col - name of collection within database"""

    client = pymongo.MongoClient(client)
    db = client[database]
    collection = db[col]

    data = df.to_dict("records")

    # Insert the data into specified MongoDB collection
    collection.insert_many(data)


def mongo_getRawData(
    client=my_client,
    database="cleaned",
    col="coal_mapper",
    TSNE = True,
):
    """this function returns a complete coal_mapper dataframe with readable values (not scaled, projected, or encoded)"""
    try:
        client = pymongo.MongoClient(client)
    except:
        return "Could not connect to MongoDB"

    db = client[database]
    collection = db[col]
    documents = list(collection.find())

    # Convert the list of documents into a Pandas DataFrame
    if TSNE == True:
        dataDict = [
        "ORISPL",
        "coal_FUELS",
        "NONcoal_FUELS",
        "ret_DATE",
        "PNAME",
        "FIPSST",
        "PLPRMFL",
        "FIPSCNTY",
        "LAT",
        "LON",
        "Utility ID",
        "Entity Type",
        "STCLPR",
        "STGSPR",
        "SECTOR",
        ]
        df = pd.DataFrame(documents).drop(columns={"_id"})
        temp=df.copy()
        df.drop(dataDict, axis=1, inplace=True)
        temp.drop(columns=[col for col in temp if not col in dataDict], inplace=True)

        df = df.dropna()
        return df.merge(temp, how='left', left_index=True, right_index=True)
    else:
        return pd.DataFrame(documents).drop(columns={"_id"})



def mongo_pull(
    client,
    filepath,
    database="cleaned",
    col="coal_mapper",
    one_hot=True,
    scaled=True,
    TSNE_project=True,
    type="csv",
):
    """This function creates a local file containing the specified dataset

    client - insert your pymongo.MongoClient token here \n
    database - name of the database you are accessing \n
    col - name of collection within database \n

    DATASET OPTIONS: \n
    – coal_mapper is a complied dataset of all information \n
    – eGrid_coal is a compiled dataset of a yearly instance of every US coal plant since 2009"""

    try:
        client = pymongo.MongoClient(client)
    except:
        return "Could not connect to MongoDB"

    db = client[database]
    collection = db[col]

    documents = list(collection.find())

    # Convert the list of documents into a Pandas DataFrame
    df = pd.DataFrame(documents).drop(columns={"_id"})
    temp = df.copy()

    dataDict = [
        "ORISPL",
        "coal_FUELS",
        "NONcoal_FUELS",
        "ret_DATE",
        "PNAME",
        "FIPSST",
        "PLPRMFL",
        "FIPSCNTY",
        "LAT",
        "LON",
        "Utility ID",
        "Entity Type",
        "STCLPR",
        "STGSPR",
        "SECTOR",
    ]
    df.drop(dataDict, axis=1, inplace=True)
    temp.drop(columns=[col for col in temp if not col in dataDict], inplace=True)
    file = col

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

    # TSNE project the data into 2 dimensions
    if TSNE_project:
        features = df.dropna()
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(features)
        df = pd.DataFrame(projections, columns=["x", "y"])
        file += "_TSNE"

    output_path = os.path.join(filepath, file)
    # Generate Output Files
    if type == "csv":

        df.to_csv(
            output_path + ".csv",
            index=None,
        )
        temp.to_csv(
            output_path + "_dict" + ".csv",
            index=None,
        )

    elif type == "pkl":
        df.to_pickle(
            output_path + ".pkl",
        )

    return file


def mongo_rename(client, database: str, col: str, new_name: str):
    """rename a mongo collection
    requires admin access to mongodb"""
    client = pymongo.MongoClient(client)
    database = client[database]
    collection = database[col]
    collection.rename(new_name, dropTarget=True)
