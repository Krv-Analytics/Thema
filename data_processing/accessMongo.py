import pandas as pd
import pymongo


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


def mongo_pull(
    client,
    database="cleaned",
    col="coal_mapper",
    one_hot=True,
    type="csv",
    filepath="./local_data/",
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

    # Encode Categorical Variables
    if one_hot:
        df = pd.get_dummies(df, prefix="One_hot", prefix_sep="_")
        file = filepath + col + "_one_hot"
    else:
        file = filepath + col

    # Generate Output Files
    if type == "csv":
        df.to_csv(
            file + ".csv",
            index=None,
        )
    elif type == "pkl":
        df.to_pickle(
            file + ".pkl",
        )
    return file


def mongo_rename(client, database: str, col: str, new_name: str):
    """rename a mongo collection
    requires admin access to mongodb"""
    client = pymongo.MongoClient(client)
    database = client[database]
    collection = database[col]
    collection.rename(new_name, dropTarget=True)
