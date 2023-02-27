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
    client, database="cleaned", col="coal_mapper", type="csv", filepath="./local_data/"
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

    if type == "csv":
        pd.DataFrame(documents).drop(columns={"_id"}).to_csv(
            filepath + col + ".csv",
            index=None,
        )
        return f"file saved to {filepath+col}.csv"
    elif type == "pkl":
        temp = pd.DataFrame(documents).drop(columns={"_id"})
        temp.to_pickle(
            filepath + col + ".pkl",
        )
        return f"Pickle file saved to {filepath+col}.pkl"


def mongo_rename(client, database: str, col: str, new_name: str):
    """rename a mongo collection
    requires admin access to mongodb"""
    client = pymongo.MongoClient(client)
    database = client[database]
    collection = database[col]
    collection.rename(new_name, dropTarget=True)
