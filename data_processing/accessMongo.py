# %%
import pandas as pd
import pymongo

# %%
def df_to_mongodb(client, database:str, col:str, df):
    '''client - insert your pymongo.MongoClient token here
    database - name of the database you are accessing
    col - name of collection within database'''
    
    client = pymongo.MongoClient(client)
    db = client[database]
    collection = db[col]

    data = df.to_dict("records")

    # Insert the data into the MongoDB collection
    collection.insert_many(data)

# %%
def mongodb_to_df(client, database:str, col:str):
    '''client - insert your pymongo.MongoClient token here
    database - name of the database you are accessing
    col - name of collection within database'''

    client = pymongo.MongoClient(client)
    db = client[database]
    collection = db[col]

    documents = list(collection.find())
    # Convert the list of documents into a Pandas DataFrame
    return pd.DataFrame(documents).drop(columns={'_id'})



