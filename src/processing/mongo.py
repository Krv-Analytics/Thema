import pandas as pd
import pymongo


class Mongo:
    def __init__(
        self,
        client_id: str,
    ):
        self._client = pymongo.MongoClient(client_id)

    def push(self, database: str, col: str, df):
        """database - name of the database on Mongo
        col - name of collection in database"""
        try:
            db = self._client[database]
            collection = db[col]

            data = df.to_dict("records")

            # Insert the data into specified MongoDB collection
            collection.insert_many(data)
        except:
            print("Failed Mongo access!")
            print("Are you sure you've configured Mongo correctly in your .env?")

    def pull(
        self,
        database="cleaned",
        col="coal_mapper",
    ):
        """this function returns a complete coal_mapper dataframe with
        readable values (not scaled, projected, or encoded)"""
        try:
            db = self._client[database]
            collection = db[col]
            documents = list(collection.find())
            return pd.DataFrame(documents).drop(columns={"_id"})
        except:
            print("Failed Mongo access!")
            print("Are you sure you've configured Mongo correctly in your .env?")

    def rename(self, database: str, col: str, new_name: str):
        """rename a mongo collection
        requires admin access to mongodb"""
        try:
            database = self._client[database]
            collection = database[col]
            collection.rename(new_name, dropTarget=True)
        except:
            print("Failed Mongo access!")
            print("Are you sure you've configured Mongo correctly in your .env?")
