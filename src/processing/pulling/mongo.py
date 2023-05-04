import pandas as pd
import pymongo


class Mongo:
    """A class to interact with MongoDB for data insertion, retrieval, and renaming.

    Attributes
    ----------
    _client : pymongo.MongoClient
        A MongoClient instance for accessing a MongoDB instance.

    Methods
    -------
    push(database: str, col: str, df) -> None
        Inserts data into a specified MongoDB collection.

    pull(database="cleaned", col="coal_mapper") -> pd.DataFrame
        Returns a complete, raw DataFrame with readable values.

    rename(database: str, col: str, new_name: str) -> None
        Renames a MongoDB collection (requires admin access to MongoDB).
    """

    def __init__(self, client_id: str):
        """Constructor for Mongo class.

        Parameters
        ----------
        client_id : str
            A string representing the MongoDB client ID.
        """
        self._client = pymongo.MongoClient(client_id)

    def push(self, database: str, col: str, df) -> None:
        """Inserts data into a specified MongoDB collection.

        Parameters
        ----------
        database : str
            The name of the database on MongoDB.
        col : str
            The name of the collection in the database.
        df : pandas.DataFrame
            The data to insert into the specified MongoDB collection.
        """
        try:
            db = self._client[database]
            collection = db[col]
            data = df.to_dict("records")
            collection.insert_many(data)
        except:
            print("Failed MongoDB access!")
            print("Please check that MongoDB is properly configured in your .env file.")

    def pull(self, database="cleaned", col="comprehensive_coal_data") -> pd.DataFrame:
        """Returns a complete DataFrame with readable values (not scaled, projected, or encoded).

        Parameters
        ----------
        database : str, optional
            The name of the database on MongoDB (default is 'cleaned').
        col : str, optional
            The name of the collection in the database (default is 'coal_mapper').

        Returns
        -------
        pandas.DataFrame
            A DataFrame with raw values.
        """
        try:
            db = self._client[database]
            collection = db[col]
            documents = list(collection.find())
            return pd.DataFrame(documents).drop(columns={"_id"})
        except:
            print("Failed MongoDB access!")
            print("Please check that MongoDB is properly configured in your .env file.")

    def rename(self, database: str, col: str, new_name: str) -> None:
        """Renames a MongoDB collection (requires admin access to MongoDB).

        Parameters
        ----------
        database : str
            The name of the database on MongoDB.
        col : str
            The name of the collection in the database.
        new_name : str
            The new name for the collection.
        """
        try:
            db = self._client[database]
            collection = db[col]
            collection.rename(new_name, dropTarget=True)
        except:
            print("Failed MongoDB access!")
            print("Please check that MongoDB is properly configured in your .env file.")
