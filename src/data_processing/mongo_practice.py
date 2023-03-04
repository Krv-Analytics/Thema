from fastapi import FastAPI
from dotenv import dotenv_values
from pymongo import MongoClient
from accessMongo import mongodb_to_df

# Take command line argument for which column to process


config = dotenv_values("../.env")

app = FastAPI()


@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient(config["mongo_client"])
    app.database = app.mongodb_client[config["db_name"]]
    print("Connected to the MongoDB database!")


@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()
