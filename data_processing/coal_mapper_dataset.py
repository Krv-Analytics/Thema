#!/usr/bin/env python
# coding: utf-8

# class coal_mapper_dataset

import pandas as pd
import pymongo
from accessMongo import *

def getData(mongo_client):
    return mongodb_to_df(client=mongo_client,database= "Merges",col= "eGRIDconsolidation")