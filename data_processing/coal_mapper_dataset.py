#!/usr/bin/env python
# coding: utf-8

# class coal_mapper_dataset

from accessMongo import *

def getCompiledData(mongo_client):
    '''returns a cleaned dataframe containing the most recent year's coal data from multiple sources'''
    return mongodb_to_df(client=mongo_client,database= "merged",col= "EIA")

def getEIA_allCoalData(mongo_client):
    '''returns a cleaned dataframe of 2008-2021 coal plant/generator data. Includes one instance of a plant per year, including plants that are no longer active'''
    return mongodb_to_df(client=mongo_client,database= "eGRID",col= "eGRID_allCoal")


