#!/usr/bin/env python
# coding: utf-8

# class coal_mapper_dataset

import pandas as pd
import pymongo
from accessMongo import *
from eGRID_helper import *

def getCompiledData(mongo_client):
    '''returns a cleaned dataframe containing the most recent year's coal data from multiple sources'''
    return mongodb_to_df(client=mongo_client,database= "merged",col= "EIA")

def getEIA_allCoalData(mongo_client):
    '''returns a cleaned dataframe of 2008-2021 coal plant/generator data. Includes one instance of a plant per year, including plants that are no longer active'''
    return mongodb_to_df(client=mongo_client,database= "eGRID",col= "eGRID_allCoal")

##to create/update the datasets in mongo##

def coalFromList(e_list:list):
    '''This function creates the eGRID_allCoal datasheet contained in MongoDB. Returns a DF of all EIA eGRID coal data from 2009 onwards.
    Takes a sorted list containing eGRID data sheets. List must contain data from 2009 and later – earlier data cannot be processed by this function

    List can be formatted as follows:
        os.chdir('/Users/<folder containing EIA eGRID data sheets>')
        e_list = sorted(os.listdir(<folder containing EIA eGRID data sheets>))
        e_list.remove(".DS_Store")'''

    print(e_list)
    coalFuels = ['BIT', 'LIG', 'RC', 'SGC', 'SUB', 'WC']
    all_plnts = egrid_PLNT(e_list)
    all_gens = egrid_coalGEN(e_list)
    c_gens=all_gens.copy()
    c_gens = c_gens[c_gens['FUELG1'].isin(coalFuels)]
    coal_plnts = all_plnts
    coal_plnts['COALFLAG']=coal_plnts['COALFLAG'].apply(clean_coalflag)
    tester = add_nonCoalGens(consolidate_genYears(c_gens, e_list), get_nonCoalGens(all_gens, c_gens, e_list), all_plnts)
    return tester.fillna('not reported')