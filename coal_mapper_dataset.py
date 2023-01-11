#!/usr/bin/env python
# coding: utf-8

# class coal_mapper_dataset


def EIA_dataset(choice: str):

    """choice should be "whole" or "2020" to specify whether 2005-2020 or just 2020 data is wanted"""
    import pandas as pd
    import os

    # ### Initial Functions – Cleaning eGrid Data

    def findYear(str):

        """Finds the eGrid Data year within the file name passed to the function.
        Used for plant labeling purposes – seperating active from decomissioned plants"""
        str = str.lower()
        for i in range(0, len(str)):
            if str[i : i + 5] == "egrid":
                return str[i + 5 : i + 9]

    def makeYearintoSheetName(str):
        """Due to the nature of the EIA eGrid datasets, each .xlsx file has many sheet names.
        This function ensures the PLNT sheet is selected and read in"""
        year = findYear(str)
        return "PLNT" + year[2:4]

    def findHeader(df):
        """Because the eGrid datasheets are mnot standardized from year to year, this function is a necessity.
        Finds the row of data labels, allowing us to make it the top of our pd.Dataframe"""
        df2 = df
        test = df2[df2.columns[1]]
        for i in range(0, 10):
            if test[i] == "AK":
                break
            nmu = i
        return nmu

    def isDecom(id):
        if id == "2020":
            return 0
        else:
            return 1

    # comment outlined sections of code to change to switch plant fuel type in the dataset; this study focuses only on coal

    def cleanEIAegrid(str):
        """Reads in and cleans an eGrid .xlsx
        Cleans it for use with the Kepler Mapper"""

        df_n = pd.read_excel(str, makeYearintoSheetName(str), header=1)

        if int(findYear(str)) < 2014:
            df = pd.read_excel(
                str, makeYearintoSheetName(str), header=findHeader(df_n) + 2
            )
        else:
            df = df_n

        eyear = findYear(str)

        #### comment out the below code to create dataset of all plants, not just coal. ####
        #### edit code below to create a dataset of any powerplant fuel category. ####
        df = df[df["PLFUELCT"] == "COAL"]
        df2 = df[df["COALFLAG"] == 1]
        if df2.size < 2:
            df2 = df[df["COALFLAG"] == "Yes"]
        df = df2
        #### end edits section ####

        df["Year"] = eyear
        if eyear == "2020":
            df = df[
                [
                    "Year",
                    "ORISPL",
                    "PSTATABB",
                    "LAT",
                    "LON",
                    "PNAME",
                    "FIPSST",
                    "FIPSCNTY",
                    "CAPFAC",
                    "PLNGENAN",
                    "PLNOXAN",
                    "PLSO2AN",
                    "PLCO2AN",
                    "PLFUELCT",
                    "PLGENACL",
                    "SECTOR",
                ]
            ]
        else:
            df = df[
                [
                    "Year",
                    "ORISPL",
                    "PSTATABB",
                    "LAT",
                    "LON",
                    "PNAME",
                    "FIPSST",
                    "FIPSCNTY",
                    "CAPFAC",
                    "PLNGENAN",
                    "PLNOXAN",
                    "PLSO2AN",
                    "PLCO2AN",
                    "PLFUELCT",
                    "PLGENACL",
                ]
            ]
        df = df.assign(label=eyear + ": " + df["PNAME"] + ", " + df["PSTATABB"])
        df.reset_index(drop=True, inplace=True)
        df = df.dropna()

        return df

    def makeEIADataSet(list):
        """Takes a list of EIA eGRID datasets as the input. Creates a combined coal_mapper dataset out of each excel file.

        EX. list = os.listdir("egrid 2005-2020") where "egrid 2005-2020" is a folder containing all EIA eGrid datasets with a sheet named PLNT05 (or any other year suffix) from the subset of years."""

        df = pd.DataFrame()

        for i in list:
            df = pd.concat([cleanEIAegrid(i), df], axis=0)

        dfCOPY = df
        dfCOPY = dfCOPY.sort_values(["ORISPL", "Year"], ascending=[True, True])
        # dfCOPY = dfCOPY.drop_duplicates(subset = "ORISPL", keep = 'last')
        dfCOPY = dfCOPY.reset_index()
        dfCOPY = dfCOPY.iloc[:, 1:]

        dfCOPY["decom"] = dfCOPY["Year"].apply(isDecom)
        dfCOPY["CarbonIntensity"] = dfCOPY["PLCO2AN"] / dfCOPY["PLNGENAN"]

        dfCOPY = dfCOPY[dfCOPY.CAPFAC > 0]

        return dfCOPY

    # ### An Example of what it looks to refrence the folder of eGrid data used to create the dataset

    list = os.listdir("egrid 2005-2020")
    print(list)

    list.remove(".DS_Store")
    print(list)

    # # Dataset Creation

    if choice == "2020":
        list2 = ["egrid2020_data.xlsx"]
        whole = makeEIADataSet(list2)
    else:
        whole = makeEIADataSet(list)

    return whole


def dataset_addOn(inDF):
    """inDF should be a dataframe recieved from coal_mapper_dataset"""
    import pandas as pd
    import numpy as np

    alldf = inDF
    alldf["percentCoal"] = alldf["PLGENACL"] / alldf["PLNGENAN"]

    podf = pd.read_csv("YCOM_2020_Data.csv")
    podf = podf[["GEOID", "CO2limits", "CO2limitsOppose"]]

    dac = pd.read_excel("power_plants_and_communities.xlsx")

    rodf = pd.read_excel(
        "eia-860_2020/6_1_EnviroAssoc_Y2020.xlsx",
        sheet_name="Emissions Control Equipment",
        header=1,
    )

    gen = pd.read_excel(
        "egrid 2005-2020/egrid2020_data.xlsx", "GEN20", header=1
    ).reset_index()

    # ### Add Yale YCOM_2020 Climate Change Public Opinion Dataset – county level data

    # ###### Due to the use of a GEOID and not state and county FIPS codes, these functions create a GEOID from the state and county FIPS that is compatible with the GEOID format of the data in the YCOM dataset. The goal of this is to join the datasets on GEOID so the powerplant county locations have YCOM county-level public opinion data associated with them

    ## copy and pasted from: https://code.activestate.com/recipes/577775-state-fips-codes-dict/

    state_codes = {
        "WA": "53",
        "DE": "10",
        "DC": "11",
        "WI": "55",
        "WV": "54",
        "HI": "15",
        "FL": "12",
        "WY": "56",
        "PR": "72",
        "NJ": "34",
        "NM": "35",
        "TX": "48",
        "LA": "22",
        "NC": "37",
        "ND": "38",
        "NE": "31",
        "TN": "47",
        "NY": "36",
        "PA": "42",
        "AK": "02",
        "NV": "32",
        "NH": "33",
        "VA": "51",
        "CO": "08",
        "CA": "06",
        "AL": "01",
        "AR": "05",
        "VT": "50",
        "IL": "17",
        "GA": "13",
        "IN": "18",
        "IA": "19",
        "MA": "25",
        "AZ": "04",
        "ID": "16",
        "CT": "09",
        "ME": "23",
        "MD": "24",
        "OK": "40",
        "OH": "39",
        "UT": "49",
        "MO": "29",
        "MN": "27",
        "MI": "26",
        "RI": "44",
        "KS": "20",
        "MT": "30",
        "MS": "28",
        "SC": "45",
        "KY": "21",
        "OR": "41",
        "SD": "46",
    }

    def clean_NaN(df):
        return df.drop_duplicates("ORISPL").fillna("")

    def add_YCOM(alldf, podf):
        """podf is the YCOM dataframe"""

        def addZeros(id: str, num: int):
            for i in range(0, num):
                id = "0" + id
            return id

        def create_GEOID(PSTATABB, FIPSCNTY):
            code = int(state_codes[PSTATABB])
            stcode = str(code)

            ctcode = str(FIPSCNTY)
            if len(ctcode) != 3:
                num = 3 - len(ctcode)
                ctcode = addZeros(ctcode, num)

            outcode = stcode + ctcode
            return int(outcode)

        alldf["FIPSCNTY"] = alldf["FIPSCNTY"].astype(int)
        alldf["GEOID"] = alldf.apply(
            lambda x: create_GEOID(x.PSTATABB, x.FIPSCNTY), axis=1
        )

        combineddf = alldf.merge(podf, how="left", left_on="GEOID", right_on="GEOID")
        del combineddf["GEOID"]

        combineddf = clean_NaN(combineddf)
        return combineddf

    combineddf = add_YCOM(alldf, podf)

    combineddf.head()

    # ### Add EIA 860 environmental controls/retrofits cost dataset to each powerplant

    def clean_EIA860(retrodf: pd.DataFrame):
        retrodf = retrodf[retrodf["Inservice Year"] > "2004"]
        retrodf = retrodf[retrodf["Retirement Year"] < "1900"]

        retrodf = retrodf.rename(
            columns={
                "Plant Code": "ORISPL",
                "Total Cost (Thousand Dollars)": "post2004RetrofitCosts",
            }
        )

        retrodf["post2004RetrofitCosts"] = (
            retrodf["post2004RetrofitCosts"].replace(" ", 0) * 1000
        )
        retrodf.drop(retrodf.tail(1).index, inplace=True)

        retrodf["post2004Retrofit"] = 1
        retrodf = (
            retrodf.groupby("ORISPL")["post2004RetrofitCosts", "post2004Retrofit"]
            .sum()
            .reset_index()
        )
        return retrodf

    retrodf = clean_EIA860(rodf)

    combineddf = combineddf.merge(
        retrodf, how="left", left_on="ORISPL", right_on="ORISPL"
    )
    combineddf = combineddf.fillna(0)

    combineddf.head()

    # ### Add EJ Screen Data – combined with eGrid data, this adds plant-level demographic data to each correspionding powerplant. Data sourced from: https://experience.arcgis.com/experience/2e3610d731cb4cfcbcec9e2dcb83fc94

    def clean_powerplantsEJSCREEN(dac: pd.DataFrame):
        dac = dac.rename(
            columns={
                "DOE/EIA ORIS plant or facility code": "ORISPL",
                "PM 2.5 Emssions (tons)": "PM_emissions",
                "total population (ACS2018)": "popWithin3miles",
                "National percentile for Demographic Index": "percDemographicIndex",
            }
        )
        dac = dac[
            [
                "ORISPL",
                "popWithin3miles",
                "percDemographicIndex",
                "PM_emissions",
                "PM 2.5 Emission Rate (lb/MWh)",
            ]
        ]
        return dac

    dac = clean_powerplantsEJSCREEN(dac)

    comboDF = combineddf.merge(dac, how="left", left_on="ORISPL", right_on="ORISPL")
    comboDF = comboDF.fillna("1")

    comboDF.head()

    # ### Adding individual generator level data to each powerplant. The goal is to create a more accurate powerplant age variable based on the age of opperating generators and determine whether a powerplant is partially retired (x number of retired coal generators).

    gen = pd.read_excel(
        "egrid 2005-2020/egrid2020_data.xlsx", "GEN20", header=1
    ).reset_index()

    indivL = gen["ORISPL"].tolist()
    plantL = comboDF["ORISPL"].tolist()

    gen_list = []
    gen2 = gen
    count = 0
    for x in indivL:
        if x in plantL:
            count += 1
            gen_list.append(x)

    gen2 = gen2[gen2["ORISPL"].isin(gen_list)]

    def removeNonCoal(gen2):
        gen2 = gen2[
            (gen2.FUELG1 != "WND")
            & (gen2.FUELG1 != "WDS")
            & (gen2.FUELG1 != "WAT")
            & (gen2.FUELG1 != "SUN")
            & (gen2.FUELG1 != "RFO")
            & (gen2.FUELG1 != "NG")
            & (gen2.FUELG1 != "JF")
            & (gen2.FUELG1 != "DFO")
            & (gen2.FUELG1 != "NUC")
            & (gen2.FUELG1 != "BLQ")
        ]
        return gen2

    gen2 = removeNonCoal(gen2)

    gen2["GENYRRET"] = gen2["GENYRRET"].fillna(3000)

    ageDF = gen2
    ageDF = ageDF[ageDF["GENYRRET"] > 2020]

    def resetGENYRRET(id):
        if id == 3000:
            return ""
        else:
            return id

    ageDF["GENYRRET"] = ageDF["GENYRRET"].apply(resetGENYRRET)

    aveAge = pd.DataFrame(ageDF.groupby("ORISPL")["GENYRONL"].mean())

    def p_ret(id):
        if id == "RE":
            return 1

    gen2["numRetiredGenerators"] = gen2["GENSTAT"].apply(p_ret)

    p_retDF = pd.DataFrame(gen2.groupby("ORISPL")["numRetiredGenerators"].sum())

    genTotals = pd.DataFrame(gen2.groupby("ORISPL")["GENNTAN"].sum())

    dfc2 = comboDF.merge(genTotals, how="left", left_on="ORISPL", right_on="ORISPL")
    dfc2 = dfc2.merge(p_retDF, how="left", left_on="ORISPL", right_on="ORISPL")
    dfc2 = dfc2.merge(aveAge, how="left", left_on="ORISPL", right_on="ORISPL")

    def ret(id):
        if id > 0:
            return 1
        else:
            return 0

    dfc2["PartiallyRetired"] = dfc2["numRetiredGenerators"].apply(ret)

    dfc2 = dfc2[dfc2["PSTATABB"] != "PR"]

    # ### Create Powerplant Community Impact Metrics and define a powerplant impact score

    dfc2 = dfc2.fillna(1)

    dfc2 = dfc2.replace("", 1)

    dfc2[
        [
            "post2004RetrofitCosts",
            "post2004Retrofit",
            "popWithin3miles",
            "percDemographicIndex",
            "PM_emissions",
            "PM 2.5 Emission Rate (lb/MWh)",
        ]
    ] = dfc2[
        [
            "post2004RetrofitCosts",
            "post2004Retrofit",
            "popWithin3miles",
            "percDemographicIndex",
            "PM_emissions",
            "PM 2.5 Emission Rate (lb/MWh)",
        ]
    ].astype(
        float
    )
    dfc2["popWithin3miles"] = dfc2["popWithin3miles"].replace(0, 1)

    test = dfc2.replace("", 1)

    test["affectedDemographicIndex"] = (
        test["popWithin3miles"] * test["percDemographicIndex"]
    )
    test = test.fillna("")

    test["CAP_level"] = np.log(
        (test["PM_emissions"].astype(float)) + test["PLNOXAN"] + test["PLSO2AN"]
    )
    test["CAP_level"] = abs(
        (test["CAP_level"]) / (test["CAP_level"].mean()) * 100
    )  # CAP: Criteria Air Pollutant
    test["pop_level"] = np.log(test["popWithin3miles"])
    test["pop_level"] = abs((test["pop_level"]) / (test["pop_level"].mean()) * 100)

    test["impact_score"] = test["pop_level"] * test["CAP_level"]

    test.head()

    # ### Add LCOE from the Coal Cost Crossover Study, using CCC data

    ccc = pd.read_excel("Coal-Cost-Crossover-Dataset-2021.xlsx", header=1)
    ccc = ccc.rename(columns={"EIA Plant ID": "ORISPL"})
    ccc = ccc[
        [
            "ORISPL",
            "Average Unit Age (years)",
            "Total Coal Going-Forward Cost",
            "Confidence Tier",
        ]
    ]
    df_all = test.merge(ccc.drop_duplicates(), on="ORISPL", how="left", indicator=True)

    test = df_all
    test = test.rename(columns={"Total Coal Going-Forward Cost": "forwardCosts"})
    # del test['Unnamed: 0']

    return test
