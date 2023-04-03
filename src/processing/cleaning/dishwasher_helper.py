import pandas as pd


def washing_cycle(data:pd.DataFrame,scaler=None,column_filter=[],encoding="integer"):
    

    
    



# Encode Categorical Variables
    if one_hot:
        df = pd.get_dummies(df, prefix="One_hot", prefix_sep="_")
        file += "_one_hot"

    # Scale Data using StandardScaler
    if scaled:
        scaler = StandardScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data, columns=list(df.columns))
        file += "_scaled"


      dataDict = [
        "ORISPL",
        "coal_FUELS",
        "NONcoal_FUELS",
        "ret_DATE",
        "PNAME",
        "FIPSST",
        "PLPRMFL",
        "FIPSCNTY",
        "LAT",
        "LON",
        "Utility ID",
        "Entity Type",
        "STCLPR",
        "STGSPR",
        "SECTOR",
        ]