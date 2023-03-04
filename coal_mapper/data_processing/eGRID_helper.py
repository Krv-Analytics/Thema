import pandas as pd

def drop_excess (id, sheet:str):
    '''Put the df on a weight loss program
    '''
    if sheet == "GEN":
        cols = [
        'ORISPL', 
        'GENSTAT', 
        'FUELG1', 
        'NAMEPCAP', 
        'CFACT', 
        'GENNTAN', 
        'GENYRONL', 
        'GENYRRET']
    elif sheet == "PLNT":
        cols = [
        'YEAR',
        'PNAME',
        'ORISPL', 
        'FIPSST',
        'FIPSCNTY', 
        'LAT', 
        'LON', 
        'NUMGEN', 
        'PLPRMFL', 
        'COALFLAG', 
        'CAPFAC', 
        'NAMEPCAP', 
        'NBFACTS', 
        'PLNGENAN', 
        'PLNOXAN', 
        'PLCO2AN', 
        'PLSO2AN', 
        'PLNGENOZ', 
        'PLGENACL', 
        'PLCLPR',
        'SECTOR']

    id.drop(columns=[col for col in id if col not in cols], inplace=True)
    return id

def findYear(str):
        """Finds the eGrid Data year within the file name passed to the function.
        Used for plant labeling purposes – seperating active from decomissioned plants"""
        str = str.lower()
        for i in range(0, len(str)):
            if str[i : i + 5] == "egrid":
                return str[i + 5 : i + 9]

def makeYearintoSheetName(str, sheet:str):
    """Due to the nature of the EIA eGrid datasets, each .xlsx file has many sheet names.
    This function ensures the PLNT sheet is selected and read in"""
    year = findYear(str)
    if sheet == "GEN":
        return "GEN" + year[2:4]
    elif sheet == "PLNT":
        return "PLNT" + year[2:4]

def PLNT_mergeYears(list):
    '''list is a list of eGRID sheet names to combine'''
    headerList = [4,4,4,1,1,1,1,1,1]
    plntDF = pd.DataFrame()
    for egrid in list:
        temp = drop_excess(pd.read_excel('eia_egrid/'+egrid, sheet_name= makeYearintoSheetName(egrid,"PLNT"), header=headerList[list.index(egrid)]), "PLNT")
        temp['YEAR']=findYear(egrid)
        plntDF = pd.concat([plntDF, temp])
    plntDF['YEAR']=plntDF['YEAR'].astype(int)
    return plntDF

def GEN_mergeYears(list):
    '''list is a list of eGRID sheets to combine'''

    headerList = [5,5,5,1,1,1,1,1,1]
    genDF = pd.DataFrame()
    for egrid in list:
        temp = pd.read_excel('eia_egrid/'+egrid, sheet_name= makeYearintoSheetName(egrid,"GEN"), header=headerList[list.index(egrid)])
        #temp = temp[temp['FUELG1'].isin(coalFuels)]
        temp = drop_excess(temp, "GEN")
        temp['YEAR']=findYear(egrid)
        genDF = pd.concat([genDF, temp])
    return genDF

def egrid_PLNT(list):
    return PLNT_mergeYears(list)

def egrid_coalGEN(list):
    return GEN_mergeYears(list)

def clean_coalflag(id):
    if id!=id:
        return 0
    elif id == 0:
        return 0
    elif id == 1:
        return 1
    elif id == 'Yes':
        return 1

def retirement(id):
    if id!=id:
        return 0
    elif id == 'Yes':
        return 1
    elif id=='No':
        return 2

def yearify(id):
    return id.year

def defineRetrofitCosts(costs:pd.DataFrame):
    costs['Inservice Year'] = costs['Inservice Year'].replace(" ", 0).fillna(0).astype(int)
    costs['Total Cost (Thousand Dollars)']=costs['Total Cost (Thousand Dollars)'].replace(" ", 0).fillna(0).astype(int)
    costs = costs[costs['Inservice Year']>=2012]
    return costs.groupby('Plant Code')['Total Cost (Thousand Dollars)'].sum().reset_index().rename(columns={'Total Cost (Thousand Dollars)': 'Retrofit Costs'})

def numgen (id, temp):
    return temp.loc[temp['ORISPL'] == id, 'NAMEPCAP'].iloc[0]

def weighted_average(df, values, weights):
    return sum(df[weights] * df[values]) / df[weights].sum()

def consolidateGen(allgen:pd.DataFrame, year):
    gen=allgen[(allgen['YEAR']==year)&(allgen['GENSTAT']!='RE')]
    genSummed = gen[['ORISPL', 'NAMEPCAP', 'GENNTAN']].groupby('ORISPL').sum()
    genCapfac = pd.DataFrame(gen.groupby('ORISPL').apply(weighted_average, 'CFACT', 'GENNTAN'), columns=['weighted_coal_CAPFAC']).reset_index()
    genAge = pd.DataFrame(int(year)-gen.groupby('ORISPL').apply(weighted_average, 'GENYRONL', 'NAMEPCAP'), columns=['weighted_coal_AGE']).reset_index()
    genMerged = pd.merge(pd.merge(genSummed, genCapfac, on='ORISPL'), genAge, on='ORISPL')
    genMerged = genMerged.merge(gen[(gen['YEAR']==year)&(gen['GENSTAT']!='RE')].groupby('ORISPL')['FUELG1'].apply(list).reset_index(name='coal_FUELS'), on='ORISPL')

    return genMerged.merge(gen[['ORISPL', 'NAMEPCAP']].groupby('ORISPL').count().reset_index().rename(columns={'NAMEPCAP':'num_coal_GENS'}), on='ORISPL')

def consolidate_genYears(allgen, e_list):
    out = pd.DataFrame()
    for sheet in e_list:
        year = str(findYear(sheet))
        temp = consolidateGen(allgen[allgen['YEAR']==year], year)
        temp['YEAR']=year
        out = out.append(temp)
    return out

def get_nonCoalGens(all_gens, c_gens, e_list):
    nonCoal = pd.DataFrame()
    coalFuels = ['BIT', 'LIG', 'RC', 'SGC', 'SUB', 'WC']
    for sheet in e_list:
        year = str(findYear(sheet))
        temp = all_gens[(all_gens['YEAR']==year)&(~all_gens['FUELG1'].isin(coalFuels))&(all_gens['ORISPL'].isin(c_gens['ORISPL']))&(all_gens['GENSTAT']!='RE')&(all_gens['GENNTAN']>0)].groupby('ORISPL')['FUELG1'].apply(list).reset_index(name='NONcoal_FUELS')
        temp['YEAR']=year
        nonCoal = nonCoal.append(temp)
    return nonCoal

def add_nonCoalGens(eia, nonCoal, all_plnts):
    temp = pd.merge(eia, nonCoal,  how='left', left_on=['ORISPL', 'YEAR'], right_on = ['ORISPL', 'YEAR']).fillna(0)
    temp['YEAR'] = temp['YEAR'].astype(int)
    return pd.merge(temp, drop_excess(all_plnts, "PLNT"), left_on=['ORISPL', 'YEAR'], right_on=['ORISPL', 'YEAR'])
    
#####compile egrid dataset using all the above functions#####

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