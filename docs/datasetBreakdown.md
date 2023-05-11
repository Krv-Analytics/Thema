### **Dataset Information**
A breakdown of all variables included in our dataset, their source, and which variables are being passed into the pipeline.

### **Dataset**
|variable       |meaning                      |source         |raw or calculated|calculation                 |pipeline variable?  |
|---------------|-----------------------------|---------------|-----------------|----------------------------|--------------------|
ORISPL          |EIA ORIS Plant/Facility Code |EIA eGRID      |raw              |                            |no                  |
NAMEPCAP        |Plant Nameplate Capacity (MW)|EIA eGRID      |raw              |                            |yes                 |
GENNTAN         |Generator Annual Net Generation (MWh)|EIA eGRID|raw            |                            |yes                 |
weighted_coal_CAPFAC|Weighted Coal Capacity Factor|EIA eGRID|calculated|Capacity factor of each generator within a plant, weighted by its 2020 net generation, averaged across a plant|yes|
weighted_coal_AGE|Weighted Average Age of Coal Plant|EIA eGRID|calculated|Age of each coal generator within a plant, weighted by its nameplate capacity, averaged across a plant|yes|
coal_FUELS      |Type of coal fuel burned by each coal generator within a plant |EIA eGRID|raw|              |no                  |
num_coal_GENS   |Number of Coal Generators per Plant|EIA eGRID|raw              |EIA eGRID|raw|              |yes                 |
NONcoal_FUELS   |Non-coal used Fuels at a Plant with Coal Generators|EIA eGRID|raw|                          |no                  |
ret_STATUS      |Coal Retirement Status       |Sierra Club    |semi-calculated  |No proposed retirements endoced with 0, Entire Plant Proposed Retirement encoded with 1, Patrial Plant Proposed Retirement encoded with 2 -- Fuel replacement/conversion counted as retirement|yes|
ret_DATE
Retrofit Costs
PNAME	PLGENACL
PLPRMFL
PLCLPR
FIPSST
FIPSCNTY
LAT
LON
STCLPR
STGSPR
2020_netCashflow
aveCashFlow
forwardCosts
PLSO2AN
SECTOR
Utility ID
Entity Type
securitization_policy
governor_party
legislation_majority_party
CO2limitsOppose
2018to2021change_CO2opposition
Mortality (high estimate)
Mortality (high estimate) DAC
Hospital Admits, All Respiratory
Hospital Admits, All Respiratory DAC
