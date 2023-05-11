### **Dataset Information**
A breakdown of all variables included in our dataset, their source, and which variables are being passed into the pipeline.

### **Dataset**
|variable       |meaning/info                 |source         |raw or calculated|calculation                 |pipeline variable?  |
|---------------|-----------------------------|---------------|-----------------|----------------------------|--------------------|
ORISPL          |EIA ORIS Plant/Facility Code |EIA eGRID      |raw              |                            |no                  |
NAMEPCAP        |Plant Nameplate Capacity (MW)|EIA eGRID      |raw              |                            |yes                 |
GENNTAN         |Generator Annual Net Generation (MWh)|EIA eGRID|semi-calculated|Summed for all coal generators within a plant|yes|
weighted_coal_CAPFAC|Weighted Coal Capacity Factor (%)|EIA eGRID|calculated|Capacity factor of each generator within a plant, weighted by its 2020 net generation, averaged across a plant|yes|
weighted_coal_AGE|Weighted Average Age of Coal Plant (years)|EIA eGRID|calculated|Age of each coal generator within a plant, weighted by its nameplate capacity, averaged across a plant|yes|
coal_FUELS      |Type of coal fuel burned by each coal generator within a plant |EIA eGRID|raw|              |no                  |
num_coal_GENS   |Number of Coal Generators per Plant|EIA eGRID|raw              |                            |yes                 |
NONcoal_FUELS   |Non-coal used Fuels at a Plant with Coal Generators|EIA eGRID|raw|                          |no                  |
ret_STATUS      |Coal Retirement Status |Sierra Club    |raw  |(Full Retirement Announced, Partial Retirement Announced, No Retirement Announced -- fuel transitions/conversions are considered retirements)|yes|
ret_DATE        |Date of Announced Retirement (partial or full)|Sierra Club|raw |                            |no                  |  
Retrofit Costs  |Total Cost of Emissions Control Equiptment Retrofits Installed since 2012 ($)|EIA 860|semi-calculated|Costs summed across all coal generators within a plant|yes|
PNAME           |Plant Name                   |EIA eGRID      |raw              |                            |no                  |
PLGENACL        |Plant Annual Coal Net Generation (MWh)|EIA eGRID|raw           |                            |yes                 |
PLPRMFL         |Plant Primary Fuel           |EIA eGRID      |raw              |                            |yes                 |
PLCLPR          |Plant Coal Generation Percent (resource mix)|EIA eGRID|raw     |                            |yes                 |
FIPSST          |Plant FIPS State Code        |EIA eGRID      |raw              |                            |no                  |
FIPSCNTY        |Plant FIPS County Code       |EIA eGRID      |raw              |                            |no                  |
LAT             |Plant Latitude               |EIA eGRID      |raw              |                            |no                  |
LON             |Plant Longitude              |EIA eGRID      |raw              |                            |no                  |
STCLPR          |State Coal Generation Percent (resource mix)|EIA eGRID|raw     |                            |no                  |
STGSPR          |State Gas Generation Percent (resource mix) |EIA eGRID|raw     |                            |no                  |
2020_netCashflow|2020 Net Cash Flow           |RMI            |semi-calculated  |Summed monthly cash flows throughout 2020|yes    |
aveCashFlow     |Ave Cash Flow since 2012     |RMI            |semi-calculated  |Average of net montly cash flow per plant|yes    |
forwardCosts    |Total Coal Cost Going Forward ($/MWh)|Energy Innovation|raw    |see [Coal Cost Crossover 3.0](https://energyinnovation.org/publication/the-coal-cost-crossover-3-0/) for detailed methodology explanation|yes|
PLSO2AN         |Plant Annual SO2 Emissions (tons)|EIA eGRID  |raw              |                            |yes                 |
SECTOR          |Plant-level Sector           |EIA eGRID      |raw              |                            |no                  |
Utility ID      |Utility ID Number            |EIA 860        |raw              |                            |no                  |
Entity Type     |Utility Entity Type          |EIA 860        |raw              |                            |no                  |
securitization_policy|Status of Securitization Legislation for Coal Plant Retirements (State-level Utility Policy in 2021)|RMI|raw| |yes  |
governor_party  |Govenor Party in 2021        |RMI            |raw              |                            |yes                 |
legislation_majority_party|Legislation Majority Party in 2021|RMI|raw           |                            |yes                 |
CO2limitsOppose |Estimated percentage who somewhat/strongly oppose setting strict limits on existing coal-fire power plants|YCOM|raw| |yes|
2018to2021change_CO2opposition|Change in CO2limitsOppose from 2018 to 2021|YCOM|semi-calculated|2018 values subtracted from 2021 values. Positive values represent an increase in opposition to setting strict limits on existing coal-fire power plants|yes|
Mortality (high estimate)||CATF
Mortality (high estimate) DAC||CATF
Hospital Admits, All Respiratory||CATF
Hospital Admits, All Respiratory DAC||CATF
