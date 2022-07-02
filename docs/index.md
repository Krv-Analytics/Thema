### An implementation of the Kepler Mapper to analyze various US Energy Information Agency (EIA) energy datasets.
- Supports EIA eGrid analysis of all US powerplants
- Sortable by timeframe, plant fuel type, and many other variables to tailor the nature of analysis
- Data exploration and viewing supported by the Kepler Mapper

[coal_mapper GitHub](https://github.com/sgathrid/coal_mapper)

### Below is an example Kepler Mapper simplicial complex made using 5 parameters:
- **affectedDAC:** number of people within a 3 mile radius of each coal plant that are considered to be members of Disadvantaged Communities<br/>
- **PM 2.5 Emssions (tons):** tons of PM 2.5 released annually<br/>
- **post2004RetrofitCosts:** dollars spent on pollution control retrofits after 2004<br/>
- **CO2limitsOppose:** percent of people _in the plant's county_ who oppose setting CO2 limits on coal powerplant emissions<br/>
- **StateMineEmployment:** number of people employed by coal mining operations _in the state_ in which the powerplant is located <br/>

The goal of using these parameters is to view the relationships between Energy Justice (EJ) indicators, public opinion data, and coal plant pollution metrics.

<iframe width="100%" height="700px" seamless="" frameborder="0" scrolling="no" src="./DAC_mapper.html"></iframe>

###### The linked cluster of plants within a three mile radius of the highest number of DAC members also happens to be a cluster of plants that generate low levels of power and criteria pollutants. However, these plants are located in areas that support limiting coal powerplant CO2 emissions. This can be seen by toggling the color scheme between the different indicator variables. By clicking the _[+] Help_ menu, a list of different key commands and viewing options will appear. <br/>

### Usage and Explanation

While the structure of the simplicial complex was generated from the above 5 parameters, changing the color shceme allows the user to view how different parameters relate to the structure of the graph. 

### Using the Kepler Mapper results to partition US Coal Plants into Scenarios:
– scenarios represent the sequential increase of decommisisoning difficulty, with scenario 1 plants being the easiest to decommission and 4 the hardest


<iframe width="100%" height="700px" seamless="" frameborder="0" scrolling="no" src="./COAL_MAP.html"></iframe>
