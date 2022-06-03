### An implementation of the Kepler Mapper to analyze various US Energy Information Agency (EIA) energy datasets.
- Supports EIA eGrid analysis of all US powerplants
- Sortable by timeframe, plant fuel type, and many other variables to tailor the nature of analysis
- Data exploration and viewing supported by the Kepler Mapper

[coal_mapper GitHub](https://github.com/sgathrid/coal_mapper)

#### A Kepler Mapper simplicial complex made using 5 parameters:
- affectedDAC: the number of people within a 3 mile radius of each coal plant that are considered to be members of Disadvantaged Communities
- PM 2.5 Emssions (tons): Amount of particulate matter 2.5 released annually
- post2004RetrofitCosts: dollars spent on pollution control retrofits after 2004
- CO2limitsOppose: the percent of people who oppose setting CO2 limits on coal powerplant emissions
- StateMineEmployment: the number of people employed by coal mining operations in the state in which the powerplant is located

[Link to test map](https://github.com/sgathrid/coal_mapper/blob/main/docs/DAC_mapper.html)

| Key Command    | Action             |
| -------------- | ------------------ |
| p              | Print View (white) |
| d              | Dark View (black)  |
| e              | Tight Layout       |
| m              | Expanded Layout    |
| f              | Freeze Layout      |
| x              | Unfreeze all Nodes |

{% include_relative DAC_mapper.html %}
