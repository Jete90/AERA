# Adaptive Emission Reduction Approach (AERA)

This module implements the AERA algorithm developed
by Terhaar et al. (2022) 
(https://www.nature.com/articles/s41558-022-01537-9), 
i.e. an algorithm to iteratively
calculate fossil fuel CO2 emissions (every 5 years) which
are consistent with (respectively lead to) certain anthropogenic
temperature increases.

## Overview of the algorithm
See Terhaar et al. (2022) for more details.
1. Calculate the anthropogenic warming for the model using
   the simulated temperature. Based on that warming, determine the 
   allowed remaining temperature increase until the temperature
   target is reached. 
2. Calculate the past CO2-fe emissions from CO2 and non-CO2 radiative
   agents. Estimate the Transient climate response to cumulative carbon 
   emissions (TCRE) based on the anthrophogenic warming and the cumulative
   past CO2-fe emissions.
3. Calculate the remaining emission budget (REB) fo CO2-fe emissions 
   that can still be emitted until the temperature target is reached based
   on the TCRE and the allowable remaing temperature increase.
4. Distribute the REB on the future years, by calculating a smooth CO2-fe 
   emission pathway, whose integrated emissions are equal to the REB.
5. Return the annual fossil fuel emissions of the following years from the 
   difference of the estimated CO2-fe emission pathway and the prescribed
   future CO2-fe emissions from non-CO2 radiative agents.

## Requirements
To use the `aera` module a python virtual environment with python>=3.8 is needed.

## Installation
1. Download the AERA source code from GitHub
   (https://github.com/Jete90/AERA)
   ```
   git clone https://github.com/Jete90/AERA
   ```
2. Change directory into the repository and check that you are in the correct python virtual environment
   (if you don't know what that means, have a look [here](https://docs.python.org/3/tutorial/venv.html)
    or [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment))
   ```
   cd AERA
   which python
   ```
   Install the AERA module using pip.
   ```
   pip install .
   ```
   If successful the following output should appear:
   ```
   ...
   Successfully installed aera-1.0
   ```

## Executing the AERA algorithm
To calculate near future fossil fuel emissions using AERA, the `aera` module must be imported and the following functions must be called:
1. `aera.get_base_df`
2. `aera.get_adaptive_emissions`

### aera.get_base_df
This function provides a template pandas.DataFrame (used as the data container for all time series) which is provided as an input parameter to `aera.get_adaptive_emissions`. This function is called without any arguments:
```python
import aera
df = aera.get_base_df()
print(df)
```
Output:
```
Use the following non-CO2 emission file: /home/aschwanden/code/aera/aera/data/nonco2_emis_ssp126_v1.dat
Use the following land use emission file: /home/aschwanden/code/aera/aera/data/lu_emis_ssp126_bern3d_v1.dat
Use the following historical fossil fuel CO2 emission file: /home/aschwanden/code/aera/aera/data/co2_ff_GCP_plus_NDC_v1.dat


      non_co2_emission  lu_emission  ff_emission  temp 
year
1751              NaN           NaN        0.003   NaN     
1752              NaN           NaN        0.003   NaN    
1753              NaN           NaN        0.003   NaN      
1754              NaN           NaN        0.003   NaN       
1755              NaN           NaN        0.003   NaN      
...               ...           ...          ...   ...   
2296              0.0        -0.047          NaN   NaN     
2297              0.0        -0.044          NaN   NaN  
2298              0.0        -0.054          NaN   NaN   
2299              0.0        -0.053          NaN   NaN    
2300              0.0        -0.040          NaN   NaN  

[550 rows x 5 columns]
```

The first five lines are printed by `aera.get_base_df` to inform the user which (default) data is used for the respective columns in the returned dataframe `df`.
The template `df` contains data from the year 1751 until 2300. As seen in the output above, not all columns contain default data:
- `non_co2_emission` : Annual global non-CO2 emissions as CO2-eq (non-CO2 emissions) [Pg C/year]. `df` contains default data (based on `rf_non_co2`) from the year 1850 until 2300.
- `lu_emission` : Annual global land use CO2 emissions as CO2-eq (LU emissions) [Pg C/year]. `df` contains default data (from Bern3D using SSP1-2.6 land use area until 2100 and no land use area change afterwards) from the year 1850 until 2300.
- `ff_emission` : Annual global fossil fuel CO2 emissions (FF emissions) [Pg C/year]. `df` contains default data from the year 1751 until 2025 (1751-2020: historical data from [Global Carbon Budget 2020](https://www.icos-cp.eu/science-and-impact/global-carbon-budget/2020) (Friedlingstein et al., 2020); 2021-2025: assumed to evolve proportionally to the estimated CO2-fe emissions estimated from the Nationally Determined Contributions (NDC) from [Climate Action Tracker](https://climateactiontracker.org/documents/853/CAT_2021-05-04_Briefing_Global-Update_Climate-Summit-Momentum.pdf)).
- `temp` : Annual global mean surface air temperature (temperature) [K]. `df` does not contain any default data.

The template `df` must now be filled with model output:
- `ff_emission`: FF emission time series for the years 2025 until year x.
- `temp`: Temperature time series from model start year (first year of the historical simulation) until year x.

If model-specific data for the historical and a SSP1-2.6 simulation for non-CO2 RF, non-CO2 emissions, or LU emissions (columns `non_co2_emission`, `lu_emission`) is available, the above provided default data in `df` should be overwritten.

The user has to load the 2-D (or 3-D) data from the model output (e.g. using xarray, netCDF4, ...) and calculate a time-series of annual global mean/integral values for each variable. These time series have to be assigned to the correct years in `df`. This could look as follows:
```
# Load model output
temp_historical = xarray.open_mfdataset('/path/to/historical/temperature/data/temp_*.nc')
temp_pre_aera = xarray.open_mfdataset('/path/to/pre_aera/temperature/data/temp_*.nc')

# Calculate annual global mean time series
temp_historical_mean = global_mean(temp_historical)
temp_pre_aera_mean = global_mean(temp_pre_aera)

# Assign time series to `df`
df['temp'].loc[1851:2014] = temp_historical_mean  # temp must contain exactly 164 elements
df['temp'].loc[2014:2025] = temp_pre_aera_mean  # temp must contain exactly 12 elements
```

Once all required time series are assigned to `df`, we are ready to call `aera.get_adaptive_emissions`.

### aera.get_adaptive_emissions
Let's have a look at the function documentation of `aera.get_adaptive_emissions`:

```
Calculate "optimal" near-future CO2 emissions.

A full time series with CO2 emissions is returned, but only the next
five years are used in an AERA simulation. However, some
models calculate monthly emission data for the second half of the year
using already the annual emissions from the following year. Such models
therefore need at least one year more than these five years.

Args:
    temp_target_rel (float): Temperature target (e.g. 1.5K).
    temp_target_type (int): Switch for different types of temperature
        targets.
        - 1: Temperature target estimated by additing remaining
             warming until the target is reached based on
             observations in 2020 to simulated anthropogenic
             warming in 2020. Thus, all models have the same
             remaining warming after 2020 independend of their
             warming over the historical period.
        - 2: Temperature target estimated based on simulated
             warming anomaly with the reference period 1850-1900
    year_x (int): Current year in which the emissions for the next
        five years should be calculated.
    model_start_year (int): Year in which the historical
        simulation (pre-cursor for the adaptive scenario simulation)
        was started.
    df (pd.DataFrame): Pandas dataframe with years (int) as index
        and the following columns (see utils.get_base_df which
        provides a skeleton of this dataframe):
        - temp:  Global annual mean temperature time series for
          the period (in Kelvin).
        - ff_emission: Global annual mean fossil fuel CO2
          emission time series (in Pg C / yr).
        - lu_emission: Global annual mean land use change
          CO2 emission time series (in Pg C / yr).
        - non_co2_emission: Global annual mean non-CO2 emission (in
          CO2-eq Pg C / yr)
    meta_file (str or pathlib.Path): File for temporary data which
        should be transfered from one run of the AERA algorithm
        to the next.
    (...)

Returns:
    s_ff_emission (pd.Series): Annual globally integrated fossil fuel
        CO2 emission time series (in Pg C / yr).
```

The function takes at least six arguments. `temp_target_rel` is the temperature target which should be reached relative to pre-industrial conditions (e.g. 1.5K or 2.0K). `temp_target_type` is a numeric switch which
specifies how the temperature target is interpreted. This can always be set to 1. `year_x` is the stocktake year. `model_start_year` is the year in which the historical simulation starts. `df` is the pandas.Dataframe we received from `aera.get_base_df` and then filled with model output data. Finally, meta_file is a path to a writable file which is used to write out meta data from the AERA execution.

The function returns a pandas.Series object (similar to a numpy array) containing near-future annual globally integrated FF emissions. These values are then used to generate a new emission file for the model which is used for the next five years-cycle.

### Get started
To get started, please first have a look at the provided [examples](https://github.com/Jete90/AERA/tree/main/examples). There are plenty of comments in theses scripts to help you understand how the `aera` module should be used.

Once you understand the examples, you can couple the AERA to your model using a [template script](https://github.com/Jete90/AERA/tree/main/templates/template1.py), which should help you to use the AERA with your specific model. The template script will guide you by several 'TODO' comments. In the same directory there is also a "filled out" [template](https://github.com/Jete90/AERA/tree/main/templates/template1_filled_gfdl.py) which was used to call AERA when used with the GFDL-ESM2M model at the Swiss National Supercomputing Centre. Thus, if you are not sure how you should adjust the template to your model, you can have a look at how it was done for GFDL-ESM2M.

## Contributions:
- Jens Terhaar (jens.terhaar@unibe.ch)
- Mathias Aschwanden (mathias.aschwanden@unibe.ch)
- Thomas Frölicher (thomas.froelicher@unibe.ch)
- Fortunat Joos (fortunat.joos@unibe.ch)
- Pierre Friedlingstein (p.friedlingstein@exeter.ac.uk)

