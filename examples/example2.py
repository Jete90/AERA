"""Apply AERA to observational data.

This script uses observational data (as described in Terhaar et al.,
in review) to calculate near future fossil fuel CO2 emissions with the
AERA algorithm.

USAGE:
    $ python example2.py

"""
from pathlib import Path
import time

import netCDF4
import pandas as pd
import numpy as np

import aera


# Directory where the future fossil fuel CO2 emission data (as *.csv)
# and further meta information is stored.
OUTPUT_DIRECTORY = Path(__file__).parent.absolute() / 'example2_output'
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
TIMESTAMP = int(time.time())

###########
# CONSTANTS
###########
# Temperature target type: Defines how the absolute temperature
# target is calculated. If it is set to "1", the absolute
# temperature target is calculated relative to the observed
# anthropogenic warming in 2020. If it is set to "2", the absolute
# temperature target is calculated relative to the mean
# temperature in the ESM between 1850 and 1900.
TEMP_TARGET_TYPE = 1
# Relative tempetature target: Defines how much the ESM is "allowed"
# to warm either relative to the observed warming in 2020
# (TEMP_TARGET_TYPE = 1) or relative to the mean ESM temperature in
# the period 1850-1900 (TEMP_TARGET_TYPE = 2).
TEMP_TARGET_REL = 1.5
# Year X: Defines the stocktake year we are currently at.
YEAR_X = 2020
# Model start year: Defines in which year the ESM historical run
# starts. In this observational-based example it's the just
# the first year for which data is available.
MODEL_START_YEAR = 1765
# Model CO2 preindustrial: Defines the preindustrial CO2 concentration
# in units of ppm in the ESM.
MODEL_CO2_PREINDUSTRIAL = 278.05

# NonCO2 radiative forcing and emission data from RCP/SSP databases 
# (derived as total-CO2 RF) or from IPCC data (IPCC, 2013: Annex II: 
# Climate System Scenario Tables)
# NON_CO2_DATA_SOURCE = 'RCPSSP'
NON_CO2_DATA_SOURCE = 'IPCC'

###################
# DEFINE DATA FILES
###################
example_data_dir = Path(__file__).parent / 'example2_data'

# Temperature anomaly from HadCRUT.5.0.1.0
temp_file = (
    example_data_dir /
    'HadCRUT.5.0.1.0.analysis.summary_series.global.annual.nc')
# Historical CO2 concentrations from Meinshausen et al. (2017) from
# 1850-2014 and from NOAA GML from 2015 to 2020
co2_file = (
    example_data_dir / 'co2_conc_historical_Meinshausen2017_NOAA_GML.dat')
# Fossil fuel CO2 emissions from GCP
ff_emission_file = example_data_dir / 'co2_ff_GCP_plus_NDC_v1.dat'

# Non-CO2 emissions as CO2-fe calculated from non-CO2 radiative
# forcing as described in Terhaar et al. (in review)
if NON_CO2_DATA_SOURCE == 'RCPSSP':
    non_co2_rf_file = example_data_dir / 'nonco2_rf_ssp126_v1.dat'
    non_co2_emission_file = example_data_dir / 'nonco2_emis_ssp126_v2.dat'
elif NON_CO2_DATA_SOURCE == 'IPCC':
    non_co2_rf_file = example_data_dir / 'nonco2_rf_ipcc_v1.dat'
    non_co2_emission_file = example_data_dir / 'nonco2_emis_IPCC_v1.dat'
else:
    raise ValueError

# Land use change emissions from GCP 2020 (Friedlingstein et al. 2020)
lu_emission_file = example_data_dir / 'lu_GCP.dat'

meta_file = OUTPUT_DIRECTORY / f'meta_example2_{TIMESTAMP}.nc'

#####################
# CALL AERA ALGORITHM
#####################


def _log_overwrite(f, col_name):
    print(f'Overwrite column "{col_name}" with data from {f}')


def _load_dat_df(f, column_names, **read_table_kwargs):
    """Function for reading *.dat files."""
    read_table_kwargs1 = dict(
            header=None, index_col=0, delim_whitespace=True)
    read_table_kwargs1.update(read_table_kwargs)
    df = pd.read_table(f, **read_table_kwargs1)
    df.columns = column_names
    df.index.name = 'year'
    df.index = [int(x) for x in df.index.values]
    df = df.reindex(
        np.arange(df.index.min(), df.index.max()+1)).interpolate()
    return df


# First get a base pandas Dataframe (df) which is later used to call
# the main function of the AERA algorithm.
df = aera.get_base_df()

# Manually overwrite the temperature, CO2 and fossil fuel
# emission, non-CO2 rf/emission, and land use emission values
# in the df:
# 1) Load the datasets
# 2) Set the values in the respective df columns.

# Temperature time series
ds = netCDF4.Dataset(temp_file, 'r')
temperature = ds.variables['tas_mean'][:].filled(np.nan)
temperature = temperature-np.nanmean(temperature)
# add 273.15+15 to go from temperature anomalies to
# absolute temperature in Kelvin as demanded by the script
# Note: The +15 is only a rough estimate of the global mean
# temperature (it doesnt matter for the algorithm whether 
# this value is 100% correct)
temperature = temperature+288.15
_log_overwrite(temp_file, 'temp')
df['temp'].loc[1850:YEAR_X] = temperature

# CO2 concentration time series
_log_overwrite(co2_file, 'co2_conc')
df.update(_load_dat_df(co2_file, ['co2_conc']))

# CO2 emissions time series
_log_overwrite(ff_emission_file, 'ff_emission')
df.update(_load_dat_df(ff_emission_file, ['ff_emission']))

# Non-CO2 emissions time series
_log_overwrite(non_co2_emission_file, 'non_co2_emission')
df.update(
    _load_dat_df(
        non_co2_emission_file, ['non_co2_emission'],
        delim_whitespace=True,
        )
    )

# Non-CO2 radiative forcing time series
_log_overwrite(non_co2_rf_file, 'rf_non_co2')
df.update(
    _load_dat_df(
        non_co2_rf_file, ['rf_total', 'rf_co2', 'rf_non_co2'],
        delim_whitespace=False,
        )['rf_non_co2']
    )

# Landuse emissions time series
_log_overwrite(lu_emission_file, 'lu_emission')
df.update(
    _load_dat_df(
        lu_emission_file, ['lu_emission'],
        delim_whitespace=True,
        )
    )

s_emission = aera.get_adaptive_emissions(
    temp_target_rel=TEMP_TARGET_REL,
    temp_target_type=TEMP_TARGET_TYPE,
    year_x=YEAR_X,
    co2_preindustrial=MODEL_CO2_PREINDUSTRIAL,
    model_start_year=MODEL_START_YEAR,
    df=df,
    meta_file=meta_file,
    )

####################################
# WRITE OUTPUT AND DEBUG INFORMATION
####################################
# Write the future fossil fuel CO2 emission data (as *.csv)
output_file = OUTPUT_DIRECTORY / f'future_emissions_example2_{TIMESTAMP}.csv'
s_emission.to_csv(output_file)

print(f'[DEBUG] YEAR_X: {YEAR_X}')
print(f'[DEBUG] TEMP_TARGET_REL: {TEMP_TARGET_REL}')
print(f'[DEBUG] TEMP_TARGET_TYPE: {TEMP_TARGET_TYPE}')
print('[DEBUG] df', df.loc[1980:2025])
print('[DEBUG] Calculated the following future CO2 emissions: ')
print('[DEBUG]', s_emission)
print(f'[DEBUG] Wrote emissions to {output_file}')
print(f'[DEBUG] Wrote meta data to {meta_file}')
