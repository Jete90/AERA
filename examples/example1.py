"""Basic AERA run script.

-> For a template of an AERA run script (which can be adapted
and then coupled to an ESM simulation to continually calculate 
fossil fuel CO2 emission every five years)
see ../templates/template1.py.

This script is an example that uses model output from the Bern3D 
model and calculates future CO2 emissions using the AERA algorithm
in year 2025.

USAGE:
    $ python example1.py

"""
from pathlib import Path
import time

import netCDF4
import pandas as pd
import numpy as np

import aera

# Directory where the future fossil fuel CO2 emission data (as *.csv)
# and further meta information is stored.
OUTPUT_DIRECTORY = Path(__file__).parent.absolute() / 'example1_output'
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
TIMESTAMP = int(time.time())

###########
# CONSTANTS
###########
# Temperature target type: Defines how the absolute temperature
# target is calculated. If it is set to "1", the absolute
# temperature target is calculated relative to the observed
# anthropogenic warming in 2020. If it is set to "2", the absolute
# temperature target is calculated relative to the mean simulated
# temperature in the model between 1850 and 1900.
TEMP_TARGET_TYPE = 1
# Relative tempetature target: Defines how much the ESM is "allowed"
# to warm either relative to the observed warming in 2020
# (TEMP_TARGET_TYPE = 1) or relative to the mean ESM temperature in
# the period 1850-1900 (TEMP_TARGET_TYPE = 2).
TEMP_TARGET_REL = 1.5
# Year X: Defines the stocktake year we are currently at.
YEAR_X = 2025
# Model start year: Defines in which year the ESM historical run
# starts.
MODEL_START_YEAR = 1850
# Model CO2 preindustrial: Defines the preindustrial CO2 concentration
# in units of ppm in the ESM.
MODEL_CO2_PREINDUSTRIAL = 278.05

###################
# DEFINE DATA FILES
###################
example_data_dir = Path(__file__).parent / 'example1_data'
model_output_file = example_data_dir / 'Bern3Doutput.nc'
ff_emission_file = example_data_dir / 'co2_ff_GCP_plus_NDC_v1.dat'

meta_file = OUTPUT_DIRECTORY / f'meta_example1_{TIMESTAMP}.nc'

#####################
# CALL AERA ALGORITHM
#####################
# First get a base pandas Dataframe (df) which is later used to call
# the main function of the AERA algorithm.
df = aera.get_base_df()

# Manually overwrite the temperature, CO2 and fossil fuel
# emission values in the df:
# 1) Load the datasets
# 2) Set the values in the respective df columns.
ds = netCDF4.Dataset(model_output_file, 'r')
temperature = ds.variables['ATMT_ALTI'][:].filled(np.nan)
co2_conc = ds.variables['atmpco2'][:].filled(np.nan)
idx = YEAR_X-2025
if idx == 0:
    idx = None
df['temp'].loc[1765:YEAR_X] = temperature[:idx]
df['co2_conc'].loc[1765:YEAR_X] = co2_conc[:idx]
df_ff_emission = pd.read_table(
    ff_emission_file, header=None, index_col=0, delim_whitespace=True)
df_ff_emission.columns = ['ff_emission']
df_ff_emission.index.name = 'year'
df_ff_emission.index = [int(x) for x in df_ff_emission.index.values]
df_ff_emission = df_ff_emission.reindex(
    np.arange(
        df_ff_emission.index.min(),
        df_ff_emission.index.max())
    ).interpolate()
df['ff_emission'].loc[1765:YEAR_X] = (
    df_ff_emission.loc[1765:YEAR_X].values.flatten())

# Now call the main AERA function `get_adaptive_emissions`.
# For further information what parameters must be provided see
# the docstring of the function `get_adaptive_emissions`.
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
output_file = OUTPUT_DIRECTORY / f'future_emissions_example1_{TIMESTAMP}.csv'
s_emission.to_csv(output_file)

print(f'[DEBUG] YEAR_X: {YEAR_X}')
print(f'[DEBUG] TEMP_TARGET_REL: {TEMP_TARGET_REL}')
print(f'[DEBUG] TEMP_TARGET_TYPE: {TEMP_TARGET_TYPE}')
print('[DEBUG] df', df.loc[1980:2025])
print('[DEBUG] Calculated the following future emissions: ')
print('[DEBUG]', s_emission)
print(f'[DEBUG] Wrote emissions to {output_file}')
print(f'[DEBUG] Wrote meta data to {meta_file}')
