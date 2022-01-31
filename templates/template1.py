"""Basic template for an AERA run script.

An "AERA run script" will load all the neccessary time series from
your ESM simulation and call the AERA algorithm with this data.
Finally it will save the near-future fossil fuel CO2 emissions in
a format that is compatible with your ESM (e.g. as a NetCDF file).

Usage:
    $ python SCRIPT_NAME SIMULATION_DIRECTORY STOCKTAKE_YEAR \
        REL_TEMP_TARGET TEMP_TARGET_TYPE

EXAMPLE RUN COMMANDS:
    1)
    $ python template1.py $SCRATCH/AERA_T15_1_ENS1 2025 1.5 1
    This command runs the AERA algorithm for the simulation
    "AERA_T15_1_ENS1" (stored in $SCRATCH/AERA_T15_1_ENS1) and
    the stocktake year 2025 (thus it calculates emissions for the
    period 2026-2030) with a relative temperature target of 1.5K.
    The temperature target is calculated relative to
    the observed anthropogenic warming in 2020 ("type 1 temperature
    target").

    2)
    $ python template1.py $SCRATCH/AERA_T25_2_ENS1 2060 2.5 2
    This command runs the AERA algorithm for the simulation
    "AERA_T25_2_ENS1" (stored in $SCRATCH/AERA_T25_2_ENS1) and
    the stocktake year 2060 (thus it calculates emissions for the
    period 2061-2065) with a relative temperature target of 2.5K.
    The temperature target is calculated relative to
    the mean temperature in the ESM in the period 1850-1900
    ("type 2 temperature target").

TODO: If you want to adapt the following script, first jump to all
the TODOs (lines that start with "# TODO") in the script and
make according changes.

"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import aera


###########
# CONSTANTS
###########
# TODO(1): Set the model start year and model preindustrial CO2
# concentrations to the correct values for the ESM in use.

# Model start year: Defines in which year the ESM historical run
# starts.
MODEL_START_YEAR = 1850

# Model CO2 preindustrial: Defines the preindustrial CO2 concentration
# in units of ppm in the ESM.
MODEL_CO2_PREINDUSTRIAL = 280

# Enable/Disable output of debug information
DEBUG = True


######################
# READ INPUT ARGUMENTS
######################
SIMULATION_DIR = Path(sys.argv[1])
YEAR_X = int(sys.argv[2])
REL_TEMP_TARGET = float(sys.argv[3])
TEMP_TARGET_TYPE = int(sys.argv[4])

# Verify that the given stocktake year (YEAR_X) is valid
if YEAR_X % 5 != 0:
    raise ValueError(
        f'YEAR_X is not valid ({YEAR_X})! Abort adaptive emission calculation.'
        'YEAR_X must be divisible by 5 (e.g. 2025, 2030, 2035).')

AERA_DIR = SIMULATION_DIR / 'AERA'


###################################
# COLLECT DATA AND CREATE DATAFRAME
###################################
# The AERA algorithm needs several time series as its input.
# To make the usage of the algorithm easier and to assure
# that the input data is correctly formated the main function
# `aera.get_adaptive_emissions` takes a pandas.DataFrame as its
# input.
# The `aera` module also provides a function which returns
# a "template dataframe". The data in this dataframe can now be
# overwritten with the data from the running simulation.

# Get the template dataframe
df = aera.get_base_df()

# This dataframe contains the following columns
# - rf_non_co2
# - non_co2_emission
# - ff_emission
# - lu_emission
# - temp
# - co2_conc
# The index of the dataframe is "year".
#
# TODO(2): The columns ff_emission, temp, and co2_conc MUST be provided
#       (i.e. the data in the dataframe in these columns must be set)
# TODO(2) (optional): The 'standard' data in the columns rf_non_co2,
#       non_co2_emission, and lu_emission should also be provided if
#       model-specific time series are available.
#
# Note: For all these time series except ff_emission, data from
# `MODEL_START_YEAR` until `YEAR_X` must be provided (e.g. 1850-2025).
# The fossil fuel emissions must be provided from 2026 onward.
# Before 2026 all models should have identical (or very similar)
# emissions (only after 2026 they begin to use the AERA algorithm),
# therefore the fossil fuel emission data from 1700 until 2025 is
# contained in the AERA python module.
#
# To provide the above mentioned data:
#   1) Load the data (e.g. using xarray.open_dataset)
#   2) Calculate global annual mean values
#   3) Overwrite values in `df`
#
# This could look something like:
# df['temp'].loc[MODEL_START_YEAR:YEAR_X] =
# df['co2_conc'].loc[MODEL_START_YEAR:YEAR_X] =
# df['ff_emission'].loc[2026:YEAR_X] =
# (optional): df['rf_non_co2'].loc[MODEL_START_YEAR:YEAR_X] =
# (optional): df['non_co2_emission'].loc[MODEL_START_YEAR:YEAR_X] =
# (optional): df['lu_emission'].loc[MODEL_START_YEAR:YEAR_X] =


#####################
# CALL AERA ALGORITHM
#####################
# Using the above created dataframe `df` we can now call the
# AERA algorithm:
s_emission_future = aera.get_adaptive_emissions(
    temp_target_rel=REL_TEMP_TARGET,
    temp_target_type=TEMP_TARGET_TYPE,
    year_x=YEAR_X,
    co2_preindustrial=MODEL_CO2_PREINDUSTRIAL,
    model_start_year=MODEL_START_YEAR,
    df=df,
    meta_file=AERA_DIR/f'meta_data_{YEAR_X}.nc',
    )

# The future ff emissions are saved to a CSV file. But the ff emission
# also must be saved in a format which can be read by the ESM in use.
s_emission_future.to_csv(emission_file)


####################################
# WRITE NEAR-FUTURE FF CO2 EMISSIONS
####################################
# TODO(3): Write the fossil fuel CO2 emissions of (at least) the
# next five years (i.e. the period YEAR_X+1 until YEAR_X+5) to an
# emission # file which can be read by the ESM in use. Then copy
# this file into the input directory of the current ESM simulation
# and continue running the model.


#############################
# WRITE DEBUG INFORMATION
#############################
# Print out some debug information and also write these to a file
if DEBUG:
    debug_str = (
        'INPUT ARGUMENTS: \n'
        f'[DEBUG] Year X: {YEAR_X}\n'
        f'[DEBUG] Relative Target Temperature: {REL_TEMP_TARGET}\n'
        f'[DEBUG] Target Temperature Type: {TEMP_TARGET_TYPE}\n'
        f'[DEBUG] Preindustrial CO2: {MODEL_CO2_PREINDUSTRIAL}\n'
        f'[DEBUG] Model start year: {MODEL_START_YEAR}\n\n\n\n'
        'OUTPUT: \n'
        f'[DEBUG] Calculated the following emissions: {s_emission_future}'
        )

    try:
        debug_filename = f'{YEAR_X}_{int(time.time())}.debug'
        # The debug file will be created where the "AERA run script" lies
        debug_file = AERA_DIR / debug_filename
        with open(debug_file, 'w') as f:
            f.write(debug_str)
    except PermissionError:
        print('[WARNING] Failed to write the debug information '
              f'to {debug_file} (permission denied).')
