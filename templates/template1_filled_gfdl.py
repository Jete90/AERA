"""Filled template1.py for use with GFDL ESM2M @CSCS (Switzerland).

An "AERA run script" will load all the neccessary time series from
your ESM simulation and call the AERA algorithm with this data.
Finally it will save the near-future fossil fuel CO2 emissions in
a format that is compatible with your ESM (e.g. as a NetCDF file).

Usage:
    $ python SCRIPT_NAME SIMULATION_DIRECTORY STOCKTAKE_YEAR \
        REL_TEMP_TARGET TEMP_TARGET_TYPE

EXAMPLE RUN COMMANDS:
    1)
    $ python template1_filled_gfdl.py $SCRATCH/AERA_T15_1_ENS1 2025 1.5 1
    This command runs the AERA algorithm for the simulation
    "AERA_T15_1_ENS1" (stored in $SCRATCH/AERA_T15_1_ENS1) and
    the stocktake year 2025 (thus it calculates emissions for the
    period 2026-2030) with a relative temperature target of 1.5K.
    The temperature target is calculated relative to
    the observed anthropogenic warming in 2020 ("type 1 temperature
    target").

    2)
    $ python template1_filled_gfdl.py $SCRATCH/AERA_T25_2_ENS1 2060 2.5 2
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

SOLVED: In addition to the "TODO comments" there are some
"SOLVED comments" (lines that start with "# SOLVED") to clarify
where code was changed/inserted.

"""
import datetime
from pathlib import Path
import sys
import time

import cftime
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import xarray as xr

import aera


PROJECT_DIR = Path('/project/s996/maschwan')
# Directory where several aera specific file are stored.
AERA_DATA_DIR = PROJECT_DIR / 'aera_data'


###########
# CONSTANTS
###########
# TODO(1): Set the model start year and model preindustrial CO2
# concentrations to the correct values for the ESM in use.

# Model start year: Defines in which year the ESM historical run
# starts.
MODEL_START_YEAR = 1861

# Model CO2 preindustrial: Defines the preindustrial CO2 concentration
# in units of ppm in the ESM.
MODEL_CO2_PREINDUSTRIAL = 286.32

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
OUTPUT_EMISSION_FILE = SIMULATION_DIR / 'INPUT/co2_emission.nc'
EMISSION_CSV_FILE = AERA_DIR / f'ff_emission.csv'
# 1 Pg C-normalized 3D (time, lat, lon) monthly fossil fuel emission file
YEAR_EMISSION_FILE = AERA_DIR / 'emission_pattern_1Pg_1year.nc'


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

# SOLVED(2): In the following temperature, CO2 concentration,
#       fossil fuel emission (from 2026 onward), and non-CO2
#       emission/radiative forcing time series are loaded and
#       written into the dataframe "df".

# Temperature and CO2 concentration data files are extracted
# from model output files using CDO (automatically after a year has
# been simulated) and copied into the following directories:
temp_data_dir = AERA_DIR / 't_ref'
co2_data_dir = AERA_DIR / 'rrvco2'

# Temperature
ds_area = xr.open_dataset(AERA_DIR/'gfdl_esm2m_atmos_area.nc')
area = ds_area.area.values
tref_files = list(temp_data_dir.glob('t_ref_*.nc'))
ds = xr.open_mfdataset(tref_files)
da = ds['t_ref']
# t_ref_*.nc are daily files: Calculate the annual mean.
da = da.groupby('time.year').mean('time')
da = da.where(da.year <= YEAR_X, drop=True)
tref = da.values
# `tref` is still 3D: Calculate global mean.
tref_ts = np.nansum(tref*area, axis=(1,2))/np.nansum(area)
df['temp'].loc[MODEL_START_YEAR:YEAR_X] = tref_ts[:YEAR_X-MODEL_START_YEAR+1]

# CO2 concentration
co2_files = list(co2_data_dir.glob('rrvco2_*.nc'))
ds = xr.open_mfdataset(co2_files)
da = ds['rrvco2']
# rrvco2_*.nc are daily files: Calculate the annual mean.
da = da.groupby('time.year').mean('time')
da = da.where(da.year <= YEAR_X, drop=True)
co2_ts = da.values.flatten()
df['co2_conc'].loc[MODEL_START_YEAR:YEAR_X] = co2_ts[:YEAR_X-1861+1]

# Fossil fuel emission
try:
    # Fossil fuel data is loaded directly from the CSV file
    # which is written out by this script (thus only data is
    # loaded which was previously calculated by this script).
    _df_tmp = pd.read_csv(EMISSION_CSV_FILE, index_col=0).dropna()
    values = _df_tmp.loc[2026:YEAR_X].values.flatten()
    df['ff_emission'].loc[2026:YEAR_X] = values
except FileNotFoundError:
    print(
        EMISSION_CSV_FILE, 'doesn\'t exist. '
        'Only emission until 2025 are available.')

# We also use model-specific non-CO2 emission and radiative forcing
# time series:
# Non-CO2 radiative forcing
nonco2_rf_gfdl_file = AERA_DATA_DIR / 'nonco2_rf_rcp26_gfdl.dat'
_df_tmp = pd.read_csv(nonco2_rf_gfdl_file, sep='\t', header=None)
_df_tmp.columns = ['year', 'data']
df['rf_non_co2'].loc[_df_tmp.year.min():_df_tmp.year.max()] = (
    _df_tmp.data.tolist())

# Non-CO2 emission
nonco2_emis_gfdl_file = AERA_DATA_DIR / 'nonco2_emis_rcp26_gfdl.dat'
_df_tmp = pd.read_csv(nonco2_emis_gfdl_file, sep='\t', header=None)
_df_tmp.columns = ['year', 'data']
df['non_co2_emission'].loc[_df_tmp.year.min():_df_tmp.year.max()] = (
    _df_tmp.data.tolist())


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
s_emission = df['ff_emission']
s_emission.update(s_emission_future)
s_emission.to_csv(EMISSION_CSV_FILE)


####################################
# WRITE NEAR-FUTURE FF CO2 EMISSIONS
####################################
# TODO(3): Write the fossil fuel CO2 emissions of (at least) the
# next five years (i.e. the period YEAR_X+1 until YEAR_X+5) to an
# emission # file which can be read by the ESM in use. Then copy
# this file into the input directory of the current ESM simulation
# and continue running the model.

# SOLVED(3): Using the annual globally integrated fossil fuel
# emission data (`s_emission_future`) a new NetCDF file is created
# which is then read in by GFDL ESM2M in the next iteration.
# First define neccessary functions, then call the function
# `create_emission_file`.


def _get_ncvar_time_value(dt, time_ref_year=0):
    if np.abs(time_ref_year) > 2:
        raise ValueError
    jan_2006 = 732692 - (time_ref_year * 365)
    dt_jan_2006 = datetime.datetime(2006, 1, 15)
    ddays = (dt - dt_jan_2006).days
    value = jan_2006 + ddays
    return value


def _get_rescaled_dim_rootgrp(rootgrp_ref, rootgrp_new, dimname_dimsize_dict):
    """Based on reference (netcdf4) rootgrp rescale dimensions."""
    dim_size_dict = {
        name: dim.size for name, dim in rootgrp_ref.dimensions.items()}
    dim_size_dict.update(dimname_dimsize_dict)
    for name, dim in rootgrp_ref.dimensions.items():
        rootgrp_new.createDimension(name, dim_size_dict[name])
    return rootgrp_new


def _slice_rootgrp(
        rootgrp_ref, rootgrp_new, varname_indecies_dict):
    """Slice all variables of reference (netcdf4) rootgrp."""
    for var_name, var in rootgrp_ref.variables.items():
        var_data = np.array(var[:])
        for dim in var.dimensions:
            if dim in varname_indecies_dict.keys():
                var_data = var_data.take(
                    indices=varname_indecies_dict[dim],
                    axis=var.dimensions.index(dim))
        fill_value = var.getncattr('_FillValue')
        var_new = rootgrp_new.createVariable(
            var_name, var.datatype, var.dimensions,
            fill_value=fill_value)
        var_new[:] = var_data
        for ncattr in var.ncattrs():
            if 'FillValue' in ncattr:
                continue
            var_new.setncattr(ncattr, var.getncattr(ncattr))
    return rootgrp_new


def _scale_variable(rootgrp, var_name, slice_factor_list):
    """Scale (multiply by factor) values of certain variables.

    Args:
        rootgrp (netcdf4.Dataset): Root group of a NetCDF variable
            loaded by netcdf4.
        var_name (str): Name of variable which should be scaled.
        indecies (array-like): Array of indecies.
        factors (array-like): Factors to scale variable values.

    Returns:
        rootgrp (netcdf4.Dataset): Root group of a NetCDF variable
            loaded by netcdf4.

    """
    for slic, factor in slice_factor_list:
        rootgrp[var_name][slic] *= factor
    return rootgrp


def create_emission_file(
        start_year, end_year, year_emission_file, output_file,
        glb_em_prev, glb_em, glb_em_next=None, overwrite=False,
        ):
    """Create monthly fossil fuel emission file (time, lat, lon) for
    GFDL ESM2M from a timeseries of emissions and a normalized emission
    file of a single year (time[12], lat, lon).

    Args:
        start_year (int): Start year for the emission file.
        end_year (int): End year for the emission file.
        year_emission_file (str or pathlib.Path): NetCDF file containing
            a 2D fossil fuel emission pattern for each
            month of the year (thus the file is 3D).
            The data must be normalized to a total emission of 1 Pg
            emission per year.
        output_file (str or pathlib.Path): Path where the resulting
            3D fossil fuel file should be stored.
        glb_em_prev (float): Global emission in the year before
            `start_year`.
        glb_em (list[float]): List of global emission for the years
            between `start_year` and `end_year`.
        glb_em_next (float): Global emission in the year after
            `end_year`. Optional. If not provided glb_em_prev is
            set to glb_em[-1].
        overwrite (bool): Flag that specifies whether the output file
            should be overwritten if it already exists.

    """
    if output_file.exists():
        if overwrite:
            output_file.unlink()
        else:
            raise ValueError(
                'Warning: output file already exists. '
                'Either delete it manually or set overwrite=True')

    nyears = (end_year - start_year) + 1
    year_month_tuples = [
        (y, m) for y in range(start_year, end_year+1) for m in range(1, 13)]
    year_month_tuples_extended = (
        [(start_year-1, 12)] + year_month_tuples + [(end_year+1, 1)])
    month_day_dict = {i: (16 if i != 2 else 15) for i in range(1, 13)}
    month_hour_dict = {
        i: (12 if i in [1, 3, 5, 7, 8, 10, 12] else 0) for i in range(1, 13)}
    time_datetimes = [
        datetime.datetime(y, m, 15) for y, m in year_month_tuples]
    time1_datetimes = [
        cftime.DatetimeNoLeap(y, m, month_day_dict[m], month_hour_dict[m])
        for y, m in year_month_tuples_extended]
    time_values = [_get_ncvar_time_value(d) for d in  time_datetimes]
    time1_values = cftime.date2num(
        time1_datetimes, units='DAYS since 0001-01-01 00:00:00')
    # Diff between mid Jan to mid Feb, mid Feb to mid March, ...,
    # mid Dec to mid Jan
    diffs = np.diff(time1_values[1:14])
    diffs = diffs.take(indices=[10, 11] + list(range(0, 12))*nyears + [0])
    time1_bnds_values = [
            (time1_values[i]-(diffs[i]/2), time1_values[i]+(diffs[i+1]/2))
            for i in range(len(time1_values))]

    rootgrp_year = Dataset(year_emission_file, "r")
    rootgrp_out = Dataset(
        output_file, "w", format=rootgrp_year.data_model)
    rootgrp_out = _get_rescaled_dim_rootgrp(
        rootgrp_year, rootgrp_out, {'time': 12*nyears, 'time1': None})
    rootgrp_out = _slice_rootgrp(
        rootgrp_year,
        rootgrp_out,
        {
            'time': list(range(12))*nyears,
            'time1': [11] + list(range(12))*nyears + [0],
            },
        )
    rootgrp_out['time'][:] = time_values
    rootgrp_out['time1'][:] = time1_values
    rootgrp_out['time1_bnds'][:] = time1_bnds_values

    slices = (
        [np.s_[:1, :, :]] +
        [np.s_[1+12*i:1+12*(i+1), :, :] for i in range(0, nyears)] +
        [np.s_[1+12*nyears:, :, :]])

    if glb_em_next is None:
        glb_em_next = glb_em[-1]
    factors = [float(glb_em_prev)] + list(glb_em) + [float(glb_em_next)]
    print('Emission scalings for first, last, and all timesteps in between:')
    print('-> First timestep: ', factors[0])
    print('-> Every 12 timesteps inbetween: ', factors[1:-1])
    print('-> Last timestep: ', factors[-1])
    _scale_variable(rootgrp_out, 'co2_emissions_ff', list(zip(slices, factors)))

    rootgrp_year.close()
    rootgrp_out.close()


create_emission_file(
    start_year=YEAR_X+1,
    end_year=YEAR_X+5,
    year_emission_file=YEAR_EMISSION_FILE,
    output_file=OUTPUT_EMISSION_FILE,
    glb_em_prev=s_emission.loc[YEAR_X],
    glb_em=s_emission.loc[YEAR_X+1:YEAR_X+5].values,
    glb_em_next=s_emission.loc[YEAR_X+6],
    overwrite=True,
    )


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
