"""Utility functions.

Contains the following important functions:
- validate_df : Used to check if an end-user given pandas.DataFrame
    contains all the neccessary data and is correctly formatted.
- get_base_df : A helper function that facilitates the creation of
    the pandas.DataFrame instance which is needed by
    `aera.core.get_adaptive_emissions`.

"""
from pathlib import Path

import numpy as np
import pandas as pd

import aera


MIN_YEAR = 1751
MAX_YEAR = 2300


def validate_df(df, year_x, model_start_year):
    """Validate whether all neccessary data is contained in `df`."""
    model_start_year = max(1850, model_start_year)
    if model_start_year >= 1900:
        raise ValueError(
            'The historical run is too short for this algorithm.')

    col_years_dict = {
        'temp': np.arange(model_start_year, year_x+1),
        'co2_conc': np.arange(model_start_year, year_x+1),
        'rf_non_co2': np.arange(model_start_year, MAX_YEAR+1),
        'ff_emission': np.arange(model_start_year, year_x+1),
        'lu_emission': np.arange(model_start_year, MAX_YEAR+1),
        'non_co2_emission': np.arange(model_start_year, MAX_YEAR+1),
        'n2o_conc': np.arange(model_start_year, MAX_YEAR+1),
        }
    for col_name, years in col_years_dict.items():
        col = df[col_name].dropna()
        if not np.all(np.isin(years, col.index.values)):
            missing_years_idx = np.argwhere(~np.isin(years, col.index.values))
            missing_years = years[missing_years_idx].flatten()
            raise ValueError(
                f'Neccessary data in column {col_name} is missing.\n'
                'Data for the following years is missing but must be '
                f'available:\n {missing_years}.')


def _load_dat_df(f, column_names, delim_whitespace=False):
    df = pd.read_table(
        f, header=None, index_col=0, delim_whitespace=delim_whitespace)
    df.columns = column_names
    df.index.name = 'year'
    df.index = [int(x) for x in df.index.values]
    df = df.reindex(
        np.arange(df.index.min(), df.index.max())).interpolate()
    return df


def get_base_df():
    """Return dataframe which is used by get_adaptive_emissions.

    The dataframe contains non-CO2 radiative forcing,
    non-CO2 emission, and land use emission data provided
    by Terhaar et al. (in prep.). This data is contained in the
    `aera` module and can be found within the official repository.

    Note: The returned pandas.DataFrame cannot be passed to
    `aera.core.get_adaptive_emissions` directly!
    The following steps are still neccessary before calling
    `get_adaptive_emissions`:
    - Fill "temp" column with temperature data from the model.
    - Fill "co2_conc" column with CO2 concentration data from
      the model.
    - Fill "ff_emission" column with CO2 emission data from
      year 2026 on.
    - If model-specific data is available for "lu_emission",
      "non_co2_emission", and "rf_non_co2" columns, please 
      overwrite the prefilled 'standard' data

    Returns:
        df (pandas.DataFrame): Template dataframe which can be
            filled with data and then used for the call to
            `aera.get_adaptive_emissions`.

    """
    data_dir = Path(aera.__file__).parent / 'data'
    rf_non_co2_file = data_dir / 'nonco2_rf_ssp126_v1.dat'
    non_co2_emission_file = data_dir / 'nonco2_emis_ssp126_v2.dat'
    lu_emission_file = data_dir / 'lu_emis_ssp126_bern3d_adj_GCB2020_v1.dat'
    ff_emission_file = data_dir / 'co2_ff_GCP_plus_NDC_v1.dat'
    n2o_conc_file = data_dir / 'n2o_conc_ssp126_v1.dat'

    print(f'Use the following non-CO2 RF file: {rf_non_co2_file}')
    print(f'Use the following non-CO2 emission file: {non_co2_emission_file}')
    print(f'Use the following land use emission file: {lu_emission_file}')
    print(
        f'Use the following historical fossil fuel CO2 emission '
        f'file: {ff_emission_file}')
    print(
        f'Use the following historical N2O concentration '
        f'file: {n2o_conc_file}')

    df_list = []
    
    df_rf = _load_dat_df(
        rf_non_co2_file, ['rf_total', 'rf_co2', 'rf_non_co2'], 
        delim_whitespace=False)
    df_list.append(df_rf[['rf_non_co2']])

    df_non_co2_emission = _load_dat_df(
        non_co2_emission_file, ['non_co2_emission'], delim_whitespace=True)
    df_list.append(df_non_co2_emission)

    df_ff_emission = _load_dat_df(
        ff_emission_file, ['ff_emission'], delim_whitespace=True)
    df_list.append(df_ff_emission)

    df_lu_emission = _load_dat_df(
        lu_emission_file, ['lu_emission'], delim_whitespace=True)
    df_list.append(df_lu_emission)

    df_n2o_conc = _load_dat_df(
        n2o_conc_file, ['n2o_conc'], delim_whitespace=True)
    df_list.append(df_n2o_conc)
    
    df = pd.concat(df_list, axis=1)
    df['temp'] = np.nan
    df['co2_conc'] = np.nan
    df['ff_emission'].loc[2026:] = np.nan
    df['lu_emission'].loc[:1849] = np.nan
    df['rf_non_co2'].loc[:1849] = np.nan
    df['non_co2_emission'].loc[:1849] = np.nan
    df['n2o_conc'].loc[:1849] = np.nan
    df.index.name = 'year'
    # Reorder columns
    df = df[['rf_non_co2', 'non_co2_emission', 'lu_emission',
             'ff_emission', 'temp', 'co2_conc','n2o_conc']]

    return df.loc[MIN_YEAR:2499]
