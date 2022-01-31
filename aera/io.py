"""Definition of meta data storage functions."""

import copy
import dataclasses
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import xarray as xr


def store_metadata(
        meta_file, temp_target_rel, temp_target_abs, year_x, model_start_year,
        s_temp_anth, s_total_emission, s_ff_emission, emission_curve):
    meta_file = Path(meta_file)
    timeseries_csv = Path(str(meta_file) + '.timeseries.csv')
    scalar_csv = Path(str(meta_file) + '.scalar.csv')

    timeseries_columns = [
        'total_emission', 'ff_emission', 'temp_anth', 'temp_anth_rel']
    scalar_columns = [
        'temp_target_abs', 'total_emission_budget', 'ff_emission_budget',
        'reb', 'ec_cost', 'ec_reb_diff', 'ec_overshoot_integral', 'ec_slope_t1',
        'ec_slope_change', 'ec_overshoot','ec_curvature', 'ec_target_year_rel',
        'ec_a', 'ec_b', 'ec_c', 'ec_d',
        ]

    try:
        df_timeseries = pd.read_csv(timeseries_csv, index_col=[0, 1], header=0)
    except FileNotFoundError:
        df_timeseries = pd.DataFrame(
            columns=timeseries_columns+['year_stocktake', 'year'])
        df_timeseries = df_timeseries.set_index(['year_stocktake', 'year'])

    try:
        df_scalar = pd.read_csv(scalar_csv, index_col=0, header=0)
    except FileNotFoundError:
        df_scalar = pd.DataFrame(
            columns=scalar_columns+['year_stocktake'])
        df_scalar = df_scalar.set_index(['year_stocktake'])

    df_scalar.loc[year_x, 'temp_target_abs'] = temp_target_abs
    target_year_abs = emission_curve.target_year_rel + year_x
    df_scalar.loc[year_x, 'total_emission_budget'] = (
        s_total_emission.loc[:target_year_abs].sum())
    df_scalar.loc[year_x, 'ff_emission_budget'] = (
        s_ff_emission.loc[:target_year_abs].sum())
    df_scalar.loc[year_x, 'reb'] = emission_curve.reb
    for col in [x for x in scalar_columns if x.startswith('ec_')]:
        attribute = col[3:]
        df_scalar.loc[year_x, col] = getattr(emission_curve, attribute)

    # round all values except ec_a, ec_b, ec_c and ec_d
    for col in df_scalar.columns:
        if col in ['ec_a', 'ec_b', 'ec_c', 'ec_d']:
            continue
        df_scalar[col] = df_scalar[col].map(lambda x: '%.3f' % x)
    df_scalar.to_csv(scalar_csv)

    for year in s_ff_emission.loc[:target_year_abs].index.values:
        df_timeseries.loc[
            (year_x, year), 'ff_emission'] = s_ff_emission.loc[year]
    for year in s_total_emission.loc[:target_year_abs].index.values:
        df_timeseries.loc[
            (year_x, year), 'total_emission'] = s_total_emission.loc[year]
    for year in s_temp_anth.loc[:target_year_abs].index.values:
        df_timeseries.loc[
            (year_x, year), 'temp_anth'] = s_temp_anth.loc[year]
        df_timeseries.loc[
            (year_x, year), 'temp_anth_rel'] = (
                  s_temp_anth.loc[year] - temp_target_abs + temp_target_rel)
    df_timeseries.to_csv(timeseries_csv)

    ds_timeseries = df_timeseries.to_xarray()
    ds_scalar = df_scalar.to_xarray()

    ds = xr.merge([ds_timeseries, ds_scalar])
    ds.to_netcdf(meta_file)
