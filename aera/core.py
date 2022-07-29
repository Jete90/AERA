"""Core functions of the AERA algorithm.

Contains the following functions:
- calculate_anth_co2_rf : Calculates the anthropogenic
    CO2 radiative forcing.
- calculate_anth_temperature : Calculates the anthropogenic temperature.
- calculate_absolute_target_temperature: Calculates the absolute target
    temperature [K].
- calculate_remaining_emission_budget : Calculates the remaining
    emission budget (REB). The REB is the amount of CO2-fe emission that
    are still allowed to be emitted in the future.
- get_adaptive_emissions : MAIN FUNCTION. Calculates "optimal"
    near-future CO2 emissions.

"""

import copy as cp
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

from aera.fit import fit_anth_temperature
from aera import constants
from aera import utils
from aera import io
from aera import emission_curve


def calculate_anth_co2_rf(co2_conc, n2o_conc):
    """CO2 radiative forcing (RF) time series from anth CO2 and total N2O
    in the atmosphere.

    Simplified RF from (AR6 WG1 IPCC, Table 7.SM.1).

    Args:
        co2_conc (float): Global mean CO2 concentration (volume mixing
        ratio in units of ppm).
        n2o_conc (float): Global mean N2O concentration (volume mixing
        ratio in units of ppm).

    """

    a1 = -2.4785e-7
    b1 = 7.5906e-4
    c1 = -2.1492e-3
    d1 = 5.2488
    C0 = 277.15

    Camax = C0 - (b1/(2*a1))

    co2_conc_ts = co2_conc.loc[1700:2499].values
    n2o_conc_ts = n2o_conc.loc[1700:2499].values
    length = co2_conc_ts.shape[0]
    
    # Calculate alpha for the pure CO2 effect

    alpha = np.zeros(length)

    for i in range(length):
        if np.isnan(co2_conc_ts[i]) == 0:
            if co2_conc_ts[i] > Camax:
                alpha[i] = d1 - ( (b1**2) / (4*a1) )
            elif co2_conc_ts[i] <= Camax:
                if co2_conc_ts[i] >= C0:
                    alpha[i] = d1 + a1 * ((co2_conc_ts[i]-C0)**2) + b1 * (co2_conc_ts[i]-C0)
                elif co2_conc_ts[i] < C0:
                    alpha[i] = d1
        else:
            alpha[i] = np.nan

    # Calculate the alpha for the CO2 and N2O overlap
    
    alpha_n2o = np.zeros(length)

    for i in range(length):
        if np.isnan(n2o_conc_ts[i]) == 0:
            alpha_n2o[i] = c1 * np.sqrt(n2o_conc_ts[i])
        else:
            alpha_n2o[i] = np.nan

    # Calculate the RF for both alphas
    
    RF_CO2 = np.zeros(length)
    RF_N2O = np.zeros(length)

    for i in range(length):
        if np.isnan(co2_conc_ts[i]) == 0:
            RF_CO2[i] = alpha[i] * np.log(co2_conc_ts[i]/C0)
        else:
            RF_CO2[i] = np.nan

        if np.isnan(n2o_conc_ts[i]) == 0:
            RF_N2O[i] = alpha_n2o[i] * np.log(co2_conc_ts[i]/C0)
        else:
            RF_N2O[i] = np.nan

    # Return the total CO2 RF multiplied by the ERF adjustment from IPCC AR6 CH7

    return 1.05 * (RF_CO2 + RF_N2O)


def calculate_anth_temperature(rf_total, temp):
    """Anthropogenic temperature from total anthropogenic radiative
    forcing.

    Args:
        rf_total (array-like): Time series ot total anthropogenic
            radiative forcing (CO2+nonCO2).
        temp (array-like): Time series of global mean surface
            air temperature.

    Returns:
        temp_anth (array-like): Anthropogenic global mean surface
            air temperature.

    """
    temp_params, _ = curve_fit(
        fit_anth_temperature,
        rf_total,
        temp,
        # Initial guess for paramaters
        p0=[287.1, 0.5, 0.3, 0.5, 2., 300., 22.],
        # Upper and lower boundaries of [absolute temperature,
        # correlation factor between T and realized RF impulse,
        # relative contribution of timescale 1, relative contribution
        # of timescale 2, timescale 1 in years, timescale 2,
        # timescale 3)
        bounds=(
            [284., 0., 0.2, 0.3, 1.5, 100., 15.],
            [306., 5., 0.4, 0.5, 2.0, 600., 30.]
        )
    )
    return fit_anth_temperature(rf_total, *temp_params)


def calculate_absolute_target_temperature(
        temp_target_rel, model_start_year, s_rf_total, s_temp,
        temp_target_type, costum_anth_temp_func=None):
    """Calculate the absolute target temperature.

    Args:
        temp_target_rel (float): Relative temperature target
            (e.g. 1.5K).
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario
            simulation) was started.
        s_rf_total (pd.Series): Total (anth) radiative forcing
            time series.
        s_temp (pd.Series): Temperature (from model) time series.
        temp_target_type (int): Switch for different types of temperature
            targets.
            - 1: Temperature target estimated by observed remaining
                 warming from 2020 onwards
            - 2: Simulated warming anomaly with the reference period
                 1850-1900
        costum_anth_temp_func (function): See documentation in
            `get_adaptive_emissions`.

    Returns:
        temp_target_abs (float): Absolute target temperature [K].

    """
    model_start_year = max(1850, model_start_year)

    if temp_target_type == 1:
        # Calculate anthropogenic warming in 2020
        if costum_anth_temp_func is None:
            temp_anth_2020 = calculate_anth_temperature(
               s_rf_total.loc[model_start_year:2020].values,
               s_temp.loc[model_start_year:2020].values,
            )[-1]
        else:
            temp_anth_2020 = costum_anth_temp_func(
                temp_target_rel, temp_target_type, 2020, co2_preindustrial,
                model_start_year, df,
            )[-1]

        # Absolute target temperature is based on observed
        # anthropogenic warming in 2020
        temp_target_abs = (temp_anth_2020 +
            (temp_target_rel - constants.OBS_ANTH_WARMING_2020))
        print(f'Anthropogenic temperature 2020: {temp_anth_2020}')
    elif temp_target_type == 2:
        temp_target_abs = np.nanmean(
            s_temp.loc[model_start_year:1900].values) + temp_target_rel
    else:
        print('Invald temperature target type chosen (options are: {1, 2}).')
    return temp_target_abs


def _calculate_previous_emission_slope(year_x, meta_file):
    """Calculate the slope at Year X by using the emission
    curve from the previous stocktake.

    To make the emission curve as smooth as possible the
    AERA algorithm has to use the previously calculated
    emission curve parameters (i.e. a, b, and c).

    Args:
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        meta_file (str or pathlib.Path): File for temporary data which
            should be transfered from one run of the AERA algorithm
            to the next.

    """
    meta_file = Path(meta_file)
    if not meta_file.exists():
        return
    ds = xr.open_dataset(meta_file)
    try:
        a = ds.ec_a.sel(year_stocktake=year_x-5)
        b = ds.ec_b.sel(year_stocktake=year_x-5)
        c = ds.ec_c.sel(year_stocktake=year_x-5)
    except KeyError:
        return
    t = 5
    return 3 * a * t**2 + 2 * b * t + c


def calculate_remaining_emission_budget(
        temp_anth, total_emission, temp_target_abs, year_x,
        model_start_year, temp_abs_ts):
    """Calculate remaining emission budget.

    Args:
        temp_anth (array-like): Time series of anthropogenic temperature
            (without any natural variablity).
        total_emission (array-like): Time series of total emissions.
        temp_target_abs (float): Absolute target temperature.
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario simulation)
            was started.
        temp_abs_ts (array-like): Time series of measured/simulated
            temperature (including natural variablity).

    Returns:
        reb (float): Remaining emission budget until the target
            temperature is reached in Pg C.

    """
    # Substract the reference period temperature (1850-1900)
    dtemp_ref_yearx = (
        temp_anth.loc[year_x]) - temp_abs_ts.loc[model_start_year:1900].mean()
    print('Relative anthropogenic warming in Year X: ', dtemp_ref_yearx)
    print('Cumulative past emissions: ',
          total_emission.loc[model_start_year:year_x-1].sum())
    # Calculate TCRE (Cum. Emissions divided by anthropogenic warming)
    slope = total_emission.loc[model_start_year:year_x -
                               1].sum() / dtemp_ref_yearx
    # Multiply TCRE with remaing allowable warming
    reb = (temp_target_abs - temp_anth.loc[year_x]) * slope
    print('REB: ', reb)
    return reb


def get_adaptive_emissions(
        temp_target_rel, temp_target_type, year_x, co2_preindustrial,
        model_start_year, df, meta_file, costum_anth_temp_func=None):
    """Calculate "optimal" near-future CO2 emissions.

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
        co2_preindustrial (float): Mean pre-industrial atmospheric CO2
            concentration (in ppmv).
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario simulation)
            was started.
        df (pd.DataFrame): Pandas dataframe with years (int) as index
            and the following columns (see utils.get_base_df which
            provides a skeleton of this dataframe):
            - temp:  Global annual mean temperature time series for
              the period (in Kelvin).
            - co2_conc: Global annual mean CO2 concentration
              time series (in ppmv).
            - ff_emission: Global annual mean fossil fuel CO2
              emission time series (in Pg C / yr).
            - lu_emission: Global annual mean land use change
              CO2 emission time series (in Pg C / yr).
            - non_co2_emission: Global annual mean non-CO2 emission (in
              CO2-eq Pg C / yr)
            - rf_non_co2: Global annual mean non-CO2
              anthropogenic radiative forcing (in units TODO).
        meta_file (str or pathlib.Path): File for temporary data which
            should be transfered from one run of the AERA algorithm
            to the next.
        costum_anth_temp_func (function): ONLY FOR ADVANCED USE CASES!
            Costum, user-defined function that calculates the
            anthropogenic temperature. This function is given the
            following args (same as given to `get_adaptive_emissions`):
                - temp_target_rel
                - temp_target_type
                - year_x
                - co2_preindustrial
                - model_start_year
                - df
            It must return a list/numpy-array with the anthropogenic
            temperature for the years from `model_start_year` until
            the given `year_x` (in the calculation of the
            anthropogenic temperature of the year 2020, "2020" is
            given instead of `year_x`).

    Returns:
        s_ff_emission (pd.Series): Annual globally integrated fossil fuel
            CO2 emission time series (in Pg C / yr).

    """
    # Some CMIP5 models start later than 1850. The earliest start year
    # is 1850 because no observed temperature record exists before.
    # Simulated parameters before 1850 are not used
    model_start_year = max(
        1850, model_start_year)
    utils.validate_df(df, year_x, model_start_year)

    total_emission_cols = ['ff_emission', 'lu_emission', 'non_co2_emission']
    s_total_emission = df[total_emission_cols].sum(skipna=True, axis=1)
    # Calculate radiative forcing from CO2 only
    s_rf_co2 = calculate_anth_co2_rf(df['co2_conc'], df['n2o_conc'])
    # Add nonCO2 RF to get total radiative forcing
    s_rf_total = df['rf_non_co2'] + s_rf_co2

    # Calculate the temperature target
    temp_target_abs = calculate_absolute_target_temperature(
        temp_target_rel, model_start_year, s_rf_total, df['temp'],
        temp_target_type=temp_target_type)

    # Extract the temperature time series until the time of the stocktake
    s_temp_anth = df['temp'].loc[model_start_year:year_x].copy()
    # Extract anthropogenic warming
    if costum_anth_temp_func is None:
        s_temp_anth.loc[:] = calculate_anth_temperature(
            s_rf_total.loc[model_start_year:year_x].values,
            df['temp'].loc[model_start_year:year_x].values,
        )
    else:
        s_temp_anth.loc[:] = costum_anth_temp_func(
            temp_target_rel, temp_target_type, year_x, co2_preindustrial,
            model_start_year, df,
        )

    # Extract again the temperature time series until the time of the
    # stocktake, simulated/measured temperature and only anthropogenic
    # temperature will be needed later
    s_temp_abs = df['temp'].loc[model_start_year:year_x].copy()

    # Calculate remaining emissions budget
    reb = calculate_remaining_emission_budget(
        s_temp_anth, s_total_emission, temp_target_abs, year_x,
        model_start_year, s_temp_abs)

    # Read in slope at Year_X as estimated at previous stocktake
    previous_slope = _calculate_previous_emission_slope(year_x, meta_file)
    if previous_slope is not None:
        previous_slope = float(previous_slope)

    # Calculate the slope of the emissions curve at year X-1
    slope_tm1 = s_total_emission.loc[year_x]-s_total_emission.loc[year_x-1]
    slope_tm1 = float(slope_tm1)

    # Calculate the future emission curves
    ec = emission_curve.EmissionCurve.get_cheapest_curve(
        s_total_emission, year_x, reb, slope_tm1, temp_target_rel, previous_slope)

    # Add 5 (arbitrary number) years more to extand the emission curve
    # further in case of extrapolation problems if models need
    # emissions from the year ahead to calculate monthly emissions in
    # the 2nd half of the year
    additional_years = 5
    t = np.arange(1, ec.target_year_rel + additional_years + 2)
    year1 = int(year_x + 1)
    year2 = int(year1 + ec.target_year_rel) + additional_years
    s_total_emission.loc[year1:year2] = ec.get_values(t=t)
    print('CO2-fe emissions [Pg C] (fossil fuel CO2 + landuse + non-CO2) '
          'over next years:')
    print(s_total_emission.loc[year1:year2-5])

    # Calculate Fossil fuel emissions as the difference between
    # estimated total emissions and prescribed land-use and nonCO2
    # emissions
    s_ff_emission = (
        s_total_emission - df['lu_emission'] - df['non_co2_emission'])
    s_ff_emission.name = 'ff_emission'

    # Store data to metafile for debug and post-analysis
    io.store_metadata(
        meta_file, temp_target_rel, temp_target_abs, year_x,
        model_start_year, s_temp_anth, s_total_emission, s_ff_emission, ec)

    return s_ff_emission.loc[year1:year2]
