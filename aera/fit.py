"""Definition of fitting functions.

These functions are used in calls to scipy.optimize.curve_fit.

"""

import numpy as np


def fit_anth_temperature(rf_total, offset, slope, A1, A2, T1, T2, T3):
    """Calculate anth temperature from total anth radiative forcing
    with an impulse-response function.

    Assumption:
    1) Temperature is linearly depented on the realized
       radiative forcing (RRF).
    2) The anth radiative focings decay on three different
       timescales (T1, T2, T3).

    Reference:
    For more information on impulse-response function, please read
    
    Joos, F. et al. Carbon dioxide and climate impulse response 
    functions for the computation of greenhouse gas metrics: a 
    multi-model analysis. Atmos. Chem. Phys. 13, 2793–2825 (2013).
    
    Notes:
    - A1, A2, A3 are the weightings of the different time scales
    - T1, T2, T3,  are the decaying timescales

    Returns:
        dT_ant (float): Anthropogenic warming due to the given
            radiative forcings.

    """
    # The third weight is calculated by difference to the other two
    # weights as the total weight must equal one
    A3 = 1.0 - (A1 + A2)

    # These coefficients calculate the realized radiative forcing in
    # year i from year k
    coeff_ik = lambda i, k: (
        A1 * (1.0 - np.exp((k - i - 1) / T1)) +
        A2 * (1.0 - np.exp((k - i - 1) / T2)) +
        A3 * (1.0 - np.exp((k - i - 1) / T3))
        )

    # RRF: Realized Radiative Forcing
    # i runs over all years between 1850 and year X (indexes from 0 to
    # X-1850) for each year marked by index i, the radtive forcing is
    # the sum of the sum of the realized radiative forcing at that time
    # k runs from 1 to i+1 to calculate the realized of each year
    # before i
    # The realized RF from each year is then summed up to to get the
    # total realized radiative forcing
    rrf_total = np.array([
        sum([(rf_total[k] - rf_total[k - 1]) * coeff_ik(i, k)
            for k in range(1, i+1)])
            for i in range(len(rf_total))
            ])
    return slope * rrf_total + offset
