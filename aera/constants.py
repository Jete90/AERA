"""Definition of constants used by the AREA algorithm."""

# Logarithmic CO2 radiative forcing correlation factor
ALPHA = 5.35

# Anthropogenic warming in 2020 calculated using observational
# data. The data includes the Hadcrut5 global temperature
# time series (https://www.metoffice.gov.uk/hadobs/hadcrut5/),
# fossil fuel CO2 emissions and land-use change CO2 emissions
# from the global carbon project 2020
# (https://www.icos-cp.eu/science-and-impact/global-carbon-budget/2020),
# historical atmospheric CO2 mixing ratio from 
# Meinshausen et al. (2017), and non-CO2 radiative forcing
# from either the RCP/SSP database (standard) or from 
# IPCC data (optional)

OBS_ANTH_WARMING_2020 = 1.227 # RCP/SSP
#OBS_ANTH_WARMING_2020 = 1.181  # IPCC
