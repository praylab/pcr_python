import os 

import numpy as np
import pandas as pd
import xarray as xr 

from datetime import datetime
from scipy.integrate import cumulative_trapezoid

from pcr import geo

def curve_ar5(scenario):
    '''
    AR5 SLR curve
    :param scenario: str, one of 'RCP26', 'RCP45', 'RCP60', 'RCP85', '0' (no SLR)
    :return: np.array, coeffients a and b for SLR = a*x^2+b*x where x is days since 2018-1-1
    '''
    if scenario == 'RCP85':
        return [3.955e-10, 9.999e-06]
    elif scenario == 'RCP60':
        return [1.708e-10, 1.035e-05]
    elif scenario == 'RCP45':
        return [1.429e-10, 1.086e-05]
    elif scenario == 'RCP26':
        return [2.188e-12, 1.173e-05]
    elif scenario == '0':
        return [0.0, 0.0]
    
def calculate_slr(days_since_2018, scenario, projection='AR5'):
    '''
    Calculate sea level rise based on days since 2018-1-1 and scenario
    :param days_since_2018: int or np.array, number of days since 2018-1-1
    :param scenario: str, one of 'RCP26', 'RCP45', 'RCP60', 'RCP85', '0' (no SLR)
    :return: float or np.array, sea level rise in meters
    '''
    if projection != 'AR5':
        raise NotImplementedError("Only AR5 projection is implemented.")
    
    if scenario not in ['RCP26', 'RCP45', 'RCP60', 'RCP85', '0']:
        raise NotImplementedError("Invalid Scenario, please choose between 'RCP26', 'RCP45', 'RCP60', 'RCP85', '0'")
    
    a, b = curve_ar5(scenario)
    slr = a * (days_since_2018 ** 2) + b * days_since_2018
    return slr

def simulate_slr(synthetic_storm: pd.DataFrame, date_start, scenario, wl0) -> pd.Series:
    # calculate days since the refer date (1st of January 2018)
    if scenario == '0':
        return pd.Series(np.zeros(len(synthetic_storm)))

    days = (date_start - datetime(2018, 1, 1)).days + synthetic_storm['day_start']

    slr_values = calculate_slr(days, scenario)

    # return to series of water level change where on the first day equal to wl0
    return pd.Series(wl0 - slr_values[0] + slr_values)

def vector_simulate_slr(day_start: np.array, date_start, scenario, wl0) -> pd.Series:
    # calculate days since the refer date (1st of January 2018)
    if scenario == '0':
        return np.zeros(len(day_start))

    days = (date_start - datetime(2018, 1, 1)).days + day_start

    slr_values = calculate_slr(days, scenario)

    # return to series of water level change where on the first day equal to wl0
    return (wl0 - slr_values[0] + slr_values)
    

def import_ar6_curve(scenario: str, lon: float, lat: float, quantile:float=0.5, date_start:np.datetime64=None, dir: str = './data/AR6_slr') -> list:
    '''
    import IPCC AR6 sea level change rate curve, return to list of rates and years if date_start is not provided else to days since date_start
    '''
    filename = f'total_{scenario}_medium_confidence_rates.nc'
    ds = xr.open_dataset(os.path.join(dir,filename))

    # find the closest point 
    idx = geo.find_closest(lon, lat, ds.lon, ds.lat, 'locations')
    point = ds.isel(locations=idx)

    # extract the curve 
    rates = point.sel(quantiles=quantile).sea_level_change_rate.values / 1000  # in meter / year
    years = point.years.values

    # convert if applicable
    if date_start:
        year_start = (years - 1970).astype('datetime64[Y]')
        days = (year_start - date_start).astype('timedelta64[D]').astype(int)

        return rates/365.25, days
    else: 
        return rates, years

def vector_simulate_slrAR6(day_start: np.array, rate_sl: np.array, day_sl: np.array, wl0: float, scenario: str):
    if scenario == '0': 
        return np.zeros(len(day_start))
    
    rate_sim = np.interp(day_start, day_sl, rate_sl)
    wls = cumulative_trapezoid(rate_sim, x=day_start, initial=wl0)

    return wls

# test out the function
if __name__ == "__main__": # this only runs when this script is executed directly
    import plotly.express as px
    
    days = np.arange(0, 365*100, 365)  # every year for 10 years
    scenario = 'RCP85'
    slr_values = calculate_slr(days, scenario)

    # plotting the results
    fig = px.line(
        x=days/365, 
        y=slr_values*1000, 
        labels={'x': 'Years since 2018', 'y': 'Sea Level Rise (mm)'},
        title=f'Sea Level Rise under {scenario} Scenario'
    )

    fig.show()