import pandas as pd 
import numpy as np

from datetime import datetime
from scripts import helper

def calculate_recovery(storms: pd.DataFrame, rec_rate: float) -> pd.Series: 
    '''
    Function to calculate shoreline recovery in between two storms 
    '''
    return storms['gap'] * rec_rate


def calculate_inundation(wl0: float, wl: float, m: float) -> float: 
    '''
    function to calculate shoreline retreat due to passive inundation 
    '''
    return (wl-wl0) / m


def calculate_slr_retreat(storms: pd.DataFrame, m: float) -> pd.Series: 
    '''
    Function to calculate shoreline retreat for a simulation of storms
    :param : storms a DataFrame which has sea level rise column
    :return : a series of shoreline retreat due to SLR
    '''
    wl1 = storms['slr'].iloc[:-1]
    wl2 = storms['slr'].iloc[1:]
    retreat = [calculate_inundation(wl1, wl2, m) for wl1, wl2 in zip(wl1, wl2)]

    retreat.insert(0,0)
    return pd.Series(retreat)


def track_shoreline(storms: pd.DataFrame) -> pd.DataFrame:
    '''

    '''
    # track shoreline position before and after storm 
    time_value = np.empty(2*len(storms), dtype=float)
    time_value[0::2] = storms['day_start'].values
    time_value[1::2] = storms['day_end'].values

    shoreline_track = pd.DataFrame({
        'day': time_value
    })

    x0 = 0
    shoreline_change = np.empty(2*len(storms))
    shoreline_change[0::2] = storms['recovery'].values - storms['slr_retreat']
    shoreline_change[1::2] = - storms['erosion_storm'].values

    shoreline_track['shoreline_change'] = shoreline_change
    shoreline_track['shoreline_position'] = x0 + shoreline_track['shoreline_change'].cumsum()

    return shoreline_track


def get_annual_statistics(shoreline_track: pd.DataFrame, kind: str, date_start: datetime) -> np.array:
    '''
    Function to calculate annual statistics of shoreline position based on the passed kind
    :param shoreline_track: DataFrame which consist of shoreline position on a simulation 
    :param kind: string of the kind of statistics. Available statistics are 'min', 'max', 'mean'
    :return : an array of chosen statistics on each year of simulation 
    '''
    # get an alias 
    track = shoreline_track
    track['time'] = helper.date_add_days(date_start, shoreline_track['day'])
    track['year'] = track['time'].dt.year
    
    valid_kinds = ['min', 'max', 'mean']

    if kind not in valid_kinds: 
        raise ValueError('Invalid kind parameter. Please choose "mean", "min", or "max" as the kind parameter.')

    return shoreline_track[['year', 'shoreline_position']].groupby('year').agg(kind).values