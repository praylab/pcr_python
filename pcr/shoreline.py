import pandas as pd 
import numpy as np

from datetime import datetime
from pcr import helper, storm, slr, erosion

def calculate_recovery(storms: pd.DataFrame, rec_rate: float) -> pd.Series: 
    '''
    Function to calculate shoreline recovery in between two storms 
    '''
    return storms['gap'] * rec_rate

def vector_calculate_recovery(gaps: np.array, rec_rate: float) -> pd.Series: 
    '''
    Function to calculate shoreline recovery in between two storms 
    '''
    return gaps * rec_rate


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


def vector_calculate_slr_retreat(slrs: np.array, m: float) -> pd.Series:
    '''
    Function to calculate shoreline retreat for a simulation of storms
    :param : storms a DataFrame which has sea level rise column
    :return : a series of shoreline retreat due to SLR
    '''
    retreat = np.empty(len(slrs))
    retreat[0] = 0
    retreat[1:] = calculate_inundation(slrs[:-1], slrs[1:], m)
    return retreat


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


def vector_track_shoreline(day_start:np.array, day_end:np.array, recovery:np.array, retreat:np.array, erosion:np.array) -> pd.DataFrame:
    '''

    '''
    # track shoreline position before and after storm
    time_value = np.empty(2*len(day_start), dtype=float)
    time_value[0::2] = day_start
    time_value[1::2] = day_end

    x0 = 0
    shoreline_change = np.empty(2*len(day_end))
    shoreline_change[0::2] = recovery - retreat
    shoreline_change[1::2] = - erosion

    shoreline_position = x0 + shoreline_change.cumsum()

    return [time_value, shoreline_change, shoreline_position] 


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


def vector_get_annual_statistics(track_time: np.array, shoreline_position: np.array, kind: str, date_start: datetime, date_end: datetime) -> np.array:
    '''
    Function to calculate annual statistics of shoreline position based on the passed kind
    :param shoreline_track: DataFrame which consist of shoreline position on a simulation
    :param kind: string of the kind of statistics. Available statistics are 'min', 'max', 'mean'
    :param date_start: start of the simulation horizon
    :param date_end: end of the simulation horizon; together with date_start this fixes the
        full set of years the returned array must cover, so years with no storms are still
        represented (via interpolation) rather than dropped, which would misalign the result
        with the fixed-size (t_years + 1) output array
    :return : an array of chosen statistics on each year of simulation, one row per calendar
        year from date_start to date_end inclusive
    '''
    valid_kinds = ['min', 'max', 'mean']

    if kind not in valid_kinds:
        raise ValueError('Invalid kind parameter. Please choose "mean", "min", or "max" as the kind parameter.')

    year_start = date_start.astype('datetime64[Y]').astype(int) + 1970
    year_end = date_end.astype('datetime64[Y]').astype(int) + 1970
    all_years = np.arange(year_start, year_end + 1)

    # track_time is chronological (storms are generated in time order), so the
    # corresponding calendar years form sorted runs -> reduce per-year with
    # np.ufunc.reduceat instead of building a DataFrame + pandas groupby per
    # simulation, which dominated runtime at scale.
    dates = helper.date_add_days(date_start, track_time)
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    # a storm's duration can push its end time past date_end, which would
    # otherwise put it in a year outside all_years -> fold it back into the
    # last simulated year instead (years is non-decreasing, so clipping keeps
    # it that way and doesn't break the contiguous per-year runs below)
    years = np.clip(years, year_start, year_end)

    is_new_year = np.empty(len(years), dtype=bool)
    is_new_year[0] = True
    is_new_year[1:] = years[1:] != years[:-1]
    year_starts = np.flatnonzero(is_new_year)
    present_years = years[year_starts]

    if kind == 'min':
        result = np.minimum.reduceat(shoreline_position, year_starts)
    elif kind == 'max':
        result = np.maximum.reduceat(shoreline_position, year_starts)
    else:  # mean
        sums = np.add.reduceat(shoreline_position, year_starts)
        counts = np.diff(np.append(year_starts, len(years)))
        result = sums / counts

    # some years may have no storms at all (no track_time entries), which would
    # otherwise leave gaps in the annual series -> reindex onto the full year
    # range and fill those gaps by linear interpolation between neighbouring years
    full_result = np.full(all_years.shape, np.nan)
    full_result[np.searchsorted(all_years, present_years)] = result

    missing = np.isnan(full_result)
    if np.any(missing):
        full_result[missing] = np.interp(all_years[missing], all_years[~missing], full_result[~missing])

    return full_result.reshape(-1, 1)


def run_monte_carlo(fitted_storms: dict, fitted_gap: dict, date_start: datetime, date_end: datetime, nr_simulation: int, nr_batch: int, stat_kind: str, rec_rate, max_dur) -> pd.DataFrame: 

    # storm char
    yearly_storm = 8
    nr_storm = (date_end.year - date_start.year) * yearly_storm

    # initialize an array 
    shoreline_stats = np.empty((101, nr_simulation))
    # shoreline_stats = []

    sim_count = 0

    while sim_count < nr_simulation:
        sampling_size = nr_storm * nr_batch

        # generate storm sample
        storms_sample = storm.generate(
            fitted_storm=fitted_storms, 
            sampling_size=sampling_size, 
            oversample=0.1, 
            max_dur=max_dur)

        # add gaps
        storms_sample = storm.sampling_gap_ecdf(
            fitted_gap=fitted_gap, 
            storms_sample=storms_sample
        )

        storm_count = 0

        for sim in range(nr_batch):

            # generate storm time series from date start to date end
            synthetic_storm = storm.generate_monsoon_ts(
                date_start=date_start, 
                date_end=date_end, 
                storms_sample=storms_sample, 
                fitted_gap=fitted_gap, 
                start_storm=storm_count
            )

            storm_count += len(synthetic_storm)

## TO-DO: check the code before this line

            # simulate sea level rise 
            synthetic_storm['slr'] = slr.simulate_slr(
                synthetic_storm=synthetic_storm, 
                date_start=date_start, 
                scenario='0', 
                wl0=0
            )

            # calculate storm-induced erosion
            _, synthetic_storm['erosion_storm'] = erosion.mendoza(synthetic_storm)

            # calculate recovery 
            synthetic_storm['recovery'] = calculate_recovery(
                storms=synthetic_storm,
                rec_rate=rec_rate #7/365
            )

            # calculate retreat due to slr 
            synthetic_storm['slr_retreat'] = calculate_slr_retreat(
                storms=synthetic_storm, 
                m=0.024
            )

            # track shoreline evolution 
            shoreline_track = track_shoreline(synthetic_storm)

            row = get_annual_statistics(shoreline_track, kind=stat_kind, date_start=date_start)
            
            try:
                shoreline_stats[:, sim_count] = row.flatten()
            except: 
                sim_count -= 1 # if there are any missing year, re-do the simulation

            # if sim_count == 0:
            #     shoreline_stats.append(row)
            # elif shoreline_stats[sim_count-1].shape[0] == row.shape[0]:
            #     shoreline_stats.append(row)
            # else: 
            #     sim_count -= 1

            sim_count += 1

            if sim_count >= nr_simulation:
                break

    return shoreline_stats