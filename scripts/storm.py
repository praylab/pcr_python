import numpy as np
import pandas as pd
import scipy.signal as signal

from datetime import datetime
from scipy import stats
from scripts import helper
from copulas.bivariate import Clayton

def detect(hs:np.array, dir:np.array, tp:np.array, time:np.array, ts_hs:float, ts_dur:float, ts_between:float=0) -> pd.DataFrame:
    '''
    Detect storms from a time series of wave with Peak Over Threshold approach
    :param hs: np.array, series of significant wave heights in meters
    :param dir: np.array, series of wave directions in degrees
    :param tp: np.array, series of peak wave periods in second
    :param time: np.array, series of time points corresponding to hs, dir, tp in days 
    :param ts_hs: float, threshold percentile of significant wave height to define a storm in percentage (e.g., 95 for 95th percentile)
    :param ts_dur: float, threshold duration (in hours) to define a storm in hours
    :param ts_between: minimum time between two consecutive storm that is assumed to be independent, in hour
    :return: dataframe of detected storms with start time, end time, duration, max hs, mean dir, mean tp
    '''

    # obtain index where hs exceeds threshold
    threshold = np.percentile(hs, ts_hs)
    storm_id = np.where(hs >= threshold, 1, 0)

    # initiate storm id to track time 
    delta_t = (time[1]-time[0])*24  # assuming uniform time steps in hour
    storm_dur = storm_id * delta_t  # duration in hours

    # get cumulative sum of time where hs exceeds threshold
    storm_dur_cum = np.zeros_like(storm_dur) 
    for i in range(1, len(storm_dur)):
        if storm_dur[i] > 0:
            storm_dur_cum[i] = storm_dur_cum[i-1] + storm_dur[i]

    # find the peak of duration at the end of each storm
    ends, props = signal.find_peaks(storm_dur_cum, height=ts_dur)
    mask = props["peak_heights"] > ts_dur
    end_indices = ends[mask]

    # count number of Hs on each storm cluster 
    storm_step = (storm_dur_cum[end_indices] / delta_t).astype(int)

    # Collect storm statistics
    start_idx = end_indices - storm_step
    end_idx = end_indices

    # detect dependent storm if any threshold is given
    if ts_between != 0:

        # detect storm which is dependent to the next storm 
        dep_storm_nr = dependent_storm(hs, delta_t, start_idx, end_idx, ts_between)

        # merge the dependent storm 
        start_idx, end_idx, storm_step = merge_dep_storm(dep_storm_nr, start_idx, end_idx)

    # calculate the duration from the end of the last storm to the start of this storm 
    end_prev = np.concat(([start_idx[0]], end_idx[:-1]))   # first storm has no previous storm

    # create dataframe of storm time series with storm id 
    storms_ts = pd.DataFrame({ 
        'time': [time[start:end] for start, end in zip(start_idx, end_idx+1)],
        'hs': [hs[start:end] for start, end in zip(start_idx, end_idx+1)],
        'tp': [tp[start:end] for start, end in zip(start_idx, end_idx+1)],
        'dir': [dir[start:end] for start, end in zip(start_idx, end_idx+1)],
    }).explode(['time', 'hs', 'tp', 'dir']).reset_index(drop=False).rename(columns={'index': 'storm_id'})

    # create DataFrame of storm statistics 
    storms_df = pd.DataFrame()

    storms_df['start'] = storms_ts.groupby('storm_id')['time'].first()
    storms_df['end'] = storms_ts.groupby('storm_id')['time'].last()
    storms_df['duration'] = storm_step*delta_t
    storms_df['hs_max'] = storms_ts.groupby('storm_id')['hs'].max().astype(float)
    storms_df['dir_mean'] = storms_ts.groupby('storm_id')['dir'].mean().astype(float)
    storms_df['tp_mean'] = storms_ts.groupby('storm_id')['tp'].mean().astype(float)
    storms_df['gap'] = [time[start] - time[prev] for start, prev in zip(start_idx, end_prev)]

    storms_df["season"] = np.where(np.array(storms_df["gap"]) > 150, 1, 0)

    return storms_df, storms_ts


def dependent_storm(hs:np.array, delta_t:float, start_indices:np.array, end_indices:np.array, ts_between:float=48):
    '''
    return an array of storm number which is dependent to the next consecutive storm. 
    :param hs: an array-like of significant wave height 
    :param delta_t: time delta between two hs point, in hour
    :param start_indices: an array-like of indices of the start of all detected storm 
    :param end_indices: an array-like of indices of the end of all detected storm 
    :param ts_between: minimum time between two consecutive storm that is assumed to be independent, in hour
    :return: list 
    '''
    # find the indices of peak of each storm
    peak_indices = [hs[start:end].argmax() for start, end in zip(start_indices, end_indices+1)] + start_indices

    # find the peaks which are assumed not independent
    dep_storm_nr = np.where(np.diff(peak_indices)*delta_t < ts_between)

    return dep_storm_nr


def merge_dep_storm(dep_storm_nr: np.array, start_indices:np.array, end_indices: np.array):
    '''
    return arrays of start and end indices of independent storm.
    :param dep_storm_nr: an array-like of the number of dependent storm 
    :param start_indices: an array-like of indices of the start of all detected storm 
    :param end_indices: an array-like of indices of the end of all detected storm 
    :return start_new: an array-like of indices of the start of independent storm
    :return end_new: an array-like of indices of the end of independent storm
    :return storm_step_new: an array-like of number of wave timestep inside each storm 
    '''
    # remove start on index dep_storm_nr + 1 
    start_new = np.array([start_indices[i] for i in range(len(start_indices)) if i not in dep_storm_nr[0] + 1])

    # remove end on index dep_storm_nr
    end_new = np.array([end_indices[i] for i in range(len(start_indices)) if i not in dep_storm_nr[0]])

    # calculate new storm step 
    storm_step_new = end_new - start_new

    return start_new, end_new, storm_step_new


def generate(fitted_storm: dict, sampling_size:int, oversample:float=0.1, max_dur:float=0) -> pd.DataFrame:
    '''
    generate storm based on fitting 
    '''
    # unpack the parameter
    pgev_hs = fitted_storm['gev_hs']
    pgev_dur = fitted_storm['gev_dur']
    clayton_copula = fitted_storm['clayton']
    ecdf_dir = fitted_storm['ecdf_dir']
    p_tp = fitted_storm['lin_tp']

    # sampling
    sampling_size = int(sampling_size * (1.0+oversample))

    # sample Hs and Duration 
    hs, dur = sample_copula_clayton(clayton_copula, pgev_hs, pgev_dur, sampling_size)

    # sample dir from empirical cdf 
    dir = sample_ecdf(ecdf_dir, sampling_size)

    # sample tp from linear fit 
    tp = f_linear(p_tp, hs)

    # cast type 
    hs = hs.astype(np.float32, copy=False)
    dur = dur.astype(np.float32, copy=False)
    dir = dir.astype(np.float32, copy=False)
    tp = tp.astype(np.float32, copy=False)

    # filter max duration 
    if max_dur > 0:
        mask = dur < max_dur
        hs = hs[mask]
        dur = dur[mask]
        dir = dir[mask]
        tp = tp[mask]

    # make a dataframe
    # storms_sample = pd.DataFrame(
    #     {
    #         "hs": hs,
    #         "duration": dur,
    #         "direction": dir,
    #         "tp": tp
    #     }, 
    #     copy=False
    # ).astype(np.float32)

    # # limit to the max duration 
    # if max_dur != 0: 
    #     storms_sample = storms_sample[storms_sample.duration < max_dur].reset_index(drop=True)

    return hs, dur, dir, tp


# TODO: maybe make the function simple? 
def generate_monsoon_ts(date_start: datetime, date_end: datetime, storms_sample: pd.DataFrame, fitted_gap: pd.DataFrame, start_storm: int = 0) -> pd.DataFrame: 
    '''
    Function to generate one simulation from date_start to date_end 
    '''

    sim_days = (date_end-date_start).days
    sim_years = date_end.year-date_start.year
    size_year = int(sim_years*1.2)
    size_storm = len(storms_sample)
    dur_day = storms_sample['duration']/24

    # sample year storm season length
    year_sample = np.random.poisson(lam=fitted_gap['lambda_year'], size=size_year)
    season_sample = np.random.poisson(lam=fitted_gap['lambda_season'], size=size_year)

    storm_count = start_storm
    season_count = 0
    day_count = 0
    storm_synth = []

    hs = storms_sample['hs']
    dir = storms_sample['direction']
    dur = storms_sample['duration']
    tp = storms_sample['tp']
    gap = storms_sample['gap']

    while day_count < sim_days:
        # fill the storm season
        start_season = day_count
        end_season = day_count + season_sample[season_count] 
        
        while (day_count < end_season) and (day_count < sim_days):
            
            if storm_count >= size_storm:
                raise ValueError('samples are exhausted')
            
            hs_i = hs[storm_count]
            dir_i = dir[storm_count]
            dur_i = dur[storm_count]
            tp_i = tp[storm_count]
            start_i = day_count
            end_i = day_count + dur_day[storm_count]

            day_count += dur_day[storm_count] + gap[storm_count]
            storm_count += 1

            storm_synth.append({
                'hs': hs_i, 
                'direction': dir_i,
                'duration': dur_i,
                'tp': tp_i,
                'day_start': start_i,
                'day_end': end_i
            })

        # move to the next season
        day_count = start_season + year_sample[season_count]
        season_count += 1

    synthetic_storm = pd.DataFrame(storm_synth)

    # gap is defined as the day length from the end of last storm to the start of this storm
    # day_end = synthetic_storm["day_end"].values
    # next_day_start = np.concatenate([synthetic_storm["day_start"][1:].values, [synthetic_storm["day_end"].iloc[-1]]]) # leaving the last storm has 0 gap
    
    # synthetic_storm["gap"] = next_day_start-day_end

    day_start = synthetic_storm["day_start"].values
    prev_day_end = np.concatenate([[synthetic_storm["day_start"].iloc[0]], synthetic_storm["day_end"][:-1].values]) # leaving the first storm has 0 gap

    synthetic_storm['gap'] = day_start - prev_day_end

    return synthetic_storm


def sampling_gap_ecdf(fitted_gap: dict, storms_sample: pd.DataFrame) -> pd.DataFrame:
    '''
    Function to sample gap from empirical CDF and add to storms_sample DataFrame
    Args:
        fitted_gap (dict): dictionary containing fitted gap information
        storms_sample (pd.DataFrame): DataFrame containing sampled storms
    '''

    # sample gap from empirical CDF
    sampling_size = storms_sample.shape[0]

    gaps = fitted_gap['ecdf_gap'][0]
    p_gap = fitted_gap['ecdf_gap'][1]

    p_below = p_gap[np.where(gaps >= 1)[0][0]]
    r = np.random.uniform(size=sampling_size) * (1-p_below) + p_below

    storms_sample["gap"] = sample_ecdf(fitted_gap['ecdf_gap'], r=r).astype(float)

    return storms_sample


def fit_storm(storms:pd.DataFrame) -> list: 
    # get gev param for hs and dur 
    pgev_hs = gev_fit(storms.hs_max)
    pgev_dur = gev_fit(storms.duration)

    # fit copula clayton 
    clayton_copula = fit_copulas_clayton(storms.hs_max, storms.duration, pgev_hs, pgev_dur)

    # fit direction to empirical cdf 
    ecdf_dir = empirical_cdf(storms.dir_mean)

    # linear fit of wave peak 
    p_tp = np.polyfit(storms.hs_max, storms.tp_mean, 1)

    return({
     "gev_hs": pgev_hs, 
     "gev_dur": pgev_dur, 
     "clayton": clayton_copula, 
     "ecdf_dir": ecdf_dir, 
     "lin_tp": p_tp   
    })


def fit_gap_monsoon(storms:pd.DataFrame) -> list: 
    '''
    Fit storm gaps to a probability distribution
    :param storms: dataframe of detected storms with gap and season information
    :return: list of fitted distribution of gap, years and season
    '''

    # fit gap to empirical distribution
    gap_ecdf = empirical_cdf(storms.gap[storms.gap < 150]) # only consider gaps within a season 

    # calculate gap between storm season 
    gk = np.array(storms.season)
    tt = np.zeros(len(gk))

    # record the start and end of every storm season 
    for i in range(len(gk)):
        if gk[i] == 1:
            tt[i] = storms["start"][i]
            tt[i-1] = storms["end"][i-1]

    t2 = tt[tt != 0]

    # calculate the duration between consecutive calm and storm season
    gap2 = np.concat(([0], np.diff(t2)))

    calm_season = gap2[1::2]
    storm_season = gap2[2::2]

    # check if we have pair of calm and storm season 
    if len(calm_season) != len(storm_season):
        calm_season = calm_season[:int(len(t2)/2-1)]

    year = calm_season + storm_season

    # MLE fit to poisson distribution 
    lambdaYear = np.mean(year)
    lambdaSts = np.mean(storm_season)

    return {
        'ecdf_gap': gap_ecdf, 
        'lambda_year': lambdaYear,
        'lambda_season': lambdaSts
    }


def fit_lambda(storms:pd.DataFrame, fillna:str=None) -> list: 
    '''
    Fit monthly lambda for non-homogeneous poisson process

    '''
    # get the monthly count of storm
    count_df = monthly_count(storms)
    
    # calculate data length in year
    date_storm = helper.datenum_to_datetime(storms['start'])
    years = date_storm[-1].year - date_storm[0].year

    # calculate the rate of event in event/year
    lambdas = (count_df['count'] / years * 12)

    # handle nans (if any)
    if fillna == 'interp': 
        lambdas = helper.interp_nan_array(lambdas.values)
    elif fillna == 'zeros': 
        lambdas = lambdas.fillna(0).values
    elif fillna == None: 
        pass
    # raise error?
    else: 
        print('Pick between "interp" or "zeros"')

    return lambdas


def fit_lambda_gap(storms:pd.DataFrame, fillna:str=None) -> list: 
    '''
    Fit yearly event intensity (lambda) from storm gaps instead of interval between storm starts
    
    :param storms: DataFrame of detected storms with 'duration' and 'start' columns
    :type storms: pd.DataFrame
    :param fillna: method to handle NaN values in lambda calculation ('interp' or 'zeros')
    :type fillna: str
    :return: lambda value for each month 
    :rtype: list
    '''
    # Fit lambda to the gap, rather than to the interval between storm starts
    # get the monthly count of storm
    count_df = monthly_count(storms)

    # create copy of detected storm to avoid modifying original data
    storms = storms[['duration', 'start']].copy()
    storms['date_start'] = helper.datenum_to_datetime(storms['start'])
    # TODO: might be wrong? if the entry is not covering the whole year. Additionally is it should be +1? 
    n_year = storms['date_start'].dt.year.max() - storms['date_start'].dt.year.min() 

    # get the monthly total duration of storm (in hours)
    count_df['sum_dur'] = storms.groupby(by=storms['date_start'].dt.month)['duration'].sum() # in hours 

    # calculate the amount of 'available time' in each month (in hours)
    hour_month = np.array([31,28.25,31,30,31,30,31,31,30,31,30,31]) * 24
    hour_month_total = hour_month * n_year

    # calculate modified year: number of year equivalent without storm activity 
    count_df['year'] = (hour_month_total - count_df['sum_dur']) / hour_month

    # calculate the intensity of storm per year for each month  
    count_df['lambda_mod'] = count_df['count'] / count_df['year'] * 12

    lambdas = count_df['lambda_mod']

    # handle nans (if any)
    fillna = 'zeros'  
    if fillna == 'interp': 
        lambdas = helper.interp_nan_array(lambdas.values)
    elif fillna == 'zeros': 
        lambdas = lambdas.fillna(0).values
    elif fillna == None: 
        pass
    # raise error?
    else: 
        print('Pick between "interp" or "zeros"')

    return lambdas


def monthly_count(storms:pd.DataFrame) -> pd.DataFrame:
    '''
    Function to count monthly storm from storms DataFrame
    '''
    # get the datetime of storm to gain month 
    date_storm = helper.datenum_to_datetime(storms['start'])
    storms['months'] = [dt.month for dt in date_storm]

    # count monthly storm 
    count_df = pd.DataFrame({
        'months': np.arange(1,13)
    }).set_index('months')

    count_df['count'] = storms[['hs_max', 'months']].groupby('months').count().rename(columns={
            'hs_max': 'count'
        })

    return count_df

def gev_fit(x: pd.Series) -> list: 
    '''
    Fit storm variable into gev distribution 
    :return : list of [shape, location, scale] parameter of fitted GEV distribution'''

    x = x.values 
    return stats.genextreme.fit(x, loc=np.mean(x), scale=np.std(x)) # optimise with the first guess of location being the mean of the data
    

def gev_cdf(x: pd.Series, pgev: list) -> np.array:
    '''
    return to cdf of gev based on given parameter 
    x: variable of interest 
    pgev: list of [shape, location, scale] parameter of fitted GEV
    return: np.array of the corresponding cdf of x
    '''
    x = x.values 
    return stats.genextreme.cdf(x, pgev[0], loc=pgev[1], scale=pgev[2])


def fit_copulas_clayton(x: pd.Series, y: pd.Series, pgev_x: list, pgev_y:list): 
    '''
    Fit two variable into bivariate joint distribution coupla clayton
    x, y: series of variable 

    '''

    x_cdf = gev_cdf(x, pgev_x)
    y_cdf = gev_cdf(y, pgev_y)

    clayton_copula = Clayton()
    clayton_copula.fit(np.column_stack((x_cdf, y_cdf)))

    return clayton_copula


def sample_copula_clayton(clayton_copula:Clayton, pgev_x:list, pgev_y:list, sampling_size:int) -> list:
    u = clayton_copula.sample(sampling_size)
    x = stats.genextreme.ppf(u[:,0], pgev_x[0], pgev_x[1], pgev_x[2]).astype(np.float32)
    y = stats.genextreme.ppf(u[:,1], pgev_y[0], pgev_y[1], pgev_y[2]).astype(np.float32)

    return [x,y]


def empirical_cdf(x: pd.Series) -> list: 
    '''
    function return to sorted x and the corresponding empirical cummulative distribution function 
    :return : list of [sorted x, corresponding CDF]
    '''
    x = np.sort(x.values)
    return [x, np.arange(1, len(x) + 1) / len(x)]


def sample_ecdf(ecdf: list, sampling_size: int=None, r: np.array=None) -> np.array:
    '''
    Function to sample from empirical pair of x and cdf.
    :param ecdf: list of [x, cdf]
    :param sampling_size: int, number of samples to generate (required if r is None)
    :param r: np.array, uniform random numbers between 0 and 1 to be used for sampling (optional)
    
    :return : np.array of sampled values from empirical distribution
    '''
    # If r is not provided, generate it and determine sampling size
    if r is None:
        if sampling_size is None:
            raise ValueError("Either 'r' or 'sampling_size' must be provided.")
        r = np.random.uniform(size=sampling_size)
    else:
        # If r is provided, use its length as the sampling size
        sampling_size = len(r)

    # Perform the interpolation
    return np.interp(r, ecdf[1], ecdf[0])

def f_linear(p: list, x:np.array) -> np.array:
    '''
    Function to return to y = a*x + b
    p is a list of [a, b]
    x is the x-coordinates at which to evaluate the function 
    return to y
    '''
    return p[0]*x + p[1]


if __name__ == "__main__": # this only runs when this script is executed directly
    # test out the function using data from data/wave_srilanka.csv
    wave_data = pd.read_csv('data/wave_srilanka.csv')

    # get hs, dir, tp, time from dataframe
    time = wave_data.iloc[:, 0].values
    hs = wave_data.iloc[:, 1].values
    dir = wave_data.iloc[:, 2].values
    tp = wave_data.iloc[:, 3].values
    ts_hs = 95
    ts_dur = 12.0

    storms, _ = detect(hs, dir, tp, time, ts_hs, ts_dur)
    fitted_storms = fit_storm(storms)
    fitted_gap = fit_gap_monsoon(storms)
    storms_sample = generate(fitted_storms, 1000)

    print('Detected storms: ', storms)
