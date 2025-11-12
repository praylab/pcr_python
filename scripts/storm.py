import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy import stats
from copulas.bivariate import Clayton

def detect(hs:np.array, dir:np.array, tp:np.array, time:np.array, ts_hs:np.array, ts_dur:np.array) -> pd.DataFrame:
    '''
    Detect storms from a time series of wave with Peak Over Threshold approach
    :param hs: np.array, series of significant wave heights in meters
    :param dir: np.array, series of wave directions in degrees
    :param tp: np.array, series of peak wave periods in second
    :param time: np.array, series of time points corresponding to hs, dir, tp in days 
    :param ts_hs: float, threshold percentile of significant wave height to define a storm in percentage (e.g., 95 for 95th percentile)
    :param ts_dur: float, threshold duration (in hours) to define a storm in hours
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
    peaks, props = signal.find_peaks(storm_dur_cum, height=ts_dur)
    mask = props["peak_heights"] > ts_dur
    peak_indices = peaks[mask]

    # count number of Hs on each storm cluster 
    storm_nr = (storm_dur_cum[peak_indices] / delta_t).astype(int)

    # Collect storm statistics
    start_idx = peak_indices - storm_nr
    end_idx = peak_indices

    # calculate the duration from the end of the last storm to the start of this storm 
    end_prev = np.concat(([start_idx[0]], peak_indices[:-1]))   # first storm has no previous storm

    storms = {
        "start": time[start_idx],
        "end": time[peak_indices],
        "duration": storm_nr*delta_t,
        "time": [time[start:end] for start, end in zip(start_idx, end_idx+1)],
        "hs": [hs[start:end] for start, end in zip(start_idx, end_idx+1)],
        "tp": [tp[start:end] for start, end in zip(start_idx, end_idx+1)],
        "dir": [dir[start:end] for start, end in zip(start_idx, end_idx+1)],
        "hs_max": [np.max(hs[start:end]) for start, end in zip(start_idx, end_idx+1)],
        "dir_mean": [np.mean(dir[start:end]) for start, end in zip(start_idx, end_idx+1)],
        "tp_mean": [np.mean(tp[start:end]) for start, end in zip(start_idx, end_idx+1)],
        "gap": [time[start] - time[prev] for start, prev in zip(start_idx, end_prev)]
    }

    storms_df = pd.DataFrame(storms)

    storms_df["season"] = np.where(np.array(storms["gap"]) > 150, 1, 0)

    return storms_df


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

    return [gap_ecdf, lambdaYear, lambdaSts]


def fit_gev(x: pd.Series) -> list: 
    '''
    Fit storm variable into gev distribution 
    :return : list of [shape, location, scale] parameter of fitted GEV distribution'''

    x = x.values 
    return stats.genextreme.fit(x, loc=np.mean(x)) # optimise with the first guess of location being the mean of the data
    

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

def empirical_cdf(x: pd.Series) -> list: 
    '''
    function return to sorted x and the corresponding empirical cummulative distribution function 
    :return : list of [sorted x, corresponding CDF]
    '''
    x = np.sort(x.values)
    return [x, np.arange(1, len(x) + 1) / len(x)]

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

    storms = detect(hs, dir, tp, time, ts_hs, ts_dur)
    print("Detected storms:", storms)

