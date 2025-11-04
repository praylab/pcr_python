import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

def detect(hs:np.array, dir, tp, time, ts_hs, ts_dur) -> pd.DataFrame:
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
    start_indices = peak_indices - storm_nr

    # Collect storm statistics
    storms = []
    prev_end_time = None

    for i, peak in enumerate(peak_indices):
        start_idx = max(0, peak - storm_nr[i])
        end_idx = peak

        storm = {
            'start': time[start_idx],
            'end': time[end_idx-1],
            'duration': (time[end_idx-1] - time[start_idx]) * 24,  # hours, if time is in days
            'time': time[start_idx:end_idx],
            # enable only if needed to save memory
            # 'hs': hs[start_idx:end_idx],
            # 'dir': dir[start_idx:end_idx],
            # 'tp': tp[start_idx:end_idx],
            'hs_max': float(np.max(hs[start_idx:end_idx])),
            'dir_mean': float(np.mean(dir[start_idx:end_idx])),
            'tp_mean': float(np.mean(tp[start_idx:end_idx])),
        }

        if prev_end_time is None: 
            storm['gap'] = 0
            storm['season'] = 0
        else: 
            calm_gap = (time[start_idx] - prev_end_time) # in days
            storm['gap'] = calm_gap
            storm['season'] = 1 if calm_gap >= 150 else 0  # start of new season if gap >= 150 days

        storms.append(storm)
        prev_end_time = time[end_idx-1]

    return pd.DataFrame(storms)


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

