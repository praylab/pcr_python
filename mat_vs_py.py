# this script compare the exceedance probability from calculation in Matlab and Python 

import mat73
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from datetime import datetime
from scripts import erosion, shoreline

# import storm
data_dict = mat73.loadmat('data/test/diagnostic_storm.mat')

# column of the storm track start day, hs, duration, tp
storm_track = data_dict['storm_track']
filtered_storm = storm_track[~np.isnan(storm_track).any(axis=1)] # remove nan values 

# calculate end days and gap 
day_start = filtered_storm[:, 0] 
end_days = day_start + (filtered_storm[:, 2] / 24)

prev_day_end = np.concatenate([[filtered_storm[0, 0]], end_days[:-1]]) # leaving the first storm has 0 gap

# calculate gap and remove gap with lower than 0
gap_days =  day_start - prev_day_end 
gap_days[gap_days < 0] = 0

# store data as dataframe
synthetic_matlab = pd.DataFrame({
    'hs': filtered_storm[:, 1],
    'duration': filtered_storm[:, 2],
    'tp': filtered_storm[:, 3],
    'day_start': day_start,
    'day_end': end_days,
    'gap': gap_days
}).astype(np.float32)

# store the index where the storm start 
start_idx = np.where(day_start==0)
start_idx = start_idx[0]
end_idx = start_idx[1:]

# set date start and initiation
date_start = datetime(2000,1,1)
nr_simulation = 100000
mins_stat = np.empty((101, nr_simulation))
sim_count = 0
nr_batch = 1000
sim_start = 0

while sim_count < nr_simulation:
    sim_until = sim_start + nr_batch

    if sim_until > end_idx.shape[0]:
        sim_until = end_idx.shape[0] - 1

    # get a slice of one simulation 
    slice_ts = synthetic_matlab.iloc[start_idx[sim_start]:end_idx[sim_until]].copy()

    slice_ts['slr_retreat'] = 0

    # calculate storm-induced erosion
    _, slice_ts['erosion_storm'] = erosion.mendoza(slice_ts)

    slice_ts['recovery'] = shoreline.calculate_recovery(
        storms=slice_ts,
        rec_rate=7/365
    )

    for i in range(sim_start, sim_until):
        temp_df = slice_ts.loc[start_idx[i]:end_idx[i]].copy()
        temp_track = shoreline.track_shoreline(temp_df)

        temp_row = shoreline.get_annual_statistics(temp_track, kind='min', date_start=date_start)
        # QUICK-FIX: force to have only 101 shape -> check later 
        # if temp_row.shape[0] >= 101:
        #     temp_row = temp_row[:101]
        # else:
        #     n_nan = 101-temp_row.shape[0]
        #     temp_row = np.concat([temp_row, np.full((n_nan, 1), np.nan)])

        if temp_row.shape[0] < 101:
            sim_count -= 1
        else:
            temp_row = temp_row[:101]
            mins_stat[:, sim_count] = temp_row.flatten()

        sim_count += 1

        if sim_count == nr_simulation:
            break

    sim_start += nr_batch

# import output from matlab 
data_dict = mat73.loadmat('data/test/diagnostic_mins.mat')
mins_stat_mat = data_dict['mins_allSims']

mins_stat_py = mins_stat

# Define the years of interest
years = [25, 50, 75, 100]

# Calculate exceedance (percentile) for each year
def get_exceedance(data, year):
    # Extract the column for the specific year
    year_data = data[year, :]  # Adjusting for zero-based index
    # Calculate the exceedance for each year (sorted data)
    sorted_data = np.sort(year_data)
    exceedance = (1 - np.arange(len(sorted_data)) / len(sorted_data)) * 100
    return sorted_data, exceedance

# Prepare the plot
plt.figure(figsize=(10, 6))

# colors
linestyles = [':', '--', '-.', '-']

# recession 
recession_min_py = -mins_stat_py
recession_min_mat = -mins_stat_mat

# Plot for stats_df (solid line)
for i, year in enumerate(years):
    sorted_data, exceedance = get_exceedance(recession_min_py, year)
    plt.plot(sorted_data, exceedance, label=f'Year {year} - Python', linestyle=linestyles[i], color='b')

# # Plot for stats_mat_df (dashed line)
for i, year in enumerate(years):
    sorted_data, exceedance = get_exceedance(recession_min_mat, year)
    plt.plot(sorted_data, exceedance, label=f'Year {year} - Matlab', linestyle=linestyles[i], color='r')

# Labels and title
plt.xlabel('Value')
plt.ylabel('Exceedance Probability (%)')
plt.title('Exceedance Plot of shoreline change for Different Years')
plt.legend()
plt.yscale('log')

# plt.xlim([-100, 100])
# plt.ylim([0.1, 100])

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
