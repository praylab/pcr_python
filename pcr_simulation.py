import os
import time

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from scripts import io, helper, storm, slr, shoreline, erosion

load_dotenv()

t0 = time.time()
print('Initialize the simulation ...')
# initialize simulation length and number of sim 
date_start = np.datetime64('2000-01-01T00:00:00')
date_end = np.datetime64('2100-12-31T23:00:00')
nr_simulation = 1000

print('Calculating Sea Level Rise ...')
# SLR scenario TODO: option to change projection (e.g., ar5 or ar6)
scenario = 'RCP85'
wl0 = 0.0 # initial water level
# days_since_2018_start = (date_start - np.datetime64('2018-01-01')).astype(int)
# slr_start = slr.calculate_slr(days_since_2018_start, scenario)
# wl0 += slr_start # initial water level including SLR at start date

print('Building Erosion Model ...')
# erosion constant initialization
doe  = 2.5
ws = 0.04 # settling velocity, 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
d = 2 + 1 # dune height + depth of closure
rec_rate = 7/365  # m/day, 29 m/year
m = 0.024

# calibration parameters
c1 = 1.339
c2 = 1.983 

# storm definition 
ts_hs = 95  # 95th percentile 
ts_dur = 12.0  # hours 
ts_between = 48.0  # hours

# import wave 
#TODO: from beach transect -> choose ERA5 point
lon = 82
lat = 7.5

print('Retrieve wave data ...')
# wave_data = io.era5_point(loc=[lon, lat], time_slice=slice('1979-01-01', '2019-12-31'), cds_api_key=os.getenv('CDS-API-KEY'))
import xarray as xr
wave_data = xr.open_dataset('./data/ERA5/B3_offshore.nc')
data_mapper = {'hs':'swh', 'dir':'mwd', 'tp':'mwp', 'time':'time'}

# get hs, dir, tp, time from datafram
hs, dir, tp, day = helper.era5_input(wave_data, data_mapper)

print('Detecting stroms ...')
# calculate storm properties
detected_storm, _ = storm.detect(hs, dir, tp, day, ts_hs, ts_dur, ts_between)
fitted_storm = storm.fit_storm(detected_storm)
fitted_lambdas = storm.fit_lambda_gap(detected_storm)
yearly_storm = np.ceil(detected_storm.shape[0] / (wave_data[data_mapper['time']][-1].dt.year - wave_data[data_mapper['time']][0].dt.year).values)

# initialize batch simulation 
nr_batch = 1000
t_days = (date_end - date_start).item().days
t_years = (date_end.astype('datetime64[Y]') - date_start.astype('datetime64[Y]')).astype(int)

# precompute once: maps each day offset from date_start to its month, used by
# gap_nhpp_thinning instead of converting every proposed day to a datetime
day_to_month = storm.build_day_to_month(date_start, t_days)

sim_count = 0
shoreline_stats = np.empty((t_years+1, nr_simulation))

print('Starting the simulation ...')
# while sim_count < nr_simulation 
while sim_count < nr_simulation: 
    # sample storm in the size of nr_storm * batch size 
	n_sample = yearly_storm * t_years * nr_batch 
	
	# sample storm in batch size 
	hss, durs, dirs, tps = storm.generate(
		fitted_storm=fitted_storm, 
		sampling_size=n_sample, 
		oversample=0.1, 
		max_dur=np.max(detected_storm.duration)
	)

	storm_count = 0

	# print progress 
	print(f'progress: {sim_count/nr_simulation * 100:.2f} %')
	# for sim 1:batch_size
	for i in range(nr_batch):  

		# simulate start of the storm 
		synth_start = storm.gap_nhpp_thinning(
			T=t_days,
			monthly_lambda=fitted_lambdas,
			date_start=date_start,
			duration=durs,
			start_storm=storm_count,
			day_to_month=day_to_month
		)

		storm_count_end = storm_count + len(synth_start)

		# get storm properties 
		synth_hs =  hss[storm_count:storm_count_end]
		synth_direction =  dirs[storm_count:storm_count_end]
		synth_duration = durs[storm_count:storm_count_end]
		synth_tp = tps[storm_count:storm_count_end]
		synth_end =  synth_start + (synth_duration / 24) 
		synth_gap = synth_start - np.roll(synth_end, 1)
		synth_gap[0] = 0.0

		storm_count = storm_count_end
		
		# calculate slr on every storm start 
		synth_slr = slr.vector_simulate_slr(
			day_start=synth_start, 
			date_start=date_start, 
			scenario=scenario, 
			wl0=wl0
		)

		# calculate erosion 
		_, synth_erosion = erosion.vector_mendoza(
			hss=synth_hs, 
			tps=synth_tp, 
			durs=synth_duration
		)

		# calculate the recovery 
		synth_recovery = shoreline.vector_calculate_recovery(
			gaps=synth_gap, 
			rec_rate=rec_rate
		)

		# calculate retreat due to slr 
		synth_retreat = shoreline.vector_calculate_slr_retreat(
			slrs=synth_slr,
			m=m
		)

		# track the shoreline evolution
		track_time, track_shoreline_change, track_shoreline_position = shoreline.vector_track_shoreline(
			day_start=synth_start, 
			day_end=synth_end, 
			recovery=synth_recovery,
			retreat=synth_retreat, 
			erosion=synth_erosion
		)

		# calculate the annual statistics 
		row = shoreline.vector_get_annual_statistics(
            track_time=track_time, 
            shoreline_position=track_shoreline_position, 
            kind='min', 
            date_start=date_start
		)

		try: 
			shoreline_stats[:, sim_count] = row.flatten()
		except: 
			sim_count -= 1

		sim_count += 1

print(f'running for {time.time()-t0:2f} s')