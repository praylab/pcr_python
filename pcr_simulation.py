import numpy as np
import pandas as pd
from scripts.slr import calculate_slr
from scripts.storm import detect

# initialize simulation length and number of sim 
date_start = '2000-01-01'
date_end = '2100-12-31'
nr_simulation = 1000

# SLR scenario TODO: option to change projection (e.g., ar5 or ar6)
scenario = 'RCP85'
wl0 = 0.0 # initial water level
days_since_2018_start = (np.datetime64(date_start) - np.datetime64('2018-01-01')).astype(int)
slr_start = calculate_slr(days_since_2018_start, scenario)
wl0 += slr_start # initial water level including SLR at start date

# erosion constant initialization
doe  = 2.5
ws = 0.04 # settling velocity, 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
d = 2 + 1 # dune height + depth of closure
rec_rate = 29/365  # m/day, 29 m/year

# calibration parameters
c1 = 2.069 
c2 = 0.830 

# detect storm 
wave_data = pd.read_csv('data/wave_srilanka.csv', header=None)

# get hs, dir, tp, time from dataframe
time = wave_data.iloc[:, 0].values
hs = wave_data.iloc[:, 1].values
dir = wave_data.iloc[:, 2].values
tp = wave_data.iloc[:, 3].values
ts_hs = 95
ts_dur = 12.0

storms = detect(hs, dir, tp, time, ts_hs, ts_dur)

# fit storm gaps

# initialize batch simulation 

# while sim_count < nr_simulation 
	
	# sample storm in the size of nr_storm * batch size 
	# sample gap in the size of nr_storm * batch size 
	# sample year and storm season in the size of nr_year * batch size 
	
	# calculate erosion 
	
	# for sim 1:batch_size
		
		# counting sim_count
		# initialize simulation variable 
			
		# while t_days <= nr_days (in a simulation) 
		
			# calculate storm season end t_days + STS_sample (seasonCount)
			
			# while t_days < storm season end 
				
				# coastline retreat due to erosion 
				# advance t_days by duration of storm 
				# update year edge
				# update coastline minima
				
				# advance t_days by storm-to-storm gap 
				# calculate current water level
				# calculate retreat due to wl change 
				
				# if t_days < storm season end 
					# update coastline due to recovery and SLR 
					# update coastline minima 
				# else 
					# start of calm season is the day before this storm
					
				# advance storm index 
			
			# advance t_days to next "year"
			
			# calculate wl change
			# update coastline due to recovery and SLR 
			# minima update 
			
			# advance season count 
		
		# record the annual minima on this simulation 	