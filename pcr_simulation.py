# initialize simulation length and number of sim 

# calculate SLR -> module, can be copot pasang 

# erosion constant initialization

# detect storm 

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