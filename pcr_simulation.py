import numpy as np

from dotenv import load_dotenv

from pcr.model import PCRModel

load_dotenv()

print('Initialize the simulation ...')
# initialize simulation length and number of sim
year_start = '2020'
year_end = '2120'

model = PCRModel(
    year_start=year_start,
    year_end=year_end,
    nr_simulation=100000,
    nr_batch=1000,
    # SLR 
    scenario='ssp585',
    slr_data_path='data/AR6_slr',
    wl0=0.0,
    lon_sl=82,
    lat_sl=7.5,
    # erosion, m/day, 29 m/year for rec_rate
    doe=2.5,
    ws=0.04,  # settling velocity, 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
    d=2 + 1,  # dune height + depth of closure
    # rec_rate=7 / 365,
    m=0.024,
    c1=1.339,
    c2=1.983,
    # storm definition
    ts_hs=95,  # 95th percentile
    ts_dur=12.0,  # hours
    ts_between=48.0,  # hours
    # wave data, TODO: from beach transect -> choose ERA5 point
    lon=82,
    lat=7.5,
    wave_data_path='data/ERA5/B3_offshore.nc',
    data_mapper={'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'},
)

shoreline_stats = model.run()