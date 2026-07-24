import numpy as np

from dotenv import load_dotenv

from pcr.model import PCRModel

load_dotenv()

print('Initialize the simulation ...')
# initialize simulation length and number of sim
date_start = np.datetime64('2000-01-01T00:00:00')
date_end = np.datetime64('2100-12-31T23:00:00')

model = PCRModel(
    date_start=date_start,
    date_end=date_end,
    nr_simulation=1000,
    nr_batch=100,
    # SLR 
    scenario='0',
    wl0=0.0,
    lon_sl=82,
    lat_sl=7.5,
    # erosion, m/day, 29 m/year for rec_rate
    doe=2.5,
    ws=0.04,  # settling velocity, 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
    d=2 + 1,  # dune height + depth of closure
    rec_rate=7 / 365,
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
    wave_data_path='./data/ERA5/B3_offshore.nc',
    data_mapper={'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'},
)

shoreline_stats = model.run()