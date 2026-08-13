import numpy as np

from dotenv import load_dotenv

from pcr import builder

load_dotenv()

print('Initialize the simulation ...')
# initialize simulation length and number of sim
year_start = '2020'
year_end = '2120'
bbox = [81.2366, 8.5625, 81.2500, 8.5763]

model = builder.build_model(
    year_start=year_start,
    year_end=year_end,
    bbox=bbox,
    nr_simulation=10000,
    nr_batch=1000,
    # SLR
    scenario='ssp585',
    wl0=0.0,
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
    # wave data
    source= 'file', 
    wave_data_path='data/ERA5/B3_offshore.nc',
    data_mapper={'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'},
)

shoreline_stats = model.run()
