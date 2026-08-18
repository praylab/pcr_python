# script for input processing inputs
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pcr import geo, helper

# repo root, anchored to this file's location so paths resolve regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent

# calibration lookup tables (see pcr.calibration); data/ is gitignored, so these
# persist locally without being committed
LOOKUP_DIR = REPO_ROOT / 'data' / 'lookup_tables'
REC_RATE_LOOKUP_PATH = LOOKUP_DIR / 'rec_rate.csv'
REC_RATE_LOOKUP_COLUMNS = ['lon_wave', 'lat_wave', 'rec_rate', 'objective', 'calibrated_at']

def lazyload_era5arco(cds_api_key:str) -> xr.Dataset:
    '''
    Lazy load ERA5 ARCO from CDS dataset (read more about dataset: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=analysis_ready_data)
    '''
    # lazy load the ERA5 data
    wave_geo_url = "https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr"

    # open the zarr object with xarray 
    ds = xr.open_zarr(
        wave_geo_url, 
        consolidated=True, 
        storage_options={
            'headers': {'Authorization': f'Bearer {cds_api_key}'}
        }
    )

    return ds 

def lazyload_ar6slr(scenario:str) -> xr.Dataset:

    DIR = REPO_ROOT / 'data' / 'AR6_slr'

    scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

    if scenario not in scenarios:
        raise ValueError(f'Not valid scenario. Available scenario: {scenarios}')

    return xr.open_dataset(os.path.join(DIR, f'total_{scenario}_medium_confidence_rates.nc'))

    

def era5arco_point(loc:list, time_slice: slice, cds_api_key:str, ds=None) -> pd.DataFrame: 
    '''
    loc: [lon, lat]
    '''

    # open the zarr object with xarray 
    if ds is None: 
            ds = lazyload_era5arco(cds_api_key)

    # load in first location 
    ds = ds.sel(
        longitude=loc[0], 
        latitude=loc[1],
        method='nearest'
    ).sel(time=time_slice).compute()

    return ds 


def era5arco_area(loc:list, cds_api_key:str, buffer:float=1, time_slice=None, ds=None) -> pd.DataFrame:
    '''
    Return to slice of ERA5 within buffer area from with loc as origin point 
    :param loc: [lon, lat]
    '''
    # lazy load ERA5 
    if ds is None: 
        ds = lazyload_era5arco(cds_api_key)

    # buffer area, make 1 degree buffer area from the center of the transect 
    slice_lon = slice(loc[0] - buffer, loc[0] + buffer)
    slice_lat = slice(loc[1] - buffer, loc[1] + buffer)

    # clip within buffered bbox and select one timestamp 
    if time_slice:
        ds = ds.sel(longitude=slice_lon, latitude=slice_lat, time=time_slice).compute()
    else: 
        ds = ds.sel(longitude=slice_lon, latitude=slice_lat).isel(time=0).compute()

    return ds 


def ar6slr_point(loc:float, scenario:str) -> xr.Dataset:
    '''
    Return to subset of Xarray Dataset in a single location pointed by 'locations' key of AR6 Sea Level dataset
    '''

    # load AR6 SL Change data 
    ds = lazyload_ar6slr(scenario)

    return ds.sel(locations=loc)


def ar6slr_area(loc:list, scenario:str, buffer: float=2, year_slice=None) -> xr.Dataset:
    '''
    Return to slice of AR6 SLR dataset within buffer area from with loc as origin point 
    :param loc: [lon, lat]
    '''
    # load AR6 SLR data 
    ds = lazyload_ar6slr(scenario)

    # create mask of lon and lat within buffer area
    mask = (
        (ds.lon.values >= (loc[0]-buffer)) & (ds.lon.values <= (loc[0]+buffer)) &
        (ds.lat.values >= (loc[1]-buffer)) & (ds.lat.values <= (loc[1]+buffer))
    )

    valid_idx = np.where(mask)[0]

    if year_slice: 
        ds_subset = ds.isel(locations=valid_idx).sel(years=year_slice, quantiles=0.5)
    else: 
        ds_subset = ds.isel(locations=valid_idx, years=0).sel(quantiles=0.5)

    return ds_subset


def import_ar6_curve(transect, scenario: str, quantile:float=0.5, date_start:np.datetime64=None, buffer:float=2) -> list:
    '''
    import IPCC AR6 sea level change rate curve, return to list of rates and years if date_start is not provided else to days since date_start
    '''

    # import ds within buffer 
    ds_slr = ar6slr_area(loc=[transect.lon, transect.lat], scenario=scenario, buffer=buffer)

    # find the closest point 
    _, idx_location = geo.nearest_ar6slr(transect, ds_slr)
    point = ar6slr_point(loc=idx_location, scenario=scenario)

    # extract the curve 
    rates = point.sel(quantiles=quantile).sea_level_change_rate.values / 1000  # in meter / year
    years = point.years.values

    # convert if applicable
    if date_start:
        year_start = (years - 1970).astype('datetime64[Y]')
        days = (year_start - date_start).astype('timedelta64[D]').astype(int)

        return rates/365.25, days
    else: 
        return rates, years


def import_era5arco(transect, cds_api_key:str, buffer:float=1):
    '''
    return to 40 years historical wave data from the closest offshore ERA5 ARCO data point
    '''

    # get ERA5 area within buffer area 
    ds = era5arco_area([transect.lon, transect.lat], cds_api_key, buffer)

    # get nearest point 
    lon, lat = geo.nearest_era5arco(transect, ds)

    # load 40 years data from 1980 - 2019
    ds = era5arco_point(
        loc=[lon, lat], 
        time_slice=slice('1980-01-01T00:00:00.000000000', '2020-01-01T00:00:00.000000000'), 
        cds_api_key=cds_api_key
    )
    year_record = 40 # HARD CODED 

    # unpack the data array 
    data_mapper = {'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'}
    hs, dir, tp, time = unpack_era5(ds, data_mapper)

    return hs, dir, tp, time, year_record, lon, lat


def load_wave_data(wave_data_path: str, data_mapper: dict = None):
    '''
    Load a wave time series file and extract hs, dir, tp, day arrays plus the
    number of years the record spans (used by PCRModel to scale yearly storm counts).
    '''
    data_mapper = data_mapper or {'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'}

    wave_data = xr.open_dataset(wave_data_path)
    hs, dir, tp, day = unpack_era5(wave_data, data_mapper)

    time_var = data_mapper['time']
    record_years = (wave_data[time_var][-1].dt.year - wave_data[time_var][0].dt.year).values

    return hs, dir, tp, day, record_years


def load_rec_rate_lookup(path=REC_RATE_LOOKUP_PATH) -> pd.DataFrame:
    '''
    Load the rec_rate calibration lookup table written by save_rec_rate_lookup().
    Returns an empty table with the expected columns if it hasn't been created yet.
    '''
    if not os.path.exists(path):
        return pd.DataFrame(columns=REC_RATE_LOOKUP_COLUMNS)
    return pd.read_csv(path)


def find_rec_rate(lon_wave: float, lat_wave: float, path=REC_RATE_LOOKUP_PATH, tol: float = 1e-6):
    '''
    Look up a previously calibrated rec_rate for the wave point closest to
    (lon_wave, lat_wave). Returns None if no entry is within `tol` degrees, i.e.
    nothing has been calibrated for this point yet (see save_rec_rate_lookup()).
    '''
    table = load_rec_rate_lookup(path)
    if table.empty:
        return None

    dist = np.hypot(table['lon_wave'] - lon_wave, table['lat_wave'] - lat_wave)
    idx = dist.idxmin()
    if dist.loc[idx] > tol:
        return None

    return float(table.loc[idx, 'rec_rate'])


def save_rec_rate_lookup(lon_wave: float, lat_wave: float, rec_rate: float, objective: float = None, path=REC_RATE_LOOKUP_PATH, tol: float = 1e-6) -> pd.DataFrame:
    '''
    Record a calibrated rec_rate for (lon_wave, lat_wave), upserting: any existing
    entry within `tol` degrees of this point is replaced rather than duplicated.
    '''
    table = load_rec_rate_lookup(path)

    if not table.empty:
        dist = np.hypot(table['lon_wave'] - lon_wave, table['lat_wave'] - lat_wave)
        table = table[dist > tol]

    new_row = pd.DataFrame([{
        'lon_wave': lon_wave,
        'lat_wave': lat_wave,
        'rec_rate': rec_rate,
        'objective': objective,
        'calibrated_at': pd.Timestamp.now().isoformat(),
    }])
    table = new_row if table.empty else pd.concat([table, new_row], ignore_index=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    return table


def unpack_era5(ds: xr.Dataset, keys: dict={'hs':'swh', 'dir':'mwd', 'tp':'mwp', 'time':'valid_time'}):
    """
    Extract time series from xarray dataset at point location 
    """
    # extract variables
    hs = ds[keys['hs']].values
    dir = ds[keys['dir']].values
    tp = ds[keys['tp']].values

    # extract time using helper to convert to matlab datenum
    time = helper.datetime64_to_datenum(pd.Series(ds[keys['time']].values))
    return hs, dir, tp, time