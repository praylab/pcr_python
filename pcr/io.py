# script for input processing inputs 
import os 

import pandas as pd 
import xarray as xr 

def era5_point(loc:list, time_slice: slice, cds_api_key:str) -> pd.DataFrame: 
    '''
    loc: [lon, lat]
    '''

    # access data from CDS ARCO 
    lon = loc[0]
    lat = loc[1]

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

    # load in first location 
    ds = ds.sel(
        longitude=lon, 
        latitude=lat,
        method='nearest'
    ).sel(time=time_slice).compute()

    return ds 