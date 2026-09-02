# helper scripts for geographical problems 

from pathlib import Path

import numpy as np
import duckdb
import geopandas as gpd
import shapely
import xarray as xr


def find_closest(lon:float, lat:float, data_lon:np.array, data_lat:np.array, dim:str=None) -> int: 
    '''
    Return to index of data that is the closest to lon and lat 
    '''

    dist = (data_lon - lon)**2 + (data_lat - lat)**2

    return dist.argmin(dim)


def extend_point(p1, p2, dist):
    p1, p2 = np.array(p1), np.array(p2)
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    return tuple(p2 + direction * dist)


def extend_line(geom: shapely.LineString, distance_m: float, ends="both"):
    coords = list(geom.coords)

    new_coords = coords.copy()
    if ends in ("start", "both"):
        new_coords[0] = extend_point(coords[1], coords[0], distance_m)
    if ends in ("end", "both"):
        new_coords[-1] = extend_point(coords[-2], coords[-1], distance_m)

    return shapely.LineString(new_coords)


def bbox_to_transect(lon_min, lat_min, lon_max, lat_max) -> gpd.GeoSeries:
    '''
    Return to GCTR transect (Calkoen and Luijendijk 2025) in the middle of the bbox which defined by lon_min, lat_min, lon_max, lat_min
    '''

    REPO_ROOT = Path(__file__).resolve().parent.parent  
    PARQUET_PATH = REPO_ROOT / 'data' / 'gctr' / '*.parquet'

    con = duckdb.connect()

    # slice parquet file within bounding box 
    df = con.sql(f"""
        SELECT transect_id, lon, lat, bearing, utm_epsg, country, ST_AsText(geometry) AS wkt
        FROM '{PARQUET_PATH}'
        WHERE lon BETWEEN {lon_min} AND {lon_max}
        AND lat BETWEEN {lat_min} AND {lat_max}
    """).df()

    # print(f'{len(df)} transects detected (~{len(df)/10} km coastline)')

    # add crs 
    gdf = gpd.GeoDataFrame(
        df.drop(columns='wkt'), 
        geometry=gpd.GeoSeries.from_wkt(df['wkt']), 
        crs='EPSG:4326'
    )

    transect = gdf.sort_values(by='transect_id').reset_index(drop=True)

    return transect.iloc[len(transect) // 2]


def nearest_era5arco(transect:gpd.GeoSeries, ds: xr.Dataset, idx: int = 0) -> list:
    '''
    Return to the closest point of ERA5 offshore to the transect 
    :param transect: pandas series of selected GCTR transect, output of bbox_to_transect
    :param cds_api_key: string of CDS API key 
    :param buffer: float of buffer around bbox 

    :return : list of coordinate [lon, lat] of the nearest offshore ERA5 point
    '''

    # convert to dataframe and remove nans (land points)
    df_era5 = ds.to_dataframe().reset_index().dropna()

    # add geographical information 
    original_crs = '4326'
    gdf_era5 = gpd.GeoDataFrame(
        df_era5, 
        geometry=[shapely.points([lon, lat]) for lon, lat in zip(df_era5.longitude, df_era5.latitude)], 
        crs=f'EPSG:{original_crs}'
    )

    # extend transect, do all operation in UTM projection 
    utm_crs = str(transect['utm_epsg'])
    
    transect_utm = gpd.GeoDataFrame([transect], crs=f'EPSG:{original_crs}').to_crs(f'EPSG:{utm_crs}')
    gdf_era5 = gdf_era5.to_crs(f'EPSG:{utm_crs}')

    # extend 55 km offshore (~ the equivalent km of the resolution)
    transect_utm['geometry'] = transect_utm['geometry'].apply(
        lambda geom: extend_line(geom, distance_m=55_000, ends='end')
    )

    # calculate distance between transect and ERA5 points 
    gdf_join = gpd.sjoin_nearest(gdf_era5, transect_utm, distance_col='dist_m').sort_values('dist_m')

    return [gdf_join.iloc[idx]['longitude'], gdf_join.iloc[idx]['latitude']]


def nearest_ar6slr(transect:gpd.GeoSeries, ds_slr:xr.Dataset, idx: int = 0) -> list: 
    '''
    return to the closest data point in AR6 IPCC data 
    '''

    # give geographical information and re-project to utm 
    ori_crs = '4326'
    utm_crs = str(transect['utm_epsg'])

    # to slr dataframe
    df_slr = ds_slr.to_dataframe().reset_index()
    gdf_slr = gpd.GeoDataFrame(
        df_slr[['locations', 'lat', 'lon', 'sea_level_change_rate']], 
        geometry=[shapely.Point(lon, lat) for lon, lat in zip(df_slr.lon, df_slr.lat)], 
        crs=f'EPSG:{ori_crs}'
    )
    gdf_slr = gdf_slr.to_crs(f'EPSG:{utm_crs}')

    # reproject transect 
    gdf_transect = gpd.GeoDataFrame([transect], crs=f'EPSG:{ori_crs}').to_crs(f'EPSG:{utm_crs}')

    # extend transect 110 km offshore (~AR6 resolution)
    gdf_transect['geometry'] = gdf_transect['geometry'].apply(
        lambda geom: extend_line(geom, distance_m=110_000, ends='end')
    )

    # calculate distance between transect and data 
    gdf_join = gpd.sjoin_nearest(gdf_slr, gdf_transect, distance_col='dist_m').sort_values('dist_m')

    return [gdf_join.iloc[idx]['lon_left'], gdf_join.iloc[idx]['lat_left']], gdf_join.iloc[idx]['locations']