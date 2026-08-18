# Factory for constructing PCRModel instances from raw config (file paths, coordinates).
# Owns the I/O that PCRModel itself no longer does: PCRModel only ever holds
# already-loaded arrays (see attach_slr()/attach_wave_data()).
import numpy as np

from pcr import io, geo
from pcr.model import PCRModel


def build_slr_curve(scenario: str, transect: float, date_start: np.datetime64):
    '''
    Import the AR6 sea level rate curve at (lon_sl, lat_sl).
    :return: (rate_ar6, days_ar6), or a zero curve when scenario == '0' (no SLR)
    '''
    if scenario == '0':
        return np.zeros(10), np.zeros(10)

    return io.import_ar6_curve(transect, scenario, date_start=date_start)


def build_wave_array(source: str, transect: float = None, cds_api_key: str = None, wave_data_path: str = None, data_mapper: dict = None):
    '''
    Load a wave time series plus the (lon_wave, lat_wave) point it came from.
    :return: (hs, dir, tp, day, record_years, lon_wave, lat_wave). lon_wave/lat_wave
        are None for source='file', since a local file has no implied ERA5 grid point.
    '''
    # Return to ARCO file in the closest offshore to the transect
    if source == 'ARCO':
        return io.import_era5arco(transect, cds_api_key)
    elif source == 'file':
        hs, dir, tp, day, record_years = io.load_wave_data(wave_data_path, data_mapper)
        return hs, dir, tp, day, record_years, None, None
    else:
        raise ValueError('Not valid source. Choose between "ARCO" or "file"')


def build_transect(bbox):
    return geo.bbox_to_transect(*bbox)


def build_model(
    # SLR scenario
    scenario: str = 'ssp126',
    # BBOX
    bbox: list = None,
    # Wave data access 
    source: str = "ARCO", 
    cds_api_key: str = None, 
    wave_data_path: str = None,
    data_mapper: dict = None, 
    **model_kwargs,
) -> PCRModel:
    '''
    Build a ready-to-run PCRModel: resolve the AR6 SLR curve and wave data from
    paths/coordinates, then attach them to a freshly constructed model.
    Call model.run() (or detect_storms() + run_simulation()) afterwards.

    :param model_kwargs: forwarded to PCRModel(...) (year_start, year_end, wl0,
        erosion/storm params, ...); do not include `scenario`, pass it here instead.
    '''
    print('Initialize model ..')
    model = PCRModel(scenario=scenario, **model_kwargs)

    # build transect 
    print('Building transect model ..')
    transect = build_transect(bbox)
    model.attach_transect(transect)

    # build slr curve
    # TODO: check datestart, ambil dari model.datestart??
    print('Building SLR ..')
    rate_ar6, days_ar6 = build_slr_curve(scenario=scenario, transect=transect, date_start=model.date_start)
    model.attach_slr(rate_ar6, days_ar6, scenario=scenario)

    print('Building wave ..')
    hs, dir, tp, day, record_years, lon_wave, lat_wave = build_wave_array(source, transect=transect, cds_api_key=cds_api_key, wave_data_path=wave_data_path, data_mapper=data_mapper)
    model.attach_wave_data(hs, dir, tp, day, record_years, lon_wave=lon_wave, lat_wave=lat_wave)

    return model
