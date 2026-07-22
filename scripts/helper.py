from datetime import datetime, timedelta

import numpy as np
import scipy.io 
import pandas as pd
import xarray as xr

def datenum_to_datetime_one(o, f, e=366) -> datetime:
    '''
    Convert a single Matlab datenum into Python datetime
    :param o: ordinal part of datenum
    :param f: fractional part of datenum
    :param e: epoch difference between matlab and python (default 366 days)
    '''
    return datetime.fromordinal(int(o)) + timedelta(days=f) - timedelta(days=e)

def datenum_to_datetime(datenum: np.array) -> datetime:
    """
    Convert Matlab datenum into Python datetime.
    matlab's datenum is the number of days from non-existent year zero (January 1, 0000)
    python datetime is the number of seconds since the "UXIX epoch" (January 1, 1970)
    :param datenum: Date in datenum format, can be array.
    :return:        Datetime object corresponding to datenum.
    """
    datenum = np.asarray(datenum)

    ordinal = datenum.astype(int)
    frac = datenum - ordinal # fractional part of day 

    # for vectorized input 
    if datenum.ndim == 0:
        return datenum_to_datetime_one(ordinal, frac)
    
    return np.array([datenum_to_datetime_one(o, f) for o, f in zip(ordinal, frac)])

def datetime_to_datenum(py_dt) -> float: 
    '''
    retun a float that represent Matlab datenum. 
    :param py_dt: datetime.datetime object 
    '''
    # datenum include ordinal part of the datenum
    ordinal = datetime.toordinal(py_dt)

    # calculate the fraction of the day 
    frac = (py_dt - datetime(year=py_dt.year, month=py_dt.month, day=py_dt.day)) / timedelta(days=1) 

    # epoch difference 
    epoch = 366 

    return ordinal + frac + epoch

def datetime64_to_datenum(series_dt: pd.Series) -> np.array:
    '''
    Convert pd.Series of datetime64 to MATLAB datenum representation.
    :param series_dt: pd.Series of datetime64 
    '''

    return np.array([datetime_to_datenum(dt) for _, dt in series_dt.items()])

def load_mat_to_df(file_path: str, struct_name: str) -> pd.DataFrame:
    '''
    Load a MATLAB struct array from a .mat file into a Pandas DataFrame.
    :param file_path: str, path to the .mat file 
    :param struct_name: str, name of the struct in the .mat file
    :return: a DataFrame containing the struct data
    '''

    # load the .mat file 
    mat_data = scipy.io.loadmat(file_path)

    # check if struct_name exist
    if struct_name not in mat_data:
        raise ValueError(f"Struct '{struct_name}' not found in the .mat file")
    
    # Extract the struct data 
    struct_data = mat_data[struct_name]

    # Initialize an empty dictionary to store the struct data 
    struct_dict = {}

    # loop through each field in the struct and add it to the dictionary 
    for field in struct_data[0].dtype.names:
        struct_dict[field] = [entry[field][0][0] for entry in struct_data[0]]

    # Return to a DataFrame
    return pd.DataFrame(struct_dict)

def calculate_days_since_date(date: datetime, date_ref:datetime) -> int: 
    '''
    Calculate days since ref date 1st January of 2018 from a datetime
    :return : int of number of days, it could be negative denoting how many days before ref date
    '''
    return (date - date_ref).days

def calculate_days_since_2018(date: datetime) -> int: 
    '''
    Extension from calculate_days_since_date to calculate days since the AR5 SLR reference date
    (1st of January 2018)
    :return : int of number of days
    '''
    return (calculate_days_since_date(date, date_ref=datetime(2018,1,1)))

def date_add_days(date_ref: datetime, days: np.array) -> list:
    return [date_ref + timedelta(days=day) for day in days]

def interp_nan_array(y:np.array):
    '''
    Helper to fill NaN values with interpolation to the surrounding value. 
    :param y: array_like of numbers with NaNs
    :return y: array_like with NaNs replaced with interpolated value
    '''

    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]

    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y 


def era5_input(ds: xr.Dataset, keys: dict={'hs':'swh', 'dir':'mwd', 'tp':'mwp', 'time':'valid_time'}):
    """
    Extract time series from xarray dataset at point location 
    """
    # extract variables
    hs = ds[keys['hs']].values
    dir = ds[keys['dir']].values
    tp = ds[keys['tp']].values

    # extract time using helper to convert to matlab datenum
    time = datetime64_to_datenum(pd.Series(ds[keys['time']].values))
    return hs, dir, tp, time