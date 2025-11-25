from datetime import datetime, timedelta

import numpy as np
import scipy.io 
import pandas as pd

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
