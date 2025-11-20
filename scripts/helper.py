from datetime import datetime, timedelta

import numpy as np

def datenum_to_datetime_one(o, f, e=366) -> datetime:
    '''
    Comvert a single Matlab datenum into Python datetime
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
