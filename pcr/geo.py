# helper scripts for geographical problems 
import numpy as np

def find_closest(lon:float, lat:float, data_lon:np.array, data_lat:np.array, dim:str=None) -> int: 
    '''
    Return to index of data that is the closest to lon and lat 
    '''

    dist = (data_lon - lon)**2 + (data_lat - lat)**2

    return dist.argmin(dim)