import pandas as pd
import numpy as np

# TODO: docstring
def mendoza(storms_sample: pd.DataFrame) -> list:
    '''
    calculate erosion adapting mendoza and jimenez (2006) erosion model

    '''

    # constants 
    Doe = 2.5 
    g = 9.81
    ws = 0.04 # 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
    m = 12/500 # average beach profile slope -> from where? 
    d = 6

    C1 = 2.069 #1.883;
    C2 = 0.830 #0.842; 

    hs = storms_sample['hs'].values
    tp = storms_sample['tp'].values 
    dur = storms_sample['duration'].values 

    D = hs / (ws * tp)
    JA = m * np.abs(Doe-D)**0.5

    delV = C1*JA*dur + C2
    delX = delV / d

    return [delV, delX]

def vector_mendoza(hss: np.array, tps: np.array, durs: np.array) -> list:
    '''
    calculate erosion adapting mendoza and jimenez (2006) erosion model

    '''

    # constants 
    Doe = 2.5 
    g = 9.81
    ws = 0.04 # 0.03 for 0.2 mm; 0.05 for 0.3 mm; 0.07 for 0.4 mm; 0.09 for 0.5 mm
    m = 12/500 # average beach profile slope -> from where? 
    d = 6

    C1 = 2.069 #1.883;
    C2 = 0.830 #0.842; 

    D = hss / (ws * tps)
    JA = m * np.abs(Doe-D)**0.5

    delV = C1*JA*durs + C2
    delX = delV / d

    return [delV, delX]