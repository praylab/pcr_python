import numpy as np

def curve_ar5(scenario):
    '''
    AR5 SLR curve
    :param scenario: str, one of 'RCP26', 'RCP45', 'RCP60', 'RCP85', '0' (no SLR)
    :return: np.array, coeffients a and b for SLR = a*x^2+b*x where x is days since 2018-1-1
    '''
    if scenario == 'RCP85':
        return [3.955e-10, 9.999e-06]
    elif scenario == 'RCP60':
        return [1.708e-10, 1.035e-05]
    elif scenario == 'RCP45':
        return [1.429e-10, 1.086e-05]
    elif scenario == 'RCP26':
        return []