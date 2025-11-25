import numpy as np
from datetime import datetime

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
        return [2.188e-12, 1.173e-05]
    elif scenario == '0':
        return [0.0, 0.0]
    
def calculate_slr(days_since_2018, scenario, projection='AR5'):
    '''
    Calculate sea level rise based on days since 2018-1-1 and scenario
    :param days_since_2018: int or np.array, number of days since 2018-1-1
    :param scenario: str, one of 'RCP26', 'RCP45', 'RCP60', 'RCP85', '0' (no SLR)
    :return: float or np.array, sea level rise in meters
    '''
    if projection != 'AR5':
        raise NotImplementedError("Only AR5 projection is implemented.")
    
    a, b = curve_ar5(scenario)
    slr = a * (days_since_2018 ** 2) + b * days_since_2018
    return slr

def simulate_slr(synthetic_storm: pd.DataFrame, date_start, scenario, wl0) -> pd.Series:
    # calculate days since the refer date (1st of January 2018)
    if scenario == '0':
        return pd.Series(np.zeros(len(synthetic_storm)))

    days = (date_start - datetime(2018, 1, 1)).days + synthetic_storm['day_start']

    slr_values = calculate_slr(days, scenario)

    # return to series of water level change where on the first day equal to wl0
    return pd.Series(wl0 - slr_values[0] + slr_values)
    

# TODO: ar6 projection

# test out the function
if __name__ == "__main__": # this only runs when this script is executed directly
    import plotly.express as px
    
    days = np.arange(0, 365*100, 365)  # every year for 10 years
    scenario = 'RCP85'
    slr_values = calculate_slr(days, scenario)

    # plotting the results
    fig = px.line(
        x=days/365, 
        y=slr_values*1000, 
        labels={'x': 'Years since 2018', 'y': 'Sea Level Rise (mm)'},
        title=f'Sea Level Rise under {scenario} Scenario'
    )

    fig.show()