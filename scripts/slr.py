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