import numpy as np
import pytest
from slr import curve_ar5, calculate_slr  # replace 'your_module' with the actual filename (without .py)

def test_curve_ar5_valid_scenarios():
    expected = {
        'RCP85': [3.955e-10, 9.999e-06],
        'RCP60': [1.708e-10, 1.035e-05],
        'RCP45': [1.429e-10, 1.086e-05],
        'RCP26': [2.188e-12, 1.173e-05],
        '0': [0.0, 0.0],
    }
    for scenario, coeffs in expected.items():
        assert np.allclose(curve_ar5(scenario), coeffs), f"Failed for {scenario}"

def test_curve_ar5_invalid_scenario():
    with pytest.raises(TypeError):
        # This will raise because curve_ar5 returns None, and unpacking will fail in calculate_slr
        a, b = curve_ar5("INVALID")

def test_calculate_slr_scalar():
    # Test with a single integer
    days = 1000
    a, b = curve_ar5("RCP85")
    expected = a * days**2 + b * days
    result = calculate_slr(days, "RCP85")
    assert np.isclose(result, expected)

def test_calculate_slr_array():
    # Test with a numpy array
    days = np.array([0, 100, 200])
    scenario = "RCP45"
    a, b = curve_ar5(scenario)
    expected = a * days**2 + b * days
    result = calculate_slr(days, scenario)
    assert np.allclose(result, expected)

def test_calculate_slr_zero_scenario():
    # Should always return 0 for scenario '0'
    days = np.arange(0, 1000, 100)
    result = calculate_slr(days, '0')
    assert np.all(result == 0)

def test_calculate_slr_type_handling():
    # Accepts both int and float
    result_int = calculate_slr(500, 'RCP26')
    result_float = calculate_slr(500.0, 'RCP26')
    assert np.isclose(result_int, result_float)
