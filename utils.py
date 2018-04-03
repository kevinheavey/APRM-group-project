"""A collection of functions written for the APRM group project"""

import numpy as np

def lpm(returns_series, target_rate, order):
    """Computes the lower partial moment of a series of returns. 
    
    Note: this is already an unbiased estimator if our target_rate is not an unknown
    population parameter which we have estimated (e.g. the mean). 
    The sample raw moment is an unbiased estimator of the raw moment. 
    See http://mathworld.wolfram.com/SampleRawMoment.html
    For unbiased estimators of central moments, see http://mathworld.wolfram.com/SampleCentralMoment.html"""
    return (returns_series
            .sub(target_rate)
            .clip_upper(0)
            .pow(order)
            .mean())

def shortfall_risk(returns_series):
    return lpm(returns_series, 0, 0)

def expected_shortfall(returns_series):
    return -returns_series[returns_series < 0].mean()

def lpm_0_2(returns_series):
    return lpm(returns_series, target_rate=0, order=2)

def root_lpm_0_2(returns_series):
    return np.sqrt(lpm_0_2(returns_series))

def corrected_semivariance(returns_series):
    n = len(returns_series)
    correction_factor = n / (n-1)
    return lpm(returns_series, returns_series.mean(), 2) * correction_factor

