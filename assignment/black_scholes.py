"""
Black-Scholes Option Pricing Module
This module calculates option prices using the Black-Scholes formula.
"""

from enum import Enum
import numpy as np
from scipy.stats import norm

class CallPut(str, Enum):
    CALL = 'call'
    PUT = 'put'  

class InvalidPriceParametersException(Exception):
    def __init__(self, message="Invalid option pricing parameters."):
        super().__init__(message)

def _black_d2(forward:float, strike:float, time_to_expiry:float, volatility:float) -> float:
    """Calculating d2 using d1"""
    d1 = _black_d1(forward, strike, time_to_expiry, volatility)
    return d1 - volatility * np.sqrt(time_to_expiry)

def _black_d1(forward:float, strike:float, time_to_expiry:float, volatility:float) -> float:
    """Calculating d1 with Black Scholes solution"""
    return (np.log(forward / strike) + (0.5 * volatility**2 * time_to_expiry)) / (volatility * np.sqrt(time_to_expiry))

def _black_76(forward:float, strike:float, time_to_expiry:float, volatility:float, interest_rate:float, call_put:CallPut) ->  float:
    d1 = _black_d1(forward, strike, time_to_expiry, volatility)
    d2 = _black_d2(forward, strike, time_to_expiry, volatility)
    discount_factor = np.exp(-interest_rate * time_to_expiry)

    if call_put == CallPut.CALL:
        return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    if call_put == CallPut.PUT:  
        return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

def black_scholes_price(forward:float, strike:float, time_to_expiry:float, volatility:float, interest_rate:float, call_put:CallPut) -> float:
    """Calculates option price for the desired option.
    Edge cases including 0>= time-to-expiry, 0>= strike, 0>= volatility and 0>=forward are all handled."""
    if strike == 0: 
        if call_put == CallPut.PUT:  
            return 0  
        if call_put == CallPut.CALL:
            return forward

    if time_to_expiry <= 0:
        if call_put == CallPut.PUT:  
            return max(0, strike - forward)  
        else:
            return max(0, forward - strike)  

    if (strike < 0 or forward <= 0 or volatility <= 0):
        raise InvalidPriceParametersException("Invalid values for strike, forward or volatility parameters")

    return _black_76(forward, strike, time_to_expiry, volatility, interest_rate, call_put)

