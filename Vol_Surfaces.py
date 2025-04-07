import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton

#Choose either 'Call' or 'Put'
def black_scholes(S, K, r, T, sigma, q, option_type):
    d1 = (np.log(S / K) + (r - q + (sigma**2 / 2)) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - (sigma**2 / 2)) * T) / (sigma * np.sqrt(T))

    if option_type == 'Call':
        #check norm.cfd (do not use only norm)
        call = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        return call
    if option_type == 'Put':
        put = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        return put
    else:
        print('Wrong option type: choose between "Call" and "Put"')




def implied_volatility(market_price, S, K, r, T, q, option_type):
    def objective_function(sigma):
        price = black_scholes(S, K, r, T, sigma, q, option_type)
        return price - market_price
    
    try:
        # Use newton raphson in a reasonable range for volatility
        return newton(objective_function, 0.10)
    except ValueError:
        return np.nan

print(implied_volatility(7.5, 50, 45, 0.02, 1, 0, 'Call'))

print(black_scholes(50, 45, 0.02, 1, 0.2, 0, 'Call'))
