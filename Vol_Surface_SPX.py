import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


data = pd.read_excel("C:/Users/Mio/OneDrive/Code_files/Uni_Quant/Derivatives/Data/Option_chain_SPX_Test.xlsx")

S = 4992.97
q = 0
r = 0.043

#you also need to add time to maturity column
call = data[['Strike', 'Ask']]
put = data[['Puts Strike', 'Ask']]


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

#Implied vol algo. The idea is to make BS price and market price converge by adjusting sigma with Newton-Raphson
def implied_volatility(market_price, S, K, r, T, q, option_type):
    def objective_function(sigma):
        price = black_scholes(S, K, r, T, sigma, q, option_type)
        return price - market_price
    
    try:
        # Use Newton Raphson with an initial vol guess of 0.1
        return newton(objective_function, 0.10)
    except ValueError:
        return np.nan



