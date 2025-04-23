import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


data = pd.read_excel("C:/Users/Mio/OneDrive/Code_files/Uni_Quant/Derivatives/Option_Data/Options_Finished_File/Option_Data.xlsx")
S = 4992.97
q = 0
r = 0.043

#you also need to add time to maturity column
call_chains = data[['Strike', 'Ask', 'Time']]
call_chains['S'] = S
call_chains['q'] = q
call_chains['r'] = r
call_chains['option_type'] = 'Call'
call_chains = call_chains.rename(columns={
    'Strike': 'K',
    'Ask': 'market_prices',
    'Time': 'T'
})
put_chains = data[['Put Strike', 'Put Ask', 'Time']]
put_chains['S'] = S
put_chains['q'] = q
put_chains['r'] = r
put_chains['option_type'] = 'Put'
put_chains = put_chains.rename(columns={
    'Strike': 'K',
    'Ask': 'market_prices',
    'Time': 'T'
})


#Choose either 'Call' or 'Put'
def black_scholes(S, K, r, T, sigma, q, option_type='Call'):
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

call_chains['IV'] = data.apply(lambda row: implied_volatility(
    row['market_prices'], 
    row['S'], 
    row['K'], 
    row['r'], 
    row['T'], 
    row['q'], 
    row['option_type']), axis=1)

put_chains['IV'] = data.apply(lambda row: implied_volatility(
    row['market_prices'], 
    row['S'], 
    row['K'], 
    row['r'], 
    row['T'], 
    row['q'], 
    row['option_type']), axis=1)

print(call_chains.head())


