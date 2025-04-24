import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton, brentq
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata



#option chains data from April 17th 2025
data = pd.read_excel("C:/Users/Mio/OneDrive/Code_files/Uni_Quant/Derivatives/Option_Data/Options_Finished_File/Option_Data_Test.xlsx")

S = 5275.70
q = 0.01387
r = 0.0396

call_chains = data[['Strike', 'Ask', 'Time']].copy()
call_chains['S'] = S
call_chains['q'] = q
call_chains['r'] = r
call_chains['option_type'] = 'Call'
call_chains = call_chains.rename(columns={
    'Strike': 'K',
    'Ask': 'market_prices',
    'Time': 'T'
})
put_chains = data[['Put Strike', 'Put Ask', 'Time']].copy()
put_chains['S'] = S
put_chains['q'] = q
put_chains['r'] = r
put_chains['option_type'] = 'Put'
put_chains = put_chains.rename(columns={
    'Put Strike': 'K',
    'Put Ask': 'market_prices',
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

#Implied vol algo. The idea is to make BS price and market price converge by adjusting sigma with brentq/Newton-Raphson
def implied_volatility(market_price, S, K, r, T, q, option_type):
    def objective_function(sigma):
        price = black_scholes(S, K, r, T, sigma, q, option_type)
        return price - market_price

    try:
        return brentq(
            objective_function, 
            1e-4, 
            3.0, 
            xtol=1e-6)
    except ValueError:        
        try:
            return newton(
                objective_function,
                0.1,
                tol=1e-6,
                maxiter=50)
        except (RuntimeError, OverflowError):
            return np.nan


call_chains['IV'] = call_chains.apply(lambda row: implied_volatility(
    row['market_prices'], 
    row['S'], 
    row['K'], 
    row['r'], 
    row['T'], 
    row['q'], 
    row['option_type']), axis=1)

put_chains['IV'] = put_chains.apply(lambda row: implied_volatility(
    row['market_prices'], 
    row['S'], 
    row['K'], 
    row['r'], 
    row['T'], 
    row['q'], 
    row['option_type']), axis=1)


call_chains = call_chains.dropna()
put_chains = put_chains.dropna()

K_vals_call = call_chains['K'].values
T_vals_call = call_chains['T'].values
IV_vals_call = call_chains['IV'].values

IV_vals_put = put_chains['IV'].values
K_vals_put = put_chains['K'].values
T_vals_put = put_chains['T'].values

x_call = K_vals_call
y_call = T_vals_call
z_call = IV_vals_call

x_put = K_vals_put
y_put = T_vals_put
z_put = IV_vals_put


def chart(x, y, z):
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = griddata((x, y), z, (X, Y), method='linear')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='YlGnBu')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

chart(x_call, y_call, z_call)
chart(x_put, y_put, z_put)