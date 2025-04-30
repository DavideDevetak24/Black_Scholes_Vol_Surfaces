import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk


#Choose either 'Call' or 'Put' !!!
#Remember that the vol surface plot works with real data only for one between put and call
#In theory the two IVs should be equal, but in practice they differ, so there's the need for two vol surfaces
def black_scholes(S, K, r, T, sigma, q, option_type):
    d1 = (np.log(S / K) + (r - q + (sigma**2 / 2)) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - (sigma**2 / 2)) * T) / (sigma * np.sqrt(T))

    if option_type == 'Call':
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

#Generating random data to double check the vol algo
#Used tuples just to visualize better. No need to flatten, use just real df on option chain data

market_prices = [[6.7, 7.1, 7.4], 
                 [7.3, 7.6, 7.9], 
                 [8.2, 8.4, 8.6]]
S = 50
K = [[45, 50, 55],
     [45, 50, 55],
     [45, 50, 55]]

r = 0.02
T = [[1,1,1],
     [2,2,2],
     [3,3,3]]

q = 0

option_type = 'Call'
flat_data = {
    'market_prices': np.array(market_prices).flatten(),
    'S': [S]*9,
    'K': np.array(K).flatten(),
    'r': [r]*9,
    'T': np.array(T).flatten(),
    'q': [q]*9,
    'option_type': [option_type]*9
}

data = pd.DataFrame(flat_data)

data['IV'] = data.apply(lambda row: implied_volatility(
    row['market_prices'], 
    row['S'], 
    row['K'], 
    row['r'], 
    row['T'], 
    row['q'], 
    row['option_type']), axis=1)

K_vals = data['K'].values
T_vals = data['T'].values
IV_vals = data['IV'].values

#Creation of a grid to interpolate values
K_grid_lin = np.linspace(min(K_vals), max(K_vals), 50)
T_grid_lin = np.linspace(min(T_vals), max(T_vals), 50)
K_grid, T_grid = np.meshgrid(K_grid_lin, T_grid_lin)

IV_grid = griddata(
    points=(K_vals, T_vals),
    values=IV_vals,
    xi=(K_grid, T_grid),
    method='linear'  #use linear or cubic (cubic is smoother)
)

#Turn the IV_grid into a continuous function
iv_interpolator = RegularGridInterpolator(
    (T_grid_lin, K_grid_lin),  #order T, then K
    IV_grid,
    bounds_error=True
)


def surface_query(K, T):
    try:
        IV = iv_interpolator([[T, K]])[0] #remember it retruns and array! So [0]
        BS_price = black_scholes(S, K, r, T, IV, q, option_type)
        return BS_price, IV
    except:
        return np.nan, np.nan


#cool interface
root = tk.Tk()
root.title("IV Surface Query")
root.geometry("1500x800")

main_frame = ttk.Frame(root)
main_frame.pack(fill='both', expand=True)

#left frame
left_frame = ttk.Frame(main_frame, padding=50)
left_frame.pack(side='left', fill='y')

ttk.Label(left_frame, text="Query IV Surface", font=('Arial', 18, 'bold')).pack(pady=20)

ttk.Label(left_frame, text="Strike (K):").pack(anchor='w')
strike_entry = ttk.Entry(left_frame)
strike_entry.pack(fill='x')

ttk.Label(left_frame, text="Time to Maturity (T):").pack(anchor='w')
time_entry = ttk.Entry(left_frame)
time_entry.pack(fill='x')

result_label = ttk.Label(left_frame, text="", font=('Arial', 9), foreground='Black')
result_label.pack(pady=20)

def query_surface():
    try:
        K = float(strike_entry.get())
        T = float(time_entry.get())
        BS_price, IV = surface_query(K, T)
        if np.isnan(IV):
            result_label.config(text="Outside data range/invalid input :-(")
        else:
            result_label.config(text=f"IV: {IV:.4f}\nBS Price: {BS_price:.4f}")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

ttk.Button(left_frame, text="Calculate IV and Price", command=query_surface).pack(pady=10)

#right frame
right_frame = ttk.Frame(main_frame)
right_frame.pack(side='right', fill='both', expand=True)

#cool chart
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(K_grid, T_grid, IV_grid, cmap='YlGnBu', edgecolor='none') #YlGnBu , coolwarm
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Implied Volatility')

ax.set_title('Volatility Surface')
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Time to Maturity (T)')
ax.set_zlabel('Implied Volatility (IV)')

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill='both', expand=True)

root.mainloop()



