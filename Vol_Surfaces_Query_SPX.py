import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton, brentq
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk


#option chains data from April 17th 2025
data = pd.read_excel("Data/Option_Data_17_Apr_2025.xlsx")

S = 5335.75
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

    #in this case preferable brentq as the initial estimate could be very far away from
    # the real IV 
    try:
        return brentq(
            objective_function, 
            1e-4,
            #for the option used sometimes vol for low strikes reaches 20.0, 
            # brentq capped at 3.0
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

#Creation of a grid to interpolate values
K_grid_lin = np.linspace(min(K_vals_call), max(K_vals_call), 1000)
T_grid_lin = np.linspace(min(T_vals_call), max(T_vals_call), 1000)
K_grid, T_grid = np.meshgrid(K_grid_lin, T_grid_lin)

IV_grid = griddata(
    points=(K_vals_call, T_vals_call),
    values=IV_vals_call,
    xi=(K_grid, T_grid),
    method='linear'  #use linear or cubic (cubic is smoother, but for real data sometimes makes a mess)
)

IV_vals_put = put_chains['IV'].values
K_vals_put = put_chains['K'].values
T_vals_put = put_chains['T'].values

IV_grid_put = griddata(
    points = (K_vals_put, T_vals_put), 
    values = IV_vals_put, 
    xi = (K_grid, T_grid), 
    method='linear'
)

#Turn the IV_grid into a continuous function
iv_interpolator_call = RegularGridInterpolator(
    (T_grid_lin, K_grid_lin),  # order T, then K
    IV_grid,
    bounds_error=True
)

iv_interpolator_put = RegularGridInterpolator(
    (T_grid_lin, K_grid_lin),  # order T, then K
    IV_grid_put,
    bounds_error=True
)

def surface_query(K, T, option_type='Call'):
    try:
        if option_type == 'Call':
            IV = iv_interpolator_call([[T, K]])[0]
        elif option_type == 'Put':
            IV = iv_interpolator_put([[T, K]])[0]
        else:
            raise ValueError("Dude, what?")
        
        BS_price = black_scholes(S, K, r, T, IV, q, option_type)
        return BS_price, IV
    except Exception as e:
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

ttk.Label(left_frame, text="Query Option Type:").pack(anchor='w')
query_option_type_var = tk.StringVar(value="Call")
query_option_type_menu = ttk.Combobox(left_frame, textvariable=query_option_type_var, state="readonly")
query_option_type_menu['values'] = ("Call", "Put")
query_option_type_menu.pack(fill='x', pady=5)

def query_surface():
    try:
        K = float(strike_entry.get())
        T = float(time_entry.get())
        selected = query_option_type_var.get()

        if selected == "Put":
            BS_price, IV = surface_query(K, T, option_type="Put")
        elif selected == "Call":
            BS_price, IV = surface_query(K, T, option_type="Call")
        else:
            result_label.config(text="Bruh")
            return

        if np.isnan(IV):
            result_label.config(text="Outside data range/invalid input :-(")
        else:
            result_label.config(text=f"{selected} IV: {IV:.4f}\nBS Price: {BS_price:.4f}")

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

ttk.Button(left_frame, text="Calculate IV and Price", command=query_surface).pack(pady=10)

result_label = ttk.Label(left_frame, text="", font=('Arial', 9), foreground='Black')
result_label.pack(pady=20)

#right frame
right_frame = ttk.Frame(main_frame)
right_frame.pack(side='right', fill='both', expand=True)

#dropdown for chart IV surface
ttk.Label(left_frame, text="Option Type:").pack(anchor='w')
option_type_var = tk.StringVar(value="Call")
option_type_menu = ttk.Combobox(left_frame, textvariable=option_type_var, state="readonly")
option_type_menu['values'] = ("Call", "Put", "Both")
option_type_menu.pack(fill='x', pady=5)

#to update the chart
def update_surface_plot():
    ax.clear()

    selected = option_type_var.get()

    if selected == "Call" or selected == "Both":
        ax.plot_surface(K_grid, T_grid, IV_grid, cmap='viridis', edgecolor='none', alpha=0.8, label="Call")

    if selected == "Put" or selected == "Both":
        ax.plot_surface(K_grid, T_grid, IV_grid_put, cmap='YlOrRd', edgecolor='none', alpha=0.8, label="Put")

    ax.set_title('Volatility Surface')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (T)')
    ax.set_zlabel('Implied Volatility (IV)')

    canvas.draw()

ttk.Button(left_frame, text="Update Chart", command=update_surface_plot).pack(pady=10)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_grid, T_grid, IV_grid, cmap='viridis', edgecolor='none') #YlGnBu
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





