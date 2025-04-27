"""
This script was used to create the images of different volatility surfaces,
in different simulated market coditions, for academic purposes.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

x = [
    [0.5, 0.75, 1.0, 1.25, 1.5],
    [0.5, 0.75, 1.0, 1.25, 1.5],
    [0.5, 0.75, 1.0, 1.25, 1.5],
    [0.5, 0.75, 1.0, 1.25, 1.5],
    [0.5, 0.75, 1.0, 1.25, 1.5]
    ]

y = [
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.5, 1.5, 1.5, 1.5, 1.5],
    [2.0, 2.0, 2.0, 2.0, 2.0],
    [2.5, 2.5, 2.5, 2.5, 2.5]
]

z_normal = [
    [0.21, 0.15, 0.13, 0.145, 0.17],
    [0.215, 0.155, 0.135, 0.15, 0.175],
    [0.21, 0.16, 0.14, 0.155, 0.18],
    [0.215, 0.165, 0.145, 0.16, 0.185],
    [0.22, 0.17, 0.15, 0.165, 0.19] 
]

z_inverted_T = [
    [0.36, 0.28, 0.23, 0.265, 0.32],
    [0.31, 0.24, 0.185, 0.225, 0.27],
    [0.26, 0.20, 0.145, 0.185, 0.22],
    [0.22, 0.17, 0.12, 0.155, 0.18],
    [0.18, 0.14, 0.10, 0.125, 0.14]
]

z_crash = [
    [0.60, 0.30, 0.23, 0.27, 0.35],
    [0.57, 0.27, 0.19, 0.23, 0.31],
    [0.53, 0.24, 0.16, 0.20, 0.28],
    [0.45, 0.20, 0.12, 0.16, 0.22],
    [0.30, 0.13, 0.10, 0.13, 0.18]
]


x = np.array(x).flatten()
y = np.array(y).flatten()
z_normal = np.array(z_normal).flatten()
z_inverted_T = np.array(z_inverted_T).flatten()
z_crash = np.array(z_crash).flatten()


"""
By changing the variable z it is possible to change the IV condition.
It is possible to choose between z_normal, z_inverted_T, z_crash.
The surface plot changes consequently.

"""
#z_normal, z_inverted_T, z_crash
z = z_normal

grid = np.column_stack((x.flatten(), y.flatten()))
values = z.flatten()

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata(grid, values, (xi, yi), method='cubic')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_zlim(0, 0.6)
ax.set_xlabel('K/S')
ax.set_ylabel('T')
ax.set_zlabel('')
plt.title('IV Surface')
plt.show()