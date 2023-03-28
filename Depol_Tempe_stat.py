import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tqdm

'''
Le script sert à quantifier la dépolarization et la température en fonction de l'ATB et SR de ER2.
Script est modifiable.
'''

# Get main directory and list files
#----------------------------------
ER2_FOLDER = Path('/homedata/nmpnguyen/ORACLES-ER2/Raws')
ER2_LISTFILES = sorted(ER2_FOLDER.glob('*R8.h5'))

# Initialize stocking variables
#------------------------------
all_dep = None #Depol variable
all_TT = None #Temperature variable
all_total_attn_bsc = None #ATB variable
wave = 532 #choose the wavelength

for file in ER2_LISTFILES:
    DataProducts = xr.open_dataset(file, group='DataProducts')
    Nav_Data = xr.open_dataset(file, group='Nav_Data')
    State = xr.open_dataset(file, group='State')

    er2alt = DataProducts['Altitude'][:]
    time = Nav_Data['gps_time'][:]

    TT = State['Temperature'] 
    TP = State['Pressure'] 
    ND = State['Number_Density']
    # er2attn532 = DataProducts["532_total_attn_bsc"].values
    # er2attn355 = DataProducts["355_total_attn_bsc"].values

    # Concat Depol & Temperature values from each file 
    #-------------------------------------------------
    if (all_dep is None) | (all_TT is None):
        all_dep = DataProducts[f'{wave}_dep'].values.ravel()
        # all_TT = TT.values.ravel()
        all_total_attn_bsc = DataProducts[f"{wave}_total_attn_bsc"].values.ravel()
    else:
        all_dep = np.concatenate([all_dep, DataProducts[f'{wave}_dep'].values.ravel()])
        # all_TT = np.concatenate([all_TT, TT.values.ravel()])
        all_total_attn_bsc = np.concatenate([all_total_attn_bsc, DataProducts[f"{wave}_total_attn_bsc"].values.ravel()])


# Plot - distribution scatterplot
#--------------------------------

# set altitude label 
a = [round(x,2) for x in er2alt[0,:].values[np.linspace(0,er2alt.shape[1]-1,9).astype(int)]]

Y = all_dep
X = all_total_attn_bsc
rangeY = [0.0, 1.0]
rangeX = [0.0, 1.0] 
H = np.histogram2d(X[~np.isnan(X)&~np.isnan(Y)], Y[~np.isnan(X)&~np.isnan(Y)], bins=100, range=[rangeX, rangeY])
HProbas = H[0]*100/len(X[~np.isnan(X)&~np.isnan(Y)])
Xxedges, Yyedges = np.meshgrid(H[1], H[2])

from matplotlib.colors import LogNorm
fig, ax = plt.subplots()
p = ax.pcolormesh(Xxedges, Yyedges, HProbas.T, norm=LogNorm(vmax=1e-1, vmin=1e-5))
c = plt.colorbar(p, ax=ax, label='%')
ax.set(xlabel='total attn bsc', ylabel='Total depolarization ratio', 
       title= f'ER2, {wave}nm')#\nLinearRegression: {round(slope,5)}x + {round(intercept,3)}
# ax.set(xlim=(-10,sr_limite), ylim=(-10,sr_limite))
# ax.grid(True)
plt.savefig(Path('/homedata/nmpnguyen/ORACLES-ER2/Figs/Depol/',f'distribution_Depol_TotalAttnBsc_{wave}.png'))
