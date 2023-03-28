import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

'''
Convert gps time to ymd-hms time 
--------------------------------
'''
def convert_gpstime(gpstime, date, convert=False):
    def frmt(decimal_time): # You can rewrite it with 'while' if you wish
        hours = int(decimal_time)#
        minutes = np.round(decimal_time - hours, 4)*60
        seconds = np.round(minutes - int(minutes), 4)*60
        HMS_time = f'{hours}:{int(minutes)}:{int(seconds)}'#"%s:%s:%f"%(hours, int(minutes), int(seconds))
        return HMS_time
    if convert==True:
        list_HMS_time = list(map(frmt, gpstime))
        list_YMD_HMS = list(map(lambda orig_string: date+' '+orig_string, list_HMS_time))
        pd_YMD_HMS = pd.to_datetime(list_YMD_HMS).strftime('%Y-%m-%d %H:%M:%S')
    else:
        list_gpstime_str = list(map(lambda n: '%.3f'%n, gpstime))
        list_YMD_HMS = list(map(lambda orig_string: date+' '+orig_string, list_gpstime_str))
        pd_YMD_HMS = list_YMD_HMS
    return pd_YMD_HMS


'''
Function of Definition of the parameters of histogram
-----------------------------------------------------
'''
def get_params_histogram(srlimite, X532, Y355):
    from scipy import stats
    if len(X532[~np.isnan(X532)&~np.isinf(X532)]) < len(Y355[~np.isnan(Y355)&~np.isinf(Y355)]):
        mask = [~np.isnan(Y355)&~np.isinf(Y355)]
        print('A')
    elif len(X532[~np.isnan(X532)&~np.isinf(X532)]) > len(Y355[~np.isnan(Y355)&~np.isinf(Y355)]):
        mask = [~np.isnan(X532)&~np.isinf(X532)]
        print('B')
    else:
        H = np.histogram2d(X532[~np.isnan(X532)], Y355[~np.isnan(Y355)], bins=100, range = [[-10, srlimite], [-10, srlimite]]) 
        mask = [~np.isnan(X532)&~np.isinf(X532)]
        print('C')
        
    H = np.histogram2d(X532[mask], Y355[mask], bins=100, range = srlimite)
    Hprobas = H[0]*100/len(Y355[mask])
    noNaNpoints = len(Y355[mask])
#   define slope and intercept of fit line
#     from scipy.optimize import curve_fit
#     # objective function for best fit
#     def objective(x, a, b):
#         return a * x + b
#     param, param_cov = curve_fit(objective, X532[mask], Y355[mask])
#     print(param, param_cov)

    print(f'nombre de points no-NaN: {noNaNpoints}')
    xedges, yedges = np.meshgrid(H[1], H[2])
#     print(slope, intercept)
#     fitLine = slope * allsr532 + intercept
    return xedges, yedges, Hprobas, noNaNpoints


'''
Input list of ER2 files 
-----------------------
'''
ER2_PATH = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/')
ER2_LISTFILES = sorted(ER2_PATH.glob('*_R8.h5'))


'''
Define SR355/SR532 dataset 
--------------------------------------
'''
allsr355 = None
allsr532 = None 

z_seuil = 0
for filepath in ER2_LISTFILES:
    print(f'FILE: {filepath}')
    data = xr.open_dataset(filepath)
#     print(data['altitude'])
    convert_timelist = convert_gpstime(data.time.values, filepath.stem.split('_')[2], convert=False)
    data = data.assign_coords(time = convert_timelist)
    limitez = (data['altitude'].values>z_seuil)
    sr = data['calibrated']/data['molecular']
    print(f'Resolution vertical: {sr.altitude.values[3]-sr.altitude.values[2]}')
    # print(f'Resolution temporelle: {sr.time.values[3]-sr.time.values[2]}')
    if (allsr532 is None) | (allsr355 is None) : 
        allsr355 = sr.sel(wavelength=355).isel(altitude=limitez)
        allsr532 = sr.sel(wavelength=532).isel(altitude=limitez)
    else:
        allsr355 = xr.concat([allsr355, sr.sel(wavelength=355).isel(altitude=limitez)], dim='time')
        allsr532 = xr.concat([allsr532, sr.sel(wavelength=532).isel(altitude=limitez)], dim='time')

# allsr355.to_netcdf(Path(ER2_PATH, 'HSRL2_ER2_allsr355_v3.nc'), 'w')
# allsr532.to_netcdf(Path(ER2_PATH, 'HSRL2_ER2_allsr532_v3.nc'), 'w')



'''
Histogram SR355/SR532
---------------------
'''
print(f'nombre total des profils: {len(allsr532.time.values)}')
print(f'nombre total des fichiers: {len(ER2_LISTFILES)}')

from matplotlib.colors import LogNorm
from scipy import stats

Xxedges, Yyedges, Hprobas, nanpoints = get_params_histogram([[0,40],[0,80]], allsr355.values.ravel(), allsr532.values.ravel())
ff, ax = plt.subplots(figsize=[6,10], nrows=1)
p = ax.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm())
# ax.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
c = plt.colorbar(p, ax=ax, label='%')
ax.set(ylabel='SR532', xlabel='SR355', 
       title= f'ORACLES-ER2, \n{nanpoints} points, {len(allsr532.time)} profiles\n{data.attrs}')#\nLinearRegression: {round(slope,5)}x + {round(intercept,3)}
ax.set(xlim=(-10,40), ylim=(-10,80))
plt.minorticks_on()
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
plt.savefig(Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/quicklookSR/',f'distributionSR_ER2_over{z_seuil}.png'))