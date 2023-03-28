import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import os, sys
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/Codes/outils.py')
from outils import find_nearest as find_nearest_index

'''
Get directory of RAW DATA & CALIBRATED DATA
'''
ER2_DIR = Path('/homedata/nmpnguyen/ORACLES-ER2/Raws/')
ER2CALIB_DIR = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/')
LISTFILENAMES = sorted(ER2CALIB_DIR.glob('*_R8.h5'))

def get_extinction_integrated(time, altitude, CloudTopHeight, extinction, sr):
    ## CALCULER ALL PROFILS D'EXTINCTION INTÉGRÉE ET SES VALEURS SR532/SR355 CORRESPONDANTES
    '''
    Intégrer le signal avec limitre entre 7km et CTH-Cloud Top Height 
    ! chaque profil 1 CTH ou Nan 
    '''

    id_cth = [find_nearest_index(altitude, cth) for cth in CloudTopHeight]
    id_limite = find_nearest_index(altitude, 7000)

    # quand CTH = 0 
    ls_ext_integ = np.zeros((len(time)))#np.zeros((np.isnan(cloud_top_height.flatten()).sum()))
    for z in range(1, id_limite):
        tmp = extinction[np.isnan(CloudTopHeight.flatten()), z]*1e-3*(altitude[z] - altitude[z-1])
        ls_ext_integ[np.isnan(CloudTopHeight.flatten())] = np.nansum([ls_ext_integ[np.isnan(CloudTopHeight.flatten())], tmp], axis=0)

    # quand CTH != 0 
    for i in np.where(~np.isnan(CloudTopHeight.flatten()))[0]:
        for z in range(id_cth[i], id_limite):
            tmp = extinction[i, z]*1e-3*(altitude[z] - altitude[z-1])
            ls_ext_integ[i] = np.nansum([ls_ext_integ[i], tmp], axis=0)
             
    ls_tau_integ = np.exp(-2*ls_ext_integ) 
    ls_ext_integ[np.isnan(extinction).all(axis=1)] = np.nan

    sr_equivalent = np.array([sr[t, find_nearest_index(altitude, CloudTopHeight[t])] for t in range(CloudTopHeight.shape[0])])
    sr_equivalent[np.isnan(CloudTopHeight.flatten())] = np.nan    
    return ls_ext_integ, sr_equivalent


append_ls_ext532integ = [] # pour le scatterplot
append_ls_ext355integ = [] # pour le scatterplot
append_rapport_sr = [] # pour le scatterplot
append_ls_sr532equivalent = [] # pour le scatterplot

# for lf in LISTFILENAMES:
    # er2path = Path(ER2CALIB_DIR, lf)
for er2path in LISTFILENAMES:
    data = xr.open_dataset(er2path)
    sr355 = (data['calibrated']/data['molecular']).sel(wavelength=355)
    sr532 = (data['calibrated']/data['molecular']).sel(wavelength=532)
    atb355 = data['calibrated'].sel(wavelength=355)
    atb532 = data['calibrated'].sel(wavelength=532)
    DataProducts = xr.open_dataset(Path(ER2_DIR, er2path.name), group='DataProducts')
    cloud_top_height = DataProducts['cloud_top_height'].values.flatten()
    ext532 = DataProducts['532_ext'].values
    ext355 = DataProducts['355_ext'].values
    time = data.time.values
    altitude = data.altitude.values

    ls_ext532integ, sr532_equivalent = get_extinction_integrated(time, altitude, cloud_top_height, ext532, sr532)
    print(f'Extinction 532 integree: {ls_ext532integ}')
    print(f'SR equivalent: {sr532_equivalent}')
    ls_ext355integ, sr355_equivalent = get_extinction_integrated(time, altitude, cloud_top_height, ext355, sr355)
    print(f'Extinction 355 integree: {ls_ext355integ}')
    print(f'SR equivalent: {sr355_equivalent}')

    # '''
    # 532 nm 
    # '''
    # fix, (ax, ax1) = plt.subplots(ncols=2, figsize=(11,5))
    # plt.rcParams['font.size']=12
    # # Set the axes labels font size
    # ax.scatter(data.time.values, ls_ext532integ, color='r')
    # # ax.plot(data.time.values, DataProducts['355_AOT_above_cloud'].values, color='g')
    # ax.set(xlabel= 'time', ylabel= 'ext532 (red)', title=f'{er2path.stem}')
    # ax.set_ylim(-0.4, ls_ext532integ.max())
    # ax2 = ax.twinx()
    # ax2.scatter(data.time.values, sr532_equivalent/sr355_equivalent, color='k')
    # ax2.set(ylabel= 'sr532/sr355 (black)')
    # ax2.set_ylim(0,12)
    # '''
    # 355 nm 
    # '''
    # ax1.scatter(data.time.values, ls_ext355integ, color='r')
    # # ax.plot(data.time.values, DataProducts['355_AOT_above_cloud'].values, color='g')
    # ax1.set(xlabel= 'time', ylabel= 'ext355 (red)', title=f'{er2path.stem}')
    # ax1.set_ylim(-0.4, ls_ext355integ.max())
    # ax22 = ax1.twinx()
    # ax22.scatter(data.time.values, sr532_equivalent/sr355_equivalent, color='k')
    # ax22.set(ylabel= 'sr532/sr355 (black)')
    # ax22.set_ylim(0,12)
    # plt.tight_layout()
    # plt.savefig(f'/homedata/nmpnguyen/ORACLES-ER2/Figs/extinction_integree_{lf.split(".")[0]}.png')
    '''
    append 
    '''
    append_ls_ext532integ.append(ls_ext532integ[~np.isnan(sr532_equivalent/sr355_equivalent)&~np.isnan(ls_ext532integ)])
    # append_ls_ext355integ.append(ls_ext355integ[~np.isnan(sr532_equivalent/sr355_equivalent)&~np.isnan(ls_ext532integ)])
    append_rapport_sr.append((sr532_equivalent/sr355_equivalent)[~np.isnan(sr532_equivalent/sr355_equivalent)&~np.isnan(ls_ext532integ)])
    append_ls_sr532equivalent.append(sr532_equivalent[~np.isnan(sr532_equivalent/sr355_equivalent)&~np.isnan(ls_ext532integ)])


append_ls_ext532integ = np.concatenate(append_ls_ext532integ)
# append_ls_ext355integ = np.concatenate(append_ls_ext355integ)
append_rapport_sr = np.concatenate(append_rapport_sr)
append_ls_sr532equivalent = np.concatenate(append_ls_sr532equivalent)

from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from scipy import stats 



count_binned, count_binned_x, count_binned_y, _ = stats.binned_statistic_2d(x=append_rapport_sr, y=append_ls_ext532integ, 
                                        values=append_ls_sr532equivalent, statistic='count', 
                                        bins=[50, 50], range=[[0,20],[0,append_ls_ext532integ.max()]])
mean_binned, mean_binned_x, mean_binned_y, _ = stats.binned_statistic_2d(x=append_rapport_sr, y=append_ls_ext532integ, 
                                        values=append_ls_sr532equivalent, statistic='mean', 
                                        bins=[50, 50], range=[[0,20],[0,append_ls_ext532integ.max()]])
std_binned, std_binned_x, std_binned_y, _ = stats.binned_statistic_2d(x=append_rapport_sr, y=append_ls_ext532integ, 
                                        values=append_ls_sr532equivalent, statistic='std', 
                                        bins=[50, 50], range=[[0,20],[0,append_ls_ext532integ.max()]])
std_binned[np.isnan(mean_binned)] = np.nan
count_binned[np.isnan(mean_binned)] = np.nan
corre_value = pearsonr(append_rapport_sr, append_ls_ext532integ)[0]

print(count_binned[~np.isnan(count_binned)].sum())
fig, (ax, ax1, ax2) = plt.subplots(ncols = 3, figsize=(15,5))
ax.set(xlabel= 'sr532/sr355', ylabel= 'ext532', title = f'corrélation: {round(corre_value,4)}')
ax1.set(xlabel= 'sr532/sr355', title = f'Total: {len(append_ls_sr532equivalent)} points')
ax2.set(xlabel= 'sr532/sr355')
cmap2= plt.cm.Spectral_r #mcolors.ListedColormap(colors_colorbar)
cmap2.set_under('lightgrey')
cmap2.set_over("dimgrey")

# norm= mcolors.Normalize(vmin=0,vmax=100)
# pcm = ax.hist2d(append_rapport_sr, append_ls_ext532integ,
#           bins=50, range=[[0,10],[0, append_ls_ext532integ.max()]], shading='auto', norm=norm, cmap='turbo')
pcm = ax.pcolormesh(count_binned_x, count_binned_y, count_binned.T, cmap=cmap2, norm=LogNorm(vmin=1))
cb = plt.colorbar(pcm, ax=ax, label='counts', extend='both')
ax.set_ylim(0, append_ls_ext532integ.max())
ax.set_xlim(0, 20)
cb.ax.minorticks_on()

pcm1 = ax1.pcolormesh(mean_binned_x, mean_binned_y, mean_binned.T, cmap=cmap2, vmin=1,vmax=300)
cb1 = plt.colorbar(pcm1, ax=ax1, extend='both', label='mean binned SR532')
ax1.set_xlim(0, 20)
ax1.set_ylim(0, append_ls_ext532integ.max())
cb1.ax.minorticks_on()

norm2= mcolors.Normalize(vmin=1e-2,vmax=100)
pcm2 = ax2.pcolormesh(std_binned_x, std_binned_y, std_binned.T, cmap=cmap2, norm=norm2)
cb2 = plt.colorbar(pcm2, ax=ax2, extend='both', label='std binned SR532')
ax2.set_xlim(0, 20)
ax2.set_ylim(0, append_ls_ext532integ.max())
cb2.ax.minorticks_on()

plt.tight_layout()
plt.savefig(Path("/homedata/nmpnguyen/ORACLES-ER2/Figs/",f'extinction_integree_scatterplot_all_v2.png'))