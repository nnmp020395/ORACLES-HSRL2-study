import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import tqdm
import warnings
warnings.filterwarnings("ignore")

'''
Ce script est pour plotter les profiles calibres du ORACLES-ER2
Author: _Phuong Nguyen_
'''


# READ FILES

ER2_DIR = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/')
CALIB_LISTFILES= sorted(ER2_DIR.glob('*20160823_R8_v2.h5'))
OUTPUT_PATH = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/profiles/')

for file in tqdm.tqdm(CALIB_LISTFILES):
    print(file)
    dt_raw = xr.open_dataset(file)
    print(dt_raw['time'])
    dt = dt_raw.dropna(dim='time', how='all')
    print(dt['time'])
    calibrated_height = dt_raw.attrs['calibrated_height'].values#[17700, 19000]
    for t in range(len(dt['time'])):
        # Initialize plot 
        #----------------
        f, (ax, ax2) = plt.subplots(figsize=[12,7], ncols=2, sharey=True)
        
        # Calculate SR from dataset built by RF_ER2_test2.py
        # Plot profil 355
        #---------------------------------------------------
        dt.isel(time = t, wavelength=1)['calibrated'].plot(y='altitude', label = f'{dt["wavelength"][1].values}:calibrated', ax =ax, xlim=(0, 0.005), color='g')
        dt.isel(time = t, wavelength=1)['molecular'].plot(y='altitude', label = f'{dt["wavelength"][1].values}:molecular', ax=ax, xlim=(0, 0.005), linestyle='--', color='g')
        (dt['calibrated']/dt['molecular']).isel(time = t, wavelength=1).plot(y='altitude', ax=ax2, xlim=(0,5), label = f'{dt["wavelength"][1].values}:sr', color='g')
        ax.axhspan(calibrated_height[0], calibrated_height[1], color='y', alpha=.2, label='calibration height')
        
        # Plot profil 532
        #---------------------------------------------------
        dt.isel(time = t, wavelength=0)['calibrated'].plot(y='altitude', label = f'{dt["wavelength"][0].values}:calibrated', ax =ax, xlim=(0, 0.005), color='b')
        dt.isel(time = t, wavelength=0)['molecular'].plot(y='altitude', label = f'{dt["wavelength"][0].values}:molecular', ax=ax, xlim=(0, 0.005), linestyle='--', color='b')
        (dt['calibrated']/dt['molecular']).isel(time = t, wavelength=0).plot(y='altitude', ax=ax2, xlim=(0,5), label = f'{dt["wavelength"][0].values}:sr', color='b')
        
        # Add line SR=1 and plot configuration 
        #-------------------------------------
        ax2.axvline(1, color='k', label='sr=1')
        ax.legend(loc='best')
        ax2.legend(loc='best')
        ax.set(title=f'file={file.name}\ntime={np.around(dt.time[t].values, decimals=3)} (h)', xlabel='backscatter (km-1.sr-1)')
        ax2.set(title=f'file={file.name}\ntime={np.around(dt.time[t].values, decimals=3)} (h)', xlabel='scattering ratio')
        plt.rcParams['font.size'] = '14'
        
        # Save plot
        #----------
        plt.savefig(Path(OUTPUT_PATH, f'{file.stem}_w532w355_{np.around(dt.time[t].values, decimals=3)}_atb_sr.png'))
        plt.close()
        plt.clf()
