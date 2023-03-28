import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import matplotlib as mpl
import warnings 
warnings.filterwarnings("ignore")
import os, sys
# os.chdir('/homedata/nmpnguyen/ORACLES-ER2/Codes')
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/Codes')
from outils import convert_gpstime
# os.chdir('/homedata/nmpnguyen/comparaison/Codes')
sys.path.append('/homedata/nmpnguyen/comparaison/Codes')
from fonctions import dataset, conditions, get_file, check, conversion, plots

'''
OBJECTIF 
----------------------------------------------

TO COLLECT ALL CONVERTED DATASET FOR COMPARING PROCESSUS
AND QUANTIFY THESE RESULTS 
----------------------------------------------
'''

DATES = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']

#-------------------
SR355 = []
SR532 = []
BETAPART355 = []
SR532_PRED = []
TIME = []
ALTITUDE_2D = []
#-------------------

for DATE in DATES: 
	'''
	IMPORT CALIBRATED DATA
	'''
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/')
	PARTTERN = f'HSRL2_ER2_{DATE}*_v2.nc'
	FILE = sorted(MAINDIR.glob(PARTTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE)
	time = DATA['time'].values
	altitude = DATA['altitude'].values/1000
	'''
	IMPORT CONVERTED DATA BY ARTEM'S SIGNAL CONVERSION METHOD
	'''
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products')
	PARTTERN = f'HSRL2_ER2_{DATE}_R8_convert_Artem_func.nc'
	FILE = sorted(MAINDIR.glob(PARTTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE)
	print(DATA['alt'])
	SR532_PRED.append(DATA['SR532_PRED'].sel(time=time).values.ravel())
	TIME.append(DATA['time'].sel(time=time).values)
	ALTITUDE_2D.append(np.tile(DATA['alt'].values/1000, (time.shape[0], 1))) ##(km)

MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products/')


SR532_PRED = np.concatenate(SR532_PRED)
pd.DataFrame(SR532_PRED).to_pickle(Path(MAINDIR, 'TOTAL_SR532_CONVERT_by_ARTEM_FUNCTION.pkl'))
# ALTITUDE_2D = np.concatenate(ALTITUDE_2D)
# pd.DataFrame(ALTITUDE_2D).to_pickle(Path(MAINDIR, 'TOTAL_ALTITUDE_2D_ravel.pkl'))