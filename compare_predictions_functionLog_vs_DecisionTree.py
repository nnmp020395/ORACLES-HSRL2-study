'''
OBJECTIF 
----------------------------------------------

COMPARE THE DATASET PREDICTED BY

1. NON-LINEAR FUNCTIONS (LOG FUNCTION)
2. MACHINE LEARNING (DECISION TREE)

AND QUANTIFY THESE RESULTS 
----------------------------------------------
'''
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
GET CALIBRATED DATASET
WITH CONVERTED GPSTIME 

INPUT 
	- MAINDIR 
	- FILE NAME PATTERN 
'''


DATES = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']
### LOG FUNCTION
# f = lambda x : 0.20704*np.log(x) + 4.8181
### POLYNOMIAL FUNCTION
# f = lambda x : 4.6758 + (-3.6411/(1 + (x/0.0038)**1.0107))

#-------------------
SR355 = []
SR532 = []
BETAPART355 = []
SR532_PRED = []
TIME = []
ALTITUDE_2D = []
AOT_355s= []
#-------------------
SR532_TREE = []
SR355_TREE = []
SR532_PRED_TREE = []
TIME_TREE = []
ALTITUDE_2D_TREE = []
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
	X = (DATA['calibrated']/DATA['molecular']).sel(wavelength = 355).values.ravel()
	SR355.append((DATA['calibrated']/DATA['molecular']).sel(wavelength = 355).values.ravel())
	SR532.append((DATA['calibrated']/DATA['molecular']).sel(wavelength = 532).values.ravel())
	TIME.append(DATA['time'].values)
	ALTITUDE_2D.append(np.tile(DATA['altitude'].values/1000, (DATA['time'].shape[0], 1)).ravel()) ##(km)
	
	'''
	IMPORT PARTICULES BACKSCATTER FROM RAW DATA 
	'''
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/')
	PARTTERN = f'HSRL2_ER2_{DATE}_R8.h5'
	FILE = sorted(MAINDIR.glob(PARTTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE, group = 'DataProducts')
	BETAPART355 = (DATA['355_bsc']).values.ravel()
	RATIO_PRED = f(BETAPART355)
	SR532_PRED.append(RATIO_PRED * X)

	# AOT_355 = DATA['355_AOT_hi'].to_dataframe()
 #    AOT_355['flags'] = 0
 #    AOT_355.loc[AOT_355['355_AOT_hi'] > 0.1, 'flags'] = 1
 #    AOT_355.loc[(AOT_355['355_AOT_hi'] >= 0.1) & (AOT_355['355_AOT_hi'] < 0.5), 'flags'] = 2
 #    AOT_355.loc[(AOT_355['355_AOT_hi'] >= 0.5) & (AOT_355['355_AOT_hi'] < 1), 'flags'] = 3
 #    AOT_355.loc[AOT_355['355_AOT_hi'] >= 1, 'flags'] = 4
 #    AOT_355s.append(pd.concat([AOT_355['flags']]*ALT.shape[0], axis=1, ignore_index=True).values.ravel())
	
	'''
	IMPORT PREDICTIVE DATA BY DECISION TREE 
	'''
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products')
	PATTERN = f'tree_3f.sav-HSRL2-ER2-{DATE}_vNA*.nc'
	FILE = sorted(MAINDIR.glob(PATTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE)
	SR532_PRED_TREE.append(DATA['SR532_predicted'].values.ravel())
	SR532_TREE.append(DATA['SR532_measured'].values.ravel())
	SR355_TREE.append(DATA['SR355_measured'].values.ravel())
	TIME_TREE.append(pd.to_datetime(DATA['time'].values))
	ALTITUDE_2D_TREE.append(np.tile(DATA['altitude'].values/1000, (DATA['time'].shape[0], 1)).ravel()) ##(km)

MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products/')

SR532 = np.concatenate(SR532)
pd.DataFrame(SR532).to_pickle(Path(MAINDIR, 'TOTAL_SR532_ravel.pkl'))
SR532_PRED = np.concatenate(SR532_PRED)
pd.DataFrame(SR532_PRED).to_pickle(Path(MAINDIR, 'TOTAL_SR532_PRED_ravel_by_LOG_FUNCTION.pkl'))
SR532_TREE = np.concatenate(SR532_TREE)
pd.DataFrame(SR532_TREE).to_pickle(Path(MAINDIR, 'TOTAL_SR532_TREE_NA_ravel.pkl'))
SR532_PRED_TREE = np.concatenate(SR532_PRED_TREE)
pd.DataFrame(SR532_PRED_TREE).to_pickle(Path(MAINDIR, 'TOTAL_SR532_PRED_NA_ravel_by_DECISION_TREE.pkl'))
TIME = np.concatenate(TIME)
pd.DataFrame(TIME).to_pickle(Path(MAINDIR, 'TOTAL_TIME.pkl'))
TIME_TREE = np.concatenate(TIME_TREE)
pd.DataFrame(TIME_TREE).to_pickle(Path(MAINDIR, 'TOTAL_TIME_TREE.pkl'))

SR355 = np.concatenate(SR355)
pd.DataFrame(SR355).to_pickle(Path(MAINDIR, 'TOTAL_SR355_ravel.pkl'))
SR355_TREE = np.concatenate(SR355_TREE)
pd.DataFrame(SR355_TREE).to_pickle(Path(MAINDIR, 'TOTAL_SR355_TREE_NA_ravel.pkl'))

ALTITUDE_2D = np.concatenate(ALTITUDE_2D)
pd.DataFrame(ALTITUDE_2D).to_pickle(Path(MAINDIR, 'TOTAL_ALTITUDE_2D_ravel.pkl'))
ALTITUDE_2D_TREE = np.concatenate(ALTITUDE_2D_TREE)
pd.DataFrame(ALTITUDE_2D_TREE).to_pickle(Path(MAINDIR, 'TOTAL_ALTITUDE_2D_TREE_ravel_by_DECISION_TREE.pkl'))