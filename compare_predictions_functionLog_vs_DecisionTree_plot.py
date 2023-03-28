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
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings 
warnings.filterwarnings("ignore")
import sys
sys.path.append('/homedata/nmpnguyen/comparaison/Codes')

from fonctions import dataset, conditions, get_file, check, conversion, plots

'''
GET CALIBRATED DATASET
WITH CONVERTED GPSTIME 

INPUT 
	- MAINDIR 
	- FILE NAME PATTERN 
'''

MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products/')
SR532 = pd.read_pickle(Path(MAINDIR, 'SR532_ravel.pkl')).values.ravel()
SR532_PRED = pd.read_pickle(Path(MAINDIR, 'SR532_PRED_ravel_by_LOG_FUNCTION.pkl')).values.ravel()
SR532_TREE = pd.read_pickle(Path(MAINDIR, 'SR532_TREE_ravel.pkl')).values.ravel()
SR532_PRED_TREE = pd.read_pickle(Path(MAINDIR, 'SR532_PRED_ravel_by_DECISION_TREE.pkl')).values.ravel()



# QUANTFY 
print('--QUANTIFY--')
UNIT_VALUE = np.arange(0.05, 2, 0.2)
MIN_MAX_VALUES = [0.0, 80.0]
PTS_STATS = []
PTS_STATS_TREE = []

for U in UNIT_VALUE:
	PTS = check(MIN_MAX_VALUES[0], MIN_MAX_VALUES[0], MIN_MAX_VALUES[1], MIN_MAX_VALUES[1], U, SR532, SR532_PRED)
	PTS_STATS.append(PTS.quantify())
	PTS = check(MIN_MAX_VALUES[0], MIN_MAX_VALUES[0], MIN_MAX_VALUES[1], MIN_MAX_VALUES[1], U, SR532_TREE, SR532_PRED_TREE)
	PTS_STATS_TREE.append(PTS.quantify())

PTS_STATS = pd.DataFrame(PTS_STATS, index=UNIT_VALUE)
PTS_STATS.to_pickle(Path(MAINDIR, f'LogFunction-HSRL2-ER2-DataTest_Stats_between_{MIN_MAX_VALUES[0]}_{MIN_MAX_VALUES[1]}.pkl'))
print(f'Log Function - Stats between SR532 and SR532_PRED : \n{PTS_STATS}')
PTS_STATS_TREE = pd.DataFrame(PTS_STATS_TREE, index=UNIT_VALUE)
PTS_STATS_TREE.to_pickle(Path(MAINDIR, f'tree_3f-HSRL2-ER2-DataTest_Stats_between_{MIN_MAX_VALUES[0]}_{MIN_MAX_VALUES[1]}.pkl'))
print(f'Decision Tree - Stats between SR532 and SR532_PRED : \n{PTS_STATS_TREE}')
print('--DONE--')

# PLOTS 
print('--PLOTS--')

plt.clf()
FIG, (AX, AX2, AX3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16,4))
# Edit cmap 
CMAP = mpl.cm.get_cmap('turbo')
CMAP.set_over('lightgrey')
CMAP.set_under('darkgrey')

# Set parameters of histogram
RANGES = [[0, 80], [0, 80]]
BINS = 100

# Set histogram on propability
COUNTS, XBINS, YBINS = np.histogram2d(SR532, SR532_PRED, range = RANGES, bins = BINS)
COUNTS_PROPA = COUNTS/SR532[~np.isnan(SR532)].shape[0]
# MAE = mean_absolute_error(SR532, SR532_PRED)
PP = AX.pcolormesh(XBINS, YBINS, COUNTS_PROPA.T, cmap = CMAP, norm=LogNorm())
CC = plt.colorbar(PP, ax=AX, label = '%')
AX.set(xlabel = 'SR532', ylabel='SR532 PREDICTED', title = 'BY LOG FUNCTION F(X)') 
# AX.annotate(f'MAE = {MAE}:.3f')

COUNTS, XBINS, YBINS = np.histogram2d(SR532_TREE, SR532_PRED_TREE, range = RANGES, bins = BINS)
COUNTS_PROPA = COUNTS/SR532_TREE[~np.isnan(SR532_TREE)].shape[0]
# MAE = mean_absolute_error(SR532.TREE, SR532_PRED.TREE)
PP2 = AX2.pcolormesh(XBINS, YBINS, COUNTS_PROPA.T, cmap = CMAP, norm=LogNorm())
CC2 = plt.colorbar(PP2, ax=AX2, label = '%')
AX2.set(xlabel = 'SR532', ylabel='SR532 PREDICTED', title = 'BY DECISION TREE') 
# AX2.annotate(f'MAE = {MAE}:.3f')


AX3.plot(UNIT_VALUE, PTS_STATS.values, label='by F(X)')
AX3.plot(UNIT_VALUE, PTS_STATS_TREE.values, label='by Decision Tree')
AX3.legend()
AX3.set(xlabel = 'Units around the diagonal', ylabel='%', title = 'Quantity') 

plt.tight_layout()
plt.savefig(Path(MAINDIR, 'HSRL2-ER2-DataTest_Stats_between_LogFunctin&DecisionTree.png'))