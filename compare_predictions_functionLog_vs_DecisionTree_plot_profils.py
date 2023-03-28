import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import matplotlib as mpl
import warnings 
warnings.filterwarnings("ignore")
import sys
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/Codes')
from outils import convert_gpstime, find_nearest


DATES = ['20160826', '20160920', '20160924', '20160916']
MOMENTS = ['13:00', '13:00', '13:00', '13:00']



for DATE, MOMENT in zip(DATES, MOMENTS):
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/')
	PARTTERN = f'HSRL2_ER2_{DATE}*_v2.nc'
	FILE = sorted(MAINDIR.glob(PARTTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE)
	SR355 = (DATA['calibrated']/DATA['molecular']).sel(wavelength = 355)
	SR532 = (DATA['calibrated']/DATA['molecular']).sel(wavelength = 532)
	TIME = DATA['time'].values
	ALTITUDE = DATA['altitude'].values
	TIME=convert_gpstime(TIME, DATE, True).values
	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/')
	PARTTERN = f'HSRL2_ER2_{DATE}_R8.h5'
	FILE = sorted(MAINDIR.glob(PARTTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE, group = 'DataProducts')
	BETAPART355 = DATA['355_bsc']

	#-------------------------------------------------------------
	# PREDICTION BY NON-LINEAR FUNCTION 
	#-------------------------------------------------------------
	f = lambda x : 0.20704*np.log(x) + 4.8181
	RATIO_PRED = f(BETAPART355)
	SR532_PRED = RATIO_PRED.values * SR355.values
	#-------------------------------------------------------------
	# PREDICTION BY DECISION TREE
	#-------------------------------------------------------------

	MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/')
	PATTERN = f'tree_3f*{DATE}_v2.nc'

	FILE = sorted(MAINDIR.glob(PATTERN))[0]
	print(FILE)
	DATA = xr.open_dataset(FILE)
	SR532_PRED_TREE = DATA['SR532_predicted']
	SR532_TREE = DATA['SR532_measured']
	TIME_TREE = pd.to_datetime(DATA['time'].values)

	plt.clf()


	FIG, ((ax), (ax2), (ax3)) = plt.subplots(ncols = 3, nrows = 3, 
	                                         figsize = (18,12), sharey=True, 
	                                         gridspec_kw={'width_ratios': [2,2,1]})
	cmapp = mpl.cm.get_cmap("jet")
	cmapp.set_over('black')
	cmapp.set_under('lightgrey')
	plt.rcParams['font.size'] = 13
	plt.rcParams['xtick.labelsize'] = 'small'

	#--------------------------------------------------
	# BY FUNCTION 
	#--------------------------------------------------
	p = ax[0].pcolormesh(TIME, ALTITUDE, SR532.T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100))
	plt.colorbar(p, ax=ax[0], label = 'SR532', extend='both')
	ax[0].set_xlabel('TIME')
	ax[0].set_ylabel('ALTITUDE km')
	ax[0].set_title('by f(x)=0.20704*np.log(x) + 4.8181')

	p2 = ax2[0].pcolormesh(TIME, ALTITUDE, SR532_PRED.T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100))
	plt.colorbar(p2, ax=ax2[0], label = 'SR532 PREDICT', extend='both')
	ax2[0].set_xlabel('TIME')
	ax2[0].set_ylabel('ALTITUDE km')

	p3 = ax3[0].pcolormesh(TIME, ALTITUDE, np.abs(SR532_PRED - SR532).T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=10))
	plt.colorbar(p3, ax=ax3[0], label = '| SR532 PREDICT - SR532 |', extend='both')
	ax3[0].set_xlabel('TIME')
	ax3[0].set_ylabel('ALTITUDE km')

	#--------------------------------------------------
	# BY DECISION TREE 
	#--------------------------------------------------
	pp = ax[1].pcolormesh(TIME_TREE, ALTITUDE, SR532_TREE.T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100))
	plt.colorbar(pp, ax=ax[1], label = 'SR532', extend='both')
	# pp = SR532_TREE.plot(x='time', y='altitude', cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100), ax=axx,
	#                     cbar_kwargs = {'label': 'SR532', 'extend':'both'})
	ax[1].set_xlabel('TIME')
	ax[1].set_ylabel('ALTITUDE km')
	ax[1].set_title('by Decision Tree (DT), dropna=True')

	pp2 = ax2[1].pcolormesh(TIME_TREE, ALTITUDE, SR532_PRED_TREE.T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100))
	plt.colorbar(pp2, ax=ax2[1], label = 'SR532 PREDICT', extend='both')
	# SR532_PRED_TREE.plot(x='time', y='altitude', cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=100), ax=axx2,
	#                     cbar_kwargs = {'label': 'SR532 PREDICT', 'extend':'both'})
	ax2[1].set_xlabel('TIME')
	ax2[1].set_ylabel('ALTITUDE km')

	pp3 = ax3[1].pcolormesh(TIME_TREE, ALTITUDE, np.abs(SR532_PRED_TREE - SR532_TREE).T, cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=10))
	plt.colorbar(pp3, ax=ax3[1], label = '| SR532 PREDICT - SR532 |', extend='both')
	# np.abs(SR532_PRED_TREE - SR532_TREE).plot(x='time', y='altitude', cmap = cmapp, norm=LogNorm(vmin=1e-1, vmax=10), ax=axx3,
	#                     cbar_kwargs = {'label': '| SR532 PREDICT - SR532 |', 'extend':'both'})
	ax3[1].set_xlabel('TIME')
	ax3[1].set_ylabel('ALTITUDE km')

	plt.setp(ax[0].get_xticklabels() + ax[1].get_xticklabels() +
	         ax2[0].get_xticklabels() + ax2[1].get_xticklabels() +
	         ax3[0].get_xticklabels() + ax3[1].get_xticklabels(), 
	         rotation=30, ha='right')

	#--------------------------------------------------
	# PROFILS
	# setting 1 profil's index
	#--------------------------------------------------


	T_INDEX, TT_INDEX = find_nearest(TIME, pd.to_datetime(DATE +' '+ MOMENT))[0], find_nearest(TIME_TREE, pd.to_datetime(DATE +' '+ MOMENT))[0]

	ax[2].plot(SR532[T_INDEX, :], ALTITUDE, color='green', label='Obs', zorder=10)
	ax[2].axvline(1, color='red', label='SR=1', zorder=0)
	ax[2].set_xlabel('SR532')
	ax[2].set_title(f'PROFILS {MOMENT}')
	ax[2].set_ylabel('ALTITUDE')
	ax[2].legend(loc='upper right')
	ax[2].set_xlim(-1,20)

	ax2[2].plot(SR532[T_INDEX, :], ALTITUDE, color='green', label='Obs', zorder=10)
	ax2[2].plot(SR532_PRED[T_INDEX, :], ALTITUDE, color='turquoise', label='by f(x)')
	ax2[2].plot(SR532_PRED_TREE[TT_INDEX, :], ALTITUDE, color='lawngreen', label='by DT')
	ax2[2].axvline(1, color='red', label='SR=1', zorder=0)
	ax2[2].set_xlabel('SR532 PREDICT')
	ax2[2].set_ylabel('ALTITUDE')
	ax2[2].legend(loc='upper right')
	ax2[2].set_xlim(-1,20)

	ax3[2].plot(np.abs(SR532_PRED[T_INDEX, :] - SR532[T_INDEX, :]), ALTITUDE, color='turquoise', label='by f(x)')
	ax3[2].plot(np.abs(SR532_PRED_TREE[TT_INDEX, :] - SR532_TREE[TT_INDEX, :]), ALTITUDE, color='lawngreen', label='by DT')
	ax3[2].axvline(0, color='red', label='SR=0', zorder=0)
	ax3[2].set_xlabel('RESIDUS |SR532 PREDICT - SR532|')
	ax3[2].set_ylabel('ALTITUDE')
	ax3[2].legend(loc='upper right')
	ax3[2].set_xlim(-1,20)

	plt.suptitle(f'HSRL2-ER2 {DATE} | PREDICTIONS')
	plt.tight_layout()
	plt.savefig(Path(MAINDIR, 'Figs', f'HSRL2-ER2-{DATE}-{MOMENT}_predictives-ql-profils.png'))