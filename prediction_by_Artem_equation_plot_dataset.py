import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import matplotlib as mpl
from tqdm import tqdm
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings 
warnings.filterwarnings("ignore")
import os, sys, math
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/Codes')
from outils import convert_gpstime

'''
Le script est pour tracer les quicklooks illustrés des résultats de la conversion du signal 
suivant l'équation d'Artem. 

Les variables peuvent être changer en fonction de l'objectif.

Le nom de output file doit être changé pour adapter à l'objectif de l'étude. Le répertoire 
Le répertoire est toujours /homedata/nmpnguyen/ORACLES-ERS/Figs

'''
DATES = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']
MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/')

'''
En lançcant, choisir dans le terminal l'option de type de plot: quicklook ou scatterplot  
'''
from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
parser.add_argument("--plot_opt", "-opt", type=str, help="Plot type, if q = quicklook, s = scatterplot", required=True)
opts = parser.parse_args()

for DATE in DATES:
    # 0. Initialiser la figure
    #-------------------------
    FIG, (AX, AX2, AX3) = plt.subplots(ncols=3, figsize = (14,4))
    cmapp = mpl.cm.get_cmap("jet")
    cmapp.set_over('black')
    cmapp.set_under('lightgrey')
    cmapp.set_bad('white')
    plt.rcParams['font.size'] = 13

    
    # 1. Extraire SR532_PRED par l'équation d'Artem
    #----------------------------------------------
    
    PARTTERN = f'HSRL2_ER2_{DATE}_R8_convert_Artem_func.nc'
    FILE = Path(MAINDIR, 'leaning_model_test', 'Products', PARTTERN)
    print(FILE)
    DATA = xr.open_dataset(FILE)
    SR532_PRED = DATA['SR532_PRED'].values
    ALT = DATA['alt'].values #(km)
    TIMEP = DATA['time'].values

    # 2. Extraire SR532 & SR355 des observations
    #------------------------------------------- 
    PARTTERN = f'HSRL2_ER2_{DATE}_R8_v2.nc'
    FILE = Path(MAINDIR, 'RF', 'Calibrated', PARTTERN)
    print(FILE)
    DATA = xr.open_dataset(FILE)
    SR355 = (DATA['calibrated']/DATA['molecular']).sel(wavelength = 355).values
    SR532 = (DATA['calibrated']/DATA['molecular']).sel(wavelength = 532).values
    # TIME = np.intersect1d(np.intersect1d(SR355.time, DATA.time), np.intersect1d(SR532.time, DATA.time))#DATA['time'].values
    ALTITUDE = DATA['altitude'].values/1000 #(km)
    TIME = DATA['time'].values

    # 3. Plots
    #---------
    if (opts.plot_opt == 'q') : 
        OUTPUT_FILENAME = f'HSRL2_ER2_{DATE}_R8_convert_Artem_SR355_SR532_SR532pred.png'
        SR355 = np.ma.masked_where(SR355 < 0, SR355)
        P = AX.pcolormesh(TIME, ALTITUDE, SR355.T, norm=LogNorm(vmin=1e-2, vmax=1e2), shading='auto', cmap = 'jet')
        plt.colorbar(P, ax=AX,extend='both')
        AX.set_title("SR355")
        AX.set_xlabel('TIME,hr')
        AX.set_ylabel('ALTITUDE,km')

        SR532 = np.ma.masked_where(SR532 < 0, SR532)
        P2 = AX2.pcolormesh(TIME, ALTITUDE, SR532.T, norm=LogNorm(vmin=1e-2, vmax=1e2), shading='auto', cmap = 'jet')
        plt.colorbar(P2, ax=AX2,extend='both')
        AX2.set_title("SR532")
        AX2.set_xlabel('TIME,hr')
        AX2.set_ylabel('ALTITUDE,km')

        SR532_PRED = np.ma.masked_where(SR532_PRED < 0, SR532_PRED)
        P3 = AX3.pcolormesh(TIMEP, ALT, SR532_PRED.T, norm=LogNorm(vmin=1e-2, vmax=1e2), shading='auto', cmap = 'jet')
        plt.colorbar(P3, ax=AX3,extend='both')
        AX3.set_title("SR'532")
        AX3.set_xlabel('TIME,hr')
        AX3.set_ylabel('ALTITUDE,km')

    else: 
        RANGES = [[0, 30], [0, 80]]
        BINS = 100
        OUTPUT_FILENAME = f'HSRL2_ER2_{DATE}_R8_convert_Artem_SR355_SR532_SR532pred_Scatterplot.png'
        # ONLY MEASUREMENTS
        #------------------
        COUNTS, XBINS, YBINS = np.histogram2d(SR355.ravel(), SR532.ravel(), range = RANGES, bins = BINS)
        COUNTS_PROPA = 100*COUNTS/SR532.ravel()[~np.isnan(SR532.ravel())].shape[0]
        P = AX.pcolormesh(XBINS, YBINS, COUNTS_PROPA.T, cmap = cmapp, norm=LogNorm(vmin=1e-4, vmax=1e0))
        plt.colorbar(P, ax=AX, label = 'PERCENTAGE OF POINTS', extend='both')
        AX.set_xlabel('SR355 MEASURED')
        AX.set_ylabel('SR532 MEASURED')
        
        # WITH CONVERT SIGNAL
        #--------------------
        COUNTS, XBINS, YBINS = np.histogram2d(SR355.ravel(), SR532_PRED.ravel(), range = RANGES, bins = BINS)
        COUNTS_PROPA = 100*COUNTS/SR532.ravel()[~np.isnan(SR532.ravel())].shape[0]
        P2 = AX2.pcolormesh(XBINS, YBINS, COUNTS_PROPA.T, cmap = cmapp, norm=LogNorm(vmin=1e-4, vmax=1e0))
        plt.colorbar(P2, ax=AX2, label = 'PERCENTAGE OF POINTS', extend='both')
        AX2.set_xlabel('SR355 MEASURED')
        AX2.set_ylabel('SR532 PREDICT')

        # SR532_PRED vs SR532
        #--------------------
        RANGES = [[0, 60], [0, 60]]
        VALIDE_MASK = np.logical_and(np.isfinite(SR532.ravel()), np.isfinite(SR532_PRED.ravel()))
        SR532_NEW = SR532.ravel()[VALIDE_MASK]
        SR532_PRED_NEW = SR532_PRED.ravel()[VALIDE_MASK]
        RANGE_MASK = np.where((SR532_NEW >= RANGES[0][0]) & (SR532_NEW <= RANGES[0][1]) & (SR532_PRED_NEW >= RANGES[1][0]) & (SR532_PRED_NEW <= RANGES[1][1]))

        MAE = mean_absolute_error(SR532_NEW[RANGE_MASK], SR532_PRED_NEW[RANGE_MASK])
        MSE = mean_squared_error(SR532_NEW[RANGE_MASK], SR532_PRED_NEW[RANGE_MASK])
        R2 = np.corrcoef(SR532_NEW[RANGE_MASK], SR532_PRED_NEW[RANGE_MASK])
        print(MAE, MSE, R2[0, 1])

        COUNTS, XBINS, YBINS = np.histogram2d(SR532.ravel(), SR532_PRED.ravel(), range = RANGES, bins = BINS)
        COUNTS_PROPA = 100*COUNTS/SR532.ravel()[~np.isnan(SR532.ravel())].shape[0]
        P3 = AX3.pcolormesh(XBINS, YBINS, COUNTS_PROPA.T, cmap = cmapp, norm=LogNorm(vmin=1e-4, vmax=1e0))
        plt.colorbar(P3, ax=AX3, label = 'PERCENTAGE OF POINTS', extend='both')
        AX3.set_xlabel('SR532 MEASURED')
        AX3.set_ylabel('SR532 PREDICT')
        AX3.annotate(f'MAE = {MAE:.4f}\nMSE = {MSE:.3e}\nCORR = {R2[0,1]:.4f}', xy=(10, 60), xytext=(10, 60), bbox=dict(boxstyle="round", fc="w"))
      

    plt.suptitle(f'DAY : {DATE}')
    plt.tight_layout()
    plt.savefig(Path(MAINDIR, 'leaning_model_test', 'Figs', OUTPUT_FILENAME))
    plt.clf()



