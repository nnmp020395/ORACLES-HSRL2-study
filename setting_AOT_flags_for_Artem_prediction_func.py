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
Le script est pour classifier les SR ou ATB, prédicts ou non en fonction de AOT-Aerosol Optical Thickness,
puis 'concatenate' en un seul fichier qui sert à la quantification globale. 


Cette classification sert principalement à l'évaluation de la performance de la conversion du signal d'après l'équation d'Artem

Etape 1: Insérer les SR'532 calculés par l'équation d'Artem
Etape 2: Insérer les AOT et AOT_col
Etape 3: Flagger / Classifier 

'''

DATES = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']
#-------------------
SR355 = []
SR532 = []
BETAPART355 = []
SR532_PRED = []
TIME = []
ALTITUDE_2D = []
AOT_355s= []
#-------------------

for DATE in DATES: 
    '''
    ETAPE 1 : Insérer les AOT et AOT_col
    '''
    MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/Raws')
    PARTTERN = f'HSRL2_ER2_{DATE}_R8.h5'
    FILE = sorted(MAINDIR.glob(PARTTERN))[0]
    print(FILE)
    DATA = xr.open_dataset(FILE, group = 'Nav_Data')
    TIME = DATA['gps_time'].values.flatten()
    DATA = xr.open_dataset(FILE, group = 'DataProducts')
    DATA = xr.open_dataset(FILE, group = 'Nav_Data')
    TIME = DATA['gps_time'].values.flatten()
    DATA = xr.open_dataset(FILE, group = 'DataProducts')
    AOT_355 = [DATA['355_AOT_hi_col'].values, np.full(DATA['355_AOT_hi_col'].values.shape, 0)]
    aot355_hi_col = DATA['355_AOT_hi_col'].values
    '''
    ETAPE 2 : FLAGGER / CLASSIFIER
    '''
    AOT_355[1][AOT_355[0] < 0.1] = 1
    AOT_355[1][(AOT_355[0] >= 0.1) & (AOT_355[0] < 0.5)] = 2
    AOT_355[1][(AOT_355[0] >= 0.5) & (AOT_355[0] < 1)] = 3
    AOT_355[1][AOT_355[0] > 1] = 4

    aot355_hi_col = DATA['355_AOT_hi_col'].values
    aotp_not_nan = np.unique(np.where(~np.isnan(DATA['355_AOT_hi_col'].values))[0])
    for i in aotp_not_nan :
        pmax = np.where(~np.isnan(aot355_hi_col[i]))[0].max()
        aot355_hi_col[i][pmax:]=0
    
    aot355_hi_col[np.where(np.isnan(aot355_hi_col).all(axis=1))[0],:]=0
    '''
    ETAPE 3 : Insérer les SR'532 calculés par l'équation d'Artem
    '''
    MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products/')
    PARTTERN = f'HSRL2_ER2_{DATE}_R8_convert_Artem_func.nc'
    FILE = Path(MAINDIR, PARTTERN)
    print(FILE)
    DATA = xr.open_dataset(FILE)
    DATA['flags_AOT_hi'] = (['time', 'alt'], AOT_355[1])
    DATA['flags_AOT_hi'].assign_attrs({'units' : "AU", 'long_name' = 'flags of AOT integrated & duplicate full dataset'})
    DATA['flags_AOT_hi_col'] = (['time', 'alt'], aot355_hi_col)
    DATA['flags_AOT_hi_col'].assign_attrs({units = "AU", 'long_name' = 'flgas of AOT cumulated & not duplicate full dataset'})
    '''
    OVERWRITE NEW DATA FILE
    '''
    DATA.to_netcdf(Path(MAINDIR, f'{FILE.stem}_convert_Artem_func.nc'), 'a')



    