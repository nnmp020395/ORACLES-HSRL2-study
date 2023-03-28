import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import matplotlib as mpl
from tqdm import tqdm
from datetime import date
import warnings 
warnings.filterwarnings("ignore")
import os, sys, math
# os.chdir('/homedata/nmpnguyen/ORACLES-ER2/Codes')
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/Codes')
from outils import convert_gpstime
# os.chdir('/homedata/nmpnguyen/comparaison/Codes')
sys.path.append('/homedata/nmpnguyen/comparaison/Codes')
from fonctions import dataset, conditions, get_file, check, conversion, plots

DATES = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']
MAINDIR = Path('/homedata/nmpnguyen/ORACLES-ER2/')


for DATE in DATES:
    PARTTERN = f'HSRL2_ER2_{DATE}_R8.h5'
    FILE = sorted(MAINDIR.glob(PARTTERN))[0]
    print(FILE)
    DATA = xr.open_dataset(FILE, group = 'Nav_Data')
    TIME = DATA['gps_time'].values.flatten()
    DATA = xr.open_dataset(FILE, group = 'DataProducts')
    ALT = DATA['Altitude'].values.flatten()*1e-3
    ALT_AIRCRAFT = ALT[-1]
    ATB_532 = DATA['532_total_attn_bsc'].values
    ATB_355 = DATA['355_total_attn_bsc'].values
    NUMB_DENSITY = xr.open_dataset(FILE, group = 'State')['Number_Density'].values
    DIFF_CROSS_SECTION = {'532' : 6.1668318e-32, '355' : 3.2897988e-31, 'unit' : 'm^2.sr^-1'}
    BETA_MOL_355 = DIFF_CROSS_SECTION['355']*NUMB_DENSITY * 1e3
    BETA_MOL_532 = DIFF_CROSS_SECTION['532']*NUMB_DENSITY * 1e3
    ALPHA_MOL_355 = (math.pi*4/1.5) * BETA_MOL_355
    ALPHA_MOL_532 = (math.pi*4/1.5) * BETA_MOL_532
    BETA_PART_355 = DATA['355_bsc'].values
    BETA_PART_532 = DATA['532_bsc'].values
    ALPHA_PART_355 = DATA['355_ext'].values
    ALPHA_PART_532 = DATA['532_ext'].values
    BETA1 = (BETA_MOL_532 + BETA_PART_355)/BETA_MOL_532
    len_time = TIME.shape[0]
    len_alt = ALT.shape[0]

    MULTI_SCATTERING_COEFF = 0.9
    INTEG1, INTEG2 = np.full(ALPHA_MOL_355.shape, np.nan), np.full(ALPHA_MOL_355.shape, np.nan)
    for a in tqdm(range(len_alt)):
        INTEG1[:,a] = -2 * np.nansum(np.dstack((ALPHA_MOL_532[:,a] , ALPHA_PART_355[:,a] * MULTI_SCATTERING_COEFF)),2) * np.abs(ALT[a]-ALT_AIRCRAFT)
        INTEG2[:,a] = -2 * ALPHA_MOL_532[:,a] * np.abs(ALT[a]-ALT_AIRCRAFT)

    SR532_NEW = BETA1 * np.exp(INTEG1)/np.exp(INTEG2)

    DATASET = xr.Dataset({
        'NUMB_DENSITY' : (('time', 'alt'),NUMB_DENSITY,
            {'units': 'm^-3', 
            'long_name':'density_number'}), 
        'BETA_MOL_355' : (('time', 'alt'),BETA_MOL_355,
            {'units': 'km^-1.sr^-1', 
            'long_name':'beta_mol_355'}), 
        'BETA_MOL_532' : (('time', 'alt'),BETA_MOL_532,
            {'units': 'km^-1.sr^-1', 
            'long_name':'beta_mol_532'}), 
        'ALPHA_MOL_355' : (('time', 'alt'),ALPHA_MOL_355,
            {'units' : 'km^-1',
            'long_name' : 'alpha_mol_355'}), 
        'ALPHA_MOL_532' : (('time', 'alt'),ALPHA_MOL_532,
            {'units' : 'km^-1',
            'long_name' : 'alpha_mol_532'}), 
        'BETA_PART_355' : (('time', 'alt'),BETA_PART_355,
            {'units' : 'km^-1.sr^-1',
            'long_name' : 'beta_part_355'}), 
        'BETA_PART_532' : (('time', 'alt'),BETA_PART_532,
            {'units' : 'km^-1.sr^-1',
            'long_name' : 'beta_part_532'}), 
        'ALPHA_PART_355' : (('time', 'alt'),ALPHA_PART_355,
            {'units' : 'km^-1',
            'long_name' : 'alpha_part_355'}), 
        'ALPHA_PART_532' : (('time', 'alt'),ALPHA_PART_532,
            {'units' : 'km^-1',
            'long_name' : 'alpha_part_532'}),
        'BETA1' : (('time', 'alt'), BETA1,
            {'units' : 'AU',
            'long_name' : '(beta_mol_532 + beta_part_355) / beta_mol_532'}),
        'INTEG1' : (('time', 'alt'), INTEG1,
            {'units' : 'AU',
            'long_name' : '-2 * integration from Z_sat to Z of (alpha_mol_532 + MULTI_SCATTERING_COEFF * alpha_part_355)'}),
        'INTEG2' : (('time', 'alt'), INTEG2,
            {'units' : 'AU',
            'long_name' : '-2 * integration from Z_sat to Z of (alpha_mol_532)'}),
        'SR532_PRED' : (('time', 'alt'), SR532_NEW,
            {'units' : 'AU',
            'long_name' : 'SR532_converted_by_Artem_equation'}),
        'ALT_AIRCRAFT' : ALT_AIRCRAFT,
        'MULTI_SCATTERING_COEFF' : MULTI_SCATTERING_COEFF,
        },
        coords = {
            'time' : (('time'), TIME, {'units' : 'hr'}),
            'alt' : (('alt'), ALT, {'units' : 'km'}),
        },
        attrs = {
            'Title' : 'ORACLES-ER2 HSRL2 Convert dataset',
            'Instrument_name' : 'ORACLES-ER2 HSLR2',
            'Station_name' : '',
            'Start_Datetime' : f'{TIME[0]}',
            'End_Datetime' : f'{TIME[-1]}',
            'Convert_method' : 'Equation in A.G. Feofilov and al.',
            'Original_dataset' : f'{FILE.name}',
            'Date_data_created' : date.today().strftime('%d/%m/%Y'),
        },
    )
    DATASET.to_netcdf(Path(MAINDIR, 'leaning_model_test', 'Products', f'{FILE.stem}_convert_Artem_func.nc'), 'w')
