import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
'''
Le script est utilisé pour traiter les données ER2 aéroportées.
'''

def get_altitude_reference(zbottom, ztop, altitude_total):
    '''
    Fonction permet de retrouver la position de l'altitude référence dans le vecteur de l'altitude et la valeur de altitude référence.
    Input:
        zbottom: le borne bas de l'intervale
        ztop: le borne haut de l'intervale
        altitude_total: le vecteur total de l'altitude
    Output:
        la valeur de l'altitude et son indexe dans le vecteur total
    '''
    def arg_median(a):
        '''
        Fonction permet de retrouver la position médiane de la zone de référence de l'altitude
        Input: 
            a = l'intervale de l'altitude où se trouve la zone de référence 
        Ouput:
            Indexe de la position dans cet intervale
        '''
        if len(a) % 2 == 1:
            return np.where(a == np.median(a))[0][0]
        else:
            l,r = len(a) // 2 - 1, len(a) // 2
            left = np.partition(a, l)[l]
            right = np.partition(a, r)[r]
            return np.where(a == left)[0][0]

    interval_ref = altitude_total[(altitude_total >= zbottom) & (altitude_total <= ztop)] 
    idxref = arg_median(interval_ref)
    zref = interval_ref[idxref]
    idxref_in_total = np.where(altitude_total == zref)[0][0]
    return zref, idxref_in_total

'''
2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
Et calculer son integrale entre zmin et zmax
'''
def get_backscatter_mol(p, T, w):
    '''
    Fonction permet de calculer le coef. de backscatter moléculaire 
    p(Pa), T(K), w(um)
    '''
    k = 1.38e-23
    betamol = (p/(k*T) * 5.45e-32 * (w/0.55)**(-4.09))
    alphamol = betamol/0.119
    return alphamol, betamol


def get_backscatter_mol_attn_v1(alphamol, betamol, alt, idxref):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de zref 
    '''    
    Tr = betamol[:].copy()
    Tr2 = np.zeros_like(betamol)
    for i in range(len(alt)-2, -1, -1):
        Tr[:,i] = Tr[:,i+1] + alphamol[:,i]*(alt[i+1]-alt[i])
        Tr2[:,i] = np.exp(-2*Tr[:,i])

    betamol_Z0 = betamol[:].copy()
    betamol_Z0[:,0:idxref] = betamol[:,0:idxref]*Tr2[:,0:idxref]
    return betamol_Z0


def get_backscatter_mol_attn_v2(alphamol, betamol, alt):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de l'altitude de l'instrument
    '''
    Tr = np.zeros_like(betamol)
    Tr2 = np.zeros_like(betamol)
    
    for i in range(len(alt)-2, -1, -1):
        Tr[i] = Tr[i-1] + alphamol[i]*(alt[i]-alt[i-1])
        Tr2[i] = np.exp(-2*Tr[i])
        
    betamol_Z0 = betamol*Tr2        
    return betamol_Z0

    
def processed(w, zbottom, ztop, er2path):
    DataProducts = xr.open_dataset(er2path, group='DataProducts')
    Nav_Data = xr.open_dataset(er2path, group='Nav_Data')
    State = xr.open_dataset(er2path, group='State')
    er2alt = DataProducts['Altitude'][:].values.flatten() #phony_dim_1: 1389
    time = Nav_Data['gps_time'][:].values.flatten() #phony_dim_7: 1404
    '''
    Appliquer aux données Opar : nuage 21.01.2019 et ciel clair 17.06.2019
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] 
    puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, er2alt)
    zintervalle = np.where((er2alt > zbottom) & (er2alt < ztop))[0]
    if w==532:
        Pr2_norm = DataProducts["532_total_attn_bsc"].values
    else:
        Pr2_norm = DataProducts["355_total_attn_bsc"].values

    f, ax = plt.subplots()
    ax.plot(DataProducts['355_total_attn_bsc']['phony_dim_0'],
       DataProducts['355_total_attn_bsc'].sel(phony_dim_1=400), label='355 ext')
    ax.plot(DataProducts['532_total_attn_bsc']['phony_dim_0'],
       DataProducts['532_total_attn_bsc'].sel(phony_dim_1=400), label='532 ext')
    ax.legend()
    plt.savefig('/homedata/nmpnguyen/ORACLES-ER2/test_total_attn_bsc_bef.png')
    plt.close()
    plt.clf()

    Pr2_integ = np.zeros(len(time))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + Pr2_norm[:,z]*(er2alt[z+1]-er2alt[z])

    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    TT = State['Temperature'] #phony_dim_9: 1404, phony_dim_10: 1389, K
    TP = State['Pressure']*101325 #phony_dim_9: 1404, phony_dim_10: 1389, atm --> Pa
    ND = State['Number_Density']
    AlphaMol, BetaMol = get_backscatter_mol(TP, TT, w*1e-3)
    BetaMol_Z0 = get_backscatter_mol_attn_v1(AlphaMol, BetaMol, er2alt, idx_ref)
    # BetaMol_Z0 = np.array([get_backscatter_mol_attn_v1(AlphaMol[i,:], BetaMol[i,:], er2alt, idx_ref) for i in tqdm(range(len(time)))])  
    
    BetaMol_integ = np.zeros(len(time))
    for z in zintervalle[:-1]:
        BetaMol_integ = BetaMol_integ + BetaMol_Z0[:,z]*(er2alt[z+1]-er2alt[z]) 
    
    '''
    3. Diviser l'un par l'autre pour obtenir cRF
    '''
    cRF = np.array(BetaMol_integ/Pr2_integ).reshape(-1,1)
    '''
    4. Normaliser les profils mesures par cRF
    '''
    # Pr2_norm = er2attn532*cRF
    BetaMol_Z0norm = BetaMol_Z0/cRF
    return Pr2_norm, BetaMol_Z0norm


from argparse import Namespace, ArgumentParser  
parser = ArgumentParser()
parser.add_argument("--ztop", "-top", type=int, help="bound top of calibration height", required=True)
parser.add_argument("--zbottom", "-bottom", type=int, help="bound top of calibration height", required=True)
# parser.add_argument("--wavelength", "-w", type=int, help="wavelength", required=True)
parser.add_argument("--add_file", "-f", type=str, help="add a file to calibrate", required=False)
parser.add_argument("--output_file", "-o", type=str, help="add a output filename", required=False)
opts = parser.parse_args()
print(opts)

### Determiner z0
zbottom = opts.zbottom #
ztop = opts.ztop #

wavelengths = [355, 532]

maindir = Path('/homedata/nmpnguyen/ORACLES-ER2/Raws/')
listpaths = sorted(maindir.glob('*20160924_R8.h5'))

for file_er2 in tqdm(listpaths):
    print(file_er2)
    DataProducts = xr.open_dataset(file_er2, group='DataProducts')
    Nav_Data = xr.open_dataset(file_er2, group='Nav_Data')   
    er2alt = DataProducts['Altitude'][:].values.flatten() 
    er2time = Nav_Data['gps_time'][:].values.flatten()
    new_signal = np.array([processed(wave, zbottom, ztop, file_er2)[0] for wave in wavelengths])
    # print(f'shape of new signal :{new_signal.shape}')
    new_simul = np.array([processed(wave, zbottom, ztop, file_er2)[1] for wave in wavelengths])
    print(new_signal. shape, er2time.shape)
    print(f'Start to write: {Path(maindir,"/RF/Calibrated/", file_er2.name)}')
    ds = xr.Dataset(data_vars = {"calibrated": (("wavelength","time","altitude"), new_signal),
                                "molecular": (("wavelength","time","altitude"), new_simul),}, 
                    coords = {
                        "time":er2time,
                        "altitude": er2alt,
                        "wavelength":wavelengths,                        
                    },
					attrs = {"calibration height":[zbottom, ztop],},)  
    if (opts.output_file):
        ds.to_netcdf(Path("/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/", file_er2.stem+opts.output_file), 'w')
    else:
        ds.to_netcdf(Path("/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/", file_er2.name), 'w')
