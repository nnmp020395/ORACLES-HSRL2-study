import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt

'''
Application de Rayleigh-Fit aux données aéroportés ORACLES-ER2 ayant déjà des signaux calibrés
Objectif: Contrôler le Rayleight-Fit
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
    for i in range(idxref, -2, -1):
        Tr[i] = Tr[i+1] + alphamol[i]*(alt[i+1]-alt[i])
        Tr2[i] = np.exp(-2*Tr[i])

    betamol_Z0 = betamol.copy()
    betamol_Z0[0:idxref] = betamol[0:idxref]*Tr2[0:idxref]
    return betamol_Z0

def get_backscatter_mol_attn_v2(alphamol, betamol, alt):
    '''
    Cette fonction permet de calculer la retrodiffusion attenuee à partir de l'altitude de l'instrument
    '''
    Tr = np.zeros_like(betamol)
    Tr2 = np.zeros_like(betamol)
    for i in range(len(alt)-2, -1, -1):
        Tr[i] = Tr[i+1] + alphamol[i]*(alt[i+1]-alt[i])
        Tr2[i] = np.exp(-2*Tr[i])

    betamol_Z0 = betamol*Tr2
    return betamol_Z0

    
### Récupérer les fichiers et lire les groups de données 
maindir = Path('/homedata/nmpnguyen/ORACLES-ER2/')
listpaths = sorted(maindir.glob('*R8.h5'))
for er2path in listpaths:
    DataProducts = xr.open_dataset(er2path, group='DataProducts')
    Nav_Data = xr.open_dataset(er2path, group='Nav_Data')
    State = xr.open_dataset(er2path, group='State')

    ### Lire les variables depuis des groups
    er2alt = DataProducts['Altitude'][:].values.flatten() #phony_dim_1: 1389
    time = Nav_Data['gps_time'][:].values.flatten()
    TT = State['Temperature'] #K
    TP = State['Pressure']*101325 #because it's in atm -> Pa. 
    ND = State['Number_Density']

    ### Determiner z0
    zbottom = 17700
    ztop = 19000
    Z0, idx_ref = get_altitude_reference(zbottom, ztop, er2alt)
    zintervalle = np.where((er2alt > zbottom) & (er2alt < ztop))[0]

    '''
    1. Calculer le profil Pr2_z0 = Pr2[z]/Pr2[z0] puis calculer la valeur de son intégrale etre zmin et zmax
    '''
    w = 355
    er2attn = DataProducts[f'{str(w)}_total_attn_bsc'].values*1e-3
    Pr2_integ = np.zeros(len(time))
    for z in zintervalle[:-1]:
        Pr2_integ = Pr2_integ + er2attn[:,z]*(er2alt[z+1]-er2alt[z])

    '''
    2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
    Et calculer son integrale entre zmin et zmax
    '''
    AlphaMol, BetaMol = get_backscatter_mol(TP.values, TT.values, w*1e-3)
    print(AlphaMol.shape)
    BetaMol_Z0_v2 = np.array([get_backscatter_mol_attn_v2(AlphaMol[i,:], BetaMol[i,:], er2alt) for i in range(len(time))])
    BetaMol_Z0 = np.array([get_backscatter_mol_attn_v1(AlphaMol[i,:], BetaMol[i,:], er2alt, idx_ref) for i in range(len(time))])

    BetaMol_integ = np.zeros(len(time))
    BetaMol_integ_v2 = np.zeros(len(time))
    for z in zintervalle[:-1]:
        BetaMol_integ = BetaMol_integ + BetaMol_Z0[:,z]*(er2alt[z+1]-er2alt[z]) 
        BetaMol_integ_v2 = BetaMol_integ_v2 + BetaMol_Z0_v2[:,z]*(er2alt[z+1]-er2alt[z]) 

    '''
    3. Diviser l'un par l'autre pour obtenir cRF
    '''
    cRF = (BetaMol_integ/Pr2_integ).reshape(-1,1)
    cRF_v2 = (BetaMol_integ_v2/Pr2_integ).reshape(-1,1)
    '''
    4. Normaliser les profils mesures par cRF
    '''
    BetaMol_Z0norm = BetaMol_Z0/cRF
    BetaMol_Z0_v2norm = BetaMol_Z0_v2/cRF_v2
    '''
    5. Calculer le résidus residus et l'intégrale sur l'altitude
    '''
    residus = np.zeros(len(time))
    for t in range(1, len(er2alt)):
        residus = residus + np.nansum([er2attn[:,t],-BetaMol_Z0norm[:,t]], axis=0)*(er2alt[t]-er2alt[t-1])

    fichier = open(Path('/homedata/nmpnguyen/ORACLES-ER2/RF/', er2path.name.split('.')[0], str(w), er2path.name.split('.')[0]+'_residus.txt'), 'w')
    fichier.write(f'calibration altitude: {[zbottom, ztop]}')
    fichier.write(f'wavelength: {w}')
    for i in range(len(residus)):
        fichier.write(f'\n{round(residus[i],6)},{str(time[i])}')

    fichier.close()
    '''
    6. Plot un profil normalisé et moléculaire
    '''
    for n in range(0, len(time), 100):
        plt.clf()
        plt.close()
        fig, (ax3,ax) = plt.subplots(nrows=1, ncols=2, figsize=[12,6], sharey=True)#
        ax3.semilogx(er2attn[n,:len(er2alt)], er2alt, label='signal normalisé')
        ax3.semilogx(BetaMol_Z0norm[n,:], er2alt, label='signal moléculaire attn depuis z0')
        # ax3.semilogx(BetaMol_Z0_v2[n,:], ipralrange, label='signal moleculaire attn depuis sol')
        ax3.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        leg3 = ax3.legend(loc='best', frameon=False)
        leg3.set_title(f'4/ {str(w)}: cRF = {cRF[n]}')
        ax3.set(xlabel='backscatter, 1/m.sr')


        ax.plot(er2attn[n,:len(er2alt)]/BetaMol_Z0norm[n,:], er2alt, label='depuis z0')
        ax.axvline(x=1, color='k')
        ax.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
        ax.set(xlabel='scatering ratio')
        leg = ax.legend(loc='best', frameon=False)
        leg.set_title(f'{str(w)}: scattering ratio')
        ax.set_xlim(-.5, 5)

        plt.suptitle(f'{er2path}\n time:{time[n]}')
        plt.savefig(Path('/homedata/nmpnguyen/ORACLES-ER2/RF', er2path.name.split('.')[0], str(w), 
            er2path.name.split('.')[0] + f'_{str(time[n])}.png'))