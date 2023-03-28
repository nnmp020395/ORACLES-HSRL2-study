import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


'''
According to **Fully automated detection of Cloud and Aerosols layers in the CALIPSO Lidar Measurements** 

Steps:

- Detecte threshold value via distribution of molecules and particles
- Etablish an initial threshold level 
- Threshold ajustments 

Database: ER2 (HSRL2)
'''

def convert_gpstime(gpstime, date, convert=False):
    def frmt(decimal_time): # You can rewrite it with 'while' if you wish
        hours = int(decimal_time)#
        minutes = np.round(decimal_time - hours, 4)*60
        seconds = np.round(minutes - int(minutes), 4)*60
        HMS_time = f'{hours}:{int(minutes)}:{int(seconds)}'#"%s:%s:%f"%(hours, int(minutes), int(seconds))
        return HMS_time
    if convert==True:
        list_HMS_time = list(map(frmt, gpstime))
        list_YMD_HMS = list(map(lambda orig_string: date+' '+orig_string, list_HMS_time))
        pd_YMD_HMS = pd.to_datetime(list_YMD_HMS).strftime('%Y-%m-%d %H:%M:%S')
    else:
        list_gpstime_str = list(map(lambda n: '%.3f'%n, gpstime))
        list_YMD_HMS = list(map(lambda orig_string: date+' '+orig_string, list_gpstime_str))
        pd_YMD_HMS = list_YMD_HMS
    return pd.to_datetime(pd_YMD_HMS)

def find_nearest(array, value):
#     time_array, value = pd.to_datetime(time_array), pd.to_datetime(value)
    idt = (np.abs(array - value)).argmin()
    value = array[idt]
    return idt, value

def integration1D_from_top(data, height):
    '''
    Intégrer le profil 1D du signal en fonction de l'altitude
    ------------------------------------------------------
    '''
    array_integrated = np.full(data.shape, np.nan)
    for h in range(len(height)-2, 0, -1):
        delta_z = height[h] - height[h-1]
        array_integrated[h] = np.nansum([array_integrated[h+1], data[h]*delta_z], axis=0)
#         X3[j] = np.nansum([X3[j+1], (mean_atb355[i,j].values*delta_z)], axis=0) 
    return array_integrated

    
class automated_detection:
    def profils_threshold_initial(data2D, data2D_ref, height, zcalib_bottom, const):
        '''
        Initialiser le seuil pour définir les couches 
        ---------------------------------------------
        C0 , C1 sont des coefs à régler le seuil 
        --> à déterminer différemment entre les instruments
        ---------------------------------------------
        data : data en 2D
        data_ref : data en 2D, ici le signal moléculaire
        '''
        C0 = C1 = const
        index_alt_ref = find_nearest(height, zcalib_bottom)[0]
        data_bck = np.nanmean(data2D[:, :index_alt_ref], axis=1)
        Rthreshold = np.full(data2D.shape, np.nan)

        for i in tqdm(range(index_alt_ref, 0, -1)):
            Rthreshold[:,i] = (1 + C0*data_bck/data2D_ref[:,i] + C1*np.sqrt(data2D_ref[:,index_alt_ref]*data2D_ref[:,i]))

        return Rthreshold

    def get_layers_boundaries_data2D(data2D, threshold2D):
        '''
        data2D : array2D of SR, type = values 
        threshold : array2D of threshold, type = numpy
        '''
        data2D_after = data2D - threshold2D
        where_profil_notNaN = np.where(~np.isnan(data2D).all(axis=1))[0]
        print('all nan values , pas de layer', np.where(np.isnan(data2D).all(axis=1))[0])
        #-----------------
        ids_superieur_selected = []
        ids_inferieur_selected = []
        for i in where_profil_notNaN:
            # lower
            #-----------------
            id_superieur = np.where(data2D_after[i, :] > 0)[0]
            id_commun = id_superieur[1:]
            id_superieur_selected = np.concatenate(([id_superieur[0]], id_commun[(id_superieur[1:]-id_superieur[:-1]>1)] + 1))
            ids_superieur_selected.append(id_superieur_selected)
            # upper
            #-----------------
            id_inferieur = np.where(data2D_after[i, :] < 0)[0]
            id_commun = id_inferieur[1:]
            id_inferieur_selected = np.concatenate(([id_inferieur[0]], id_commun[(id_inferieur[1:]-id_inferieur[:-1]>1)] + 1))
            ids_inferieur_selected.append(id_inferieur_selected)
        
        return ids_superieur_selected, ids_inferieur_selected, where_profil_notNaN


    def labeling_layers(sr355, sr532, height, ida_lower, ida_upper, zcalib_bottom):
        '''
            ida_lower : ida_chosen --> index de l'altitude bottom
            ida_upper : ida_chosen2 --> index de l'altitude top
        '''
        # setting altitude lower / upper
        #--------------------------------
        alt_lower = np.concatenate(([0], height[ida_lower]))
        alt_upper = np.concatenate((height[ida_upper], [height[-1]]))
        right = np.sort(np.concatenate(([0], height[ida_lower], height[ida_upper], [height[-1]])))[1:]
        left = np.sort(np.concatenate(([0], height[ida_lower], height[ida_upper], [height[-1]])))[:-1]
        ida_intervals = pd.IntervalIndex.from_arrays(left, right, closed='left')
        # setting dataframe that contains interval layers
        #-------------------------------------------------
        sr_layers = pd.DataFrame(np.array([sr355, sr532, height]).T, 
                                columns=['sr355', 'sr532', 'alt'])
        sr_layers['layers'] = pd.cut(sr_layers['alt'], ida_intervals)
        sr_layers_mean = sr_layers.groupby('layers').agg({'sr355': lambda x: x.mean(skipna=True)})
        sr_layers_mean['layout'] = 0

        # labeling: 0: surface/couche limite, 1: clear sky, 2: aerosols, 3: clouds, 4: toa
        for r in range(len(left)): 
            if (left[r] < 1000):
                sr_layers_mean['layout'].iloc[r] = '0'
            elif (left[r] > 1000) & (left[r] < zcalib_bottom): #file.attrs['calibration height'][0]
                if (sr_layers_mean['sr355'].iloc[r] < 1.2) :
                    if right[r] > zcalib_bottom:
                        sr_layers_mean['layout'].iloc[r] = '4'
                    else:
                        sr_layers_mean['layout'].iloc[r] = '1'
                elif (sr_layers_mean['sr355'].iloc[r] > 1.2) & (sr_layers_mean['sr355'].iloc[r] < 5) : 
                    sr_layers_mean['layout'].iloc[r] = '2'
                elif (sr_layers_mean['sr355'].iloc[r] > 5) : 
                    sr_layers_mean['layout'].iloc[r] = '3'
            else:
                sr_layers_mean['layout'].iloc[r] = '4'

        idNotNan_sr_layers = np.where(~pd.isna(sr_layers['layers']))[0]
        layout_array = sr_layers_mean.loc[sr_layers.dropna(subset=['layers'])['layers']]['layout']
        full_layout = np.full(sr_layers.shape[0], np.nan)
        full_layout[idNotNan_sr_layers] = layout_array

        sr_layers['layout'] = full_layout
        return sr_layers, sr_layers_mean



    