import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def convert_gpstime(gpstime, date, convert=False):
    '''
    Cette fonction convertit les jours du format décimal en format date-heure
    '''
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
        list_YMD_HMS = list(map(lambda orig_string: orig_string, list_gpstime_str))
        pd_YMD_HMS = list_YMD_HMS
    return pd.to_datetime(pd_YMD_HMS)


def find_nearest_time(time_array, value):
    '''
    Cette fonction permet de retrouver l'index et la valeur de array le plus proche à la valeur référence. 

    Paramètres:
    --------------
    Input: 
        array: numpy array 1D
        value: valeur référence
        time_option: si True, convertir les arrays en format datetime

    Output:
        data
        index
    '''
    time_array, value = pd.to_datetime(time_array), pd.to_datetime(value)
    idt = (np.abs(time_array - value)).argmin()
    value = time_array[idt]
    return idt, value

def find_nearest(array, value):
    idt = (np.abs(array - value)).argmin()
    value = array[idt]
    return idt, value

def remove_clouds_profils(data, cloud_height_array, alt_max, 
                          mask_option, return_index = False):
    """
    Remove ER2 profiles containing cloud allow a defined altitude.

    Parameters
    ----------
    Input:
        data : matrice input, data need to remove profiles
        alt_max : float, the maximum altitude of clouds in meters.
        mask_option : 'top' ou 'bottom'
    Output : 
        if return_index == True, return index of clouds profils remainded, 
        if not return only validated data
    """    
    if (mask_option == 'top'):
        id_profiles_mask = np.where((cloud_height_array < alt_max)|np.isnan(cloud_height_array))[0]
    else:
        id_profiles_mask = np.where((cloud_height_array > alt_max)|np.isnan(cloud_height_array))[0]
    
    
    if (return_index):
        return id_profiles_mask
    else:
        masked_data = data[id_profiles_mask, :]
        return masked_data


def get_all_flags(input_data, limites):
    """
    Get all flags and remove 
    
    Parameters
    ----------
    Input:
        data where stock variables and mask signal 
        limites where stock altitude maximum and calibration height 
    Output:
        masked_data 
    """
    # Remove cloud in calibration aera 
    id_masked_cloud = remove_clouds_profils(input_data, input_data['cloud_top_height'].values, 
                                               limites['alt_max'], mask_option=limites['mask_cloud_option'], 
                                               return_index = True)
    
    data_masked_cloud = input_data.isel(time=id_masked_cloud)
#     final_masked_data = data_masked_cloud.where(data_masked_cloud['flags'] == 1 , drop=False)
    return data_masked_cloud
    

def formatter(filepath):
    """
    pour re-formatter les données ER2 en netcdf
    """
    Nav_Data = xr.open_dataset(filepath, group = 'Nav_Data')
    Data_Products = xr.open_dataset(filepath, group = 'DataProducts')
    date = filepath.stem.split('_')[2]
    time_converted = convert_gpstime(Nav_Data['gps_time'].values.ravel(), date, True)
    
    output_data = xr.Dataset(
        coords = {
            'time' : ('time', time_converted),
            'height' : ('height', Data_Products['Altitude'].values.ravel()),
        },
        data_vars = {
            'attn_bsc_355' : (['time', 'height'], Data_Products['355_total_attn_bsc'].values),
            'attn_bsc_532' : (['time', 'height'], Data_Products['532_total_attn_bsc'].values),
            'flags' : (['time', 'height'], Data_Products['mask_low'].values),
            'cloud_top_height' : ('time', Data_Products['cloud_top_height'].values.ravel()),
        },
        attrs = {
            'calibration_height' : [17700,19000]
        }
    )
    return output_data