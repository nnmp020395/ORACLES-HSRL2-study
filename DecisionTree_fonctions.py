import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
import warnings 
warnings.filterwarnings("ignore")
import pickle
import math

'''
PRE AND POST-PROCESSING
'''
def train_test_days_split(list_days, test_size, random=False):
    '''
    Setting randomly or generally index for list of date files 
    with a chosen ratio size for testing dataset and training dataset. 
    '''
    # get number of list training days 
    counts_days = len(list_days)
    counts_train_days = np.int((1-test_size)*counts_days)
    li1 = np.arange(0, len(list_days),1)
    
    if random:
        # set random index for training days 
        ind_train_days = np.random.choice(range(counts_days), counts_train_days, replace=False)
        
        # get index left for testing days 
        li2 = np.array(ind_train_days)
        dif1 = np.setdiff1d(li1, li2)
        dif2 = np.setdiff1d(li2, li1)
        ind_test_days = np.concatenate((dif1, dif2))

        # get training & testing days 
        train_days = np.array(list_days)[ind_train_days]
        test_days = np.array(list_days)[ind_test_days]
    else:
        train_days = list_days[:counts_train_days]
        test_days = list_days[counts_train_days:]
    return train_days, test_days


def generate_data(list_days, configs, wavelength, variable_name):
    '''
    1. from list day chosen, go to the directory and get path of all files 
    2. choose variables & wavelength necessary 
    3. get dataset 

    configs = {
        'main_dir' : Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/'),
        'pattern_filename' : ['HSRL2_ER2_','_R8_v2.nc'],
        'year' : '',   
        'variables_name' : {
            'ATB' : 'calibrated', 
            'AMB' : 'molecular', 
            'time' : 'time',
            'range' : 'altitude'
        }, 
        'instrument' : 'ER2',
        'output_dir' : Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test')
        'random_version' : random_version
    }
    '''    

    # get all paths from list_days
    def get_path(list_days, configs):
        list_paths = []
        for day in list_days:
            filename = configs['pattern_filename'][0]+pd.to_datetime(day).strftime('%Y%m%d')+configs['pattern_filename'][1]
            list_paths.append(Path(configs['main_dir'], filename))
        return list_paths
         
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

    # get all dataset from list_days
    def get_all_data(list_paths, list_days, variable_name, wavelength, concat_dim):
        allData = []
        for path, day in zip(list_paths, list_days):
            try:
                file = xr.open_dataset(path)
                convert_timelist = convert_gpstime(file.time.values, day, convert=True)
                file = file.assign_coords(time = convert_timelist)
                if wavelength is None:
                    data = file[variable_name]#.dropna(dim='time', how='all')
                else:
                    data = file[variable_name].sel(wavelength=wavelength)#.dropna(dim='time', how='all')

                data = data.dropna(dim='time', how='all')
                allData.append(data)
            except FileNotFoundError:
                print('Cannot found filepath')
                pass
        
        if len(allData) == 0:
            print('Cannot concatenate')
            return 1
        else:
            allData = xr.concat(allData, dim=concat_dim)   
            return allData
    
    list_paths = get_path(list_days, configs)
    print(f'List of files: {list_paths}')
    time_name = configs['variables_name']['time']
    if (variable_name == 'SR'):
        ATB = get_all_data(list_paths, list_days, configs['variables_name']['ATB'], wavelength, time_name)
        AMB = get_all_data(list_paths, list_days, configs['variables_name']['AMB'], wavelength, time_name)
        dataset_generated = ATB/AMB
    else:
        dataset_generated = get_all_data(list_paths, list_days, configs['variables_name'].get(variable_name), wavelength, time_name)
    return dataset_generated
        
        
def generate_feature(data, features_data): 
    # input all data needed to generate on feature       
    def add_feature(data_before, adding):
        '''
        function is used to add features create test/train/validation dataset
        '''
        if (len(data_before.shape)<2):
            data_before = data_before.reshape(-1,1)
        if (len(adding.shape)<2):
            adding = adding.reshape(-1,1)

        print(f'Updating shape: {data_before.shape}, {adding.shape}')        
        data_after = np.hstack((data_before, adding))
        print(f'shape of data after adding feature: {data_after.shape}')
        if data_after.shape[1]>1:
            return data_after
        else:
            print('Error')
            return 1
        
    nb_features = len(features_data)
    if nb_features == 0:
        print('Any feature to adding')
        data = data.reshape(-1,1)
    else:
        for i in range(len(features_data)):
            print(f'add feature numero {i}')
            data = add_feature(data, features_data[i])
    return data

class clean_data :
    def clean_data_target(data, target):
        '''
        Use to clean Nan/Inf or negative values of data
        '''
        print(f'Before: data = {data.shape}, target = {target.shape}')
        # data 
    #     data = pd.DataFrame(data)
        mask_data = np.isfinite(data).all(1)
        # target
    #     target = pd.DataFrame(target)
        mask_target = np.isfinite(target).all(1)
        # intersection
        mask = np.logical_and(mask_data, mask_target)
        print(mask)
    #     mask = np.logical_and(np.isfinite(X).all(1), np.isfinite(Y.values.ravel()))
        print(f'shape of mask array {mask.shape}')
        new_data = data[mask, :]
        new_target = target[mask]
        print(f'After : new_data = {new_data.shape}, new_target = {new_target.shape}')
        return new_data, new_target, mask 

    def recover_data(raw_data, clean_data, no_masked_pos): 
        '''
        Input: 
            - raw_data: data before cleaning and array in 2D
            - clean_data: data after cleaning by clean_data_target function, in 1D
            - no_masked_pos: position of validated data after cleaning processing, array in 1D
        
        Output: 
            - recovered_data: clean_data what refill all masked positions
        '''
        # initiate data ravel & fill all by NaN
        recovered_data = np.full(raw_data.values.ravel().shape, np.nan)
        
        # fill validated position of data 
        recovered_data[np.where(no_masked_pos)] = clean_data
        
        # reshape new data
        recovered_data = recovered_data.reshape(raw_data.shape)

        # transform en xarray 
        recovered_data = xr.DataArray(recovered_data, coords = raw_data.coords)
        return recovered_data

'''
MACHINE LEARNING PROCESSING
'''
def generating_dataset_processing(list_days_random, configs, variable_chosen, recover=False):
    '''
    this processing can apply for generating training dataset also testing dataset 
    '''
    # before adding features
    data_train = generate_data(list_days_random, configs, 355, variable_chosen)
    target_train = generate_data(list_days_random, configs, 532, variable_chosen)
    print(target_train)
    target_train = target_train.sel(time= np.intersect1d(data_train.time, target_train.time))


    altitude_2d = np.tile(data_train.altitude.values, (data_train.time.shape[0], 1))
    altitude = data_train.altitude.values
    #feature 2
    from tqdm import tqdm
    X3 = np.zeros(data_train.shape)

    for j in tqdm(range(altitude.shape[0]-2, 0, -1)):
        delta_z = (altitude_2d[:,j] - altitude_2d[:,j-1])
        X3[:,j] = np.nansum([X3[:,j+1], (data_train[:,j].values*delta_z)], axis=0)
    #     print(j, X3[40:50,j])
    # for t in tqdm(range(train355.time.shape[0])):
    #     for j in (range(altitude.shape[0]-2, 0, -1)):
    #         X3[t,j] = np.nansum(X3[t, j+1] + train355[t,j]*(altitude[j+1] - altitude[j]))
            
        
    features= {
        0 : X3.ravel()
        # 0 : altitude_2d.ravel(),
        # 1 : X3.ravel()
    }
    new_data_train = generate_feature(data_train.values.ravel(), features)

    features={}
    new_target_train = generate_feature(target_train.values.ravel(), features)

    clean_data_train, clean_target_train, idmask = clean_data.clean_data_target(new_data_train, new_target_train)
    generated_dataframe = pd.DataFrame(np.concatenate([new_data_train, new_target_train], axis=1),
                               columns=['sr355', 'sr355_integrated', 'sr532']) #'alt', 
    if recover :
        recovered_clean_data_train = clean_data.recover_data(data_train, clean_data_train[:,0], idmask)
        recovered_clean_target_train = clean_data.recover_data(target_train, clean_target_train[:,0], idmask)
        return clean_data_train, clean_target_train, recovered_clean_data_train, recovered_clean_target_train, idmask
    else:
        return clean_data_train, clean_target_train, idmask

def build_learning_model(data_train, target_train, preprocessing=False):
    if preprocessing : 
        # training with preprocessing
        # ---------------------------
        model_loaded = make_pipeline(StandardScaler(), DecisionTreeRegressor())
        name_model_loaded = 'tree_3f_preprocessed.sav'

        # from sklearn.model_selection import GridSearchCV
        # param_grid = {
        #     'max_depth' : (3, 7, 15, 30),
        #     'min_samples_leaf' : (5, 10, 15, 20, 30),
        # }

        # model_grid_search = GridSearchCV(
        #     model_loaded, param_grid=param_grid, n_jobs=2, cv=10
        # )
        # model_grid_search.fit(clean_train355, clean_train532)
        # model_grid_search.best_params_
    else:
        # training without preprocessing
        # ------------------------------
        model_loaded = DecisionTreeRegressor() #max_depth=7, min_samples_leaf=15
        name_model_loaded = 'tree_3f.sav'

    model_loaded.fit(data_train, target_train)
    return model_loaded, name_model_loaded

def prediction_processing(data_test, model_train): 
    result_from_testing = model_train.predict(data_test)
    return result_from_testing