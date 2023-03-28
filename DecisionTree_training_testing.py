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

import sys
sys.path.append('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Codes/')
import DecisionTree_fonctions as DTfct

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
    'output_dir' : Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/Products/'),
    'random_version' : None #random.randint(0,10000) #
}

list_days = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']

days_train, days_test = DTfct.train_test_days_split(list_days, test_size=0.3, random=True)
print('LIST DAYS FOR TRAINING\n', days_train)
print('LIST DAYS FOR TESTING\n', days_test)


for i in range(0, len(list_days), 3):
    days_test = np.array(list_days[i:i+3])
    days_train = np.concatenate((np.setdiff1d(list_days, days_test), np.setdiff1d(days_test, list_days)))
    print('LIST DAYS FOR TRAINING\n', days_train)
    print('LIST DAYS FOR TESTING\n', days_test)
    configs['random_version'] = f'NA{i}'

    # prepare training dataset and training target 
    print('PREPARE TRAINING DATASET\n')
    training_dataset, training_target, _ = DTfct.generating_dataset_processing(days_train, configs, 'SR', recover=False)

    # buil model learning
    print('BUILD LEARNING MODEL')
    model_learned, model_learned_name = DTfct.build_learning_model(training_dataset, training_target, preprocessing=False)

    # prepare testing dataset from list days of testing
    print('PREPARE TESTING DATASET\n')
    for day in days_test:
        # prepare testing dataset and testing target
        testing_dataset, testing_target, recovered_testing_dataset, recovered_testing_target, idmask = DTfct.generating_dataset_processing([day], configs, 'SR', recover=True)

        # prediction testing dataset
        predictive_data = DTfct.prediction_processing(testing_dataset, model_learned)

        # recover testing & predictive data, then saving new datafile included data test, target test and data predict
        recovered_predictive_data = DTfct.clean_data.recover_data(recovered_testing_target, predictive_data, idmask)

        # saving
        ds = xr.merge([xr.DataArray(recovered_testing_dataset, name='SR355_measured', coords = recovered_testing_dataset.coords), 
                   xr.DataArray(recovered_testing_target, name='SR532_measured', coords = recovered_testing_target.coords),
                   xr.DataArray(recovered_predictive_data, name = 'SR532_predicted', coords = recovered_predictive_data.coords)],
                  compat='override')
        ds.to_netcdf(Path(configs['output_dir'], f'{model_learned_name}-HSRL2-ER2-{day}_v{configs["random_version"]}.nc'))
        




