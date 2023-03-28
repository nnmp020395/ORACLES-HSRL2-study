#!/bin/bash

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
import DecisionTree_fonctions as train_test_days_split

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

list_days = ['20160819', '20160823', '20160826', '20160912', '20160916', '20160918', '20160920', '20160922', '20160924']

days_train, days_test = train_test_days_split(list_days, test_size=0.5, random=True)
print(days_train, days_test)

