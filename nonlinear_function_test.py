'''
OBJECTIF 
-------------------------------------------

FIT CALIBRATED DATASET TO FIND A FUNCTION THAT CAN RE-BUILD SCATTERING RATIO IN 532nm 
AND SAVE A VERSION OF PREDICTED DATASET

-------------------------------------------
'''

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import matplotlib as mpl
import warnings 
warnings.filterwarnings("ignore")
import os
os.chdir('/homedata/nmpnguyen/ORACLES-ER2/Codes')
from outils import convert_gpstime
os.chdir('/homedata/nmpnguyen/comparaison/Codes')
from fonctions import dataset, conditions, get_file, check, conversion, plots

#-------------------------------------------------
# FITTING PROCESS
#-------------------------------------------------
# copy from notebooks

#-------------------------------------------------
# CONFIGURE RATIO SR532/SR355 IN FUNCTION OF BETA_PART_355 ON LOGARITHMIC RELATIONSHIP
#-------------------------------------------------
F = lambda X : 0.20704*np.log(X) + 4.8181

#-------------------------------------------------
# GET PREDICTED DATA
#-------------------------------------------------
'''
Input
	- MAINDIR OF BETA_PART_355 AND SR RATIO 
	- FILE NAME PATTERN 
Process:
	- PREDICT RATIO FROM FUNCTION F(BETA_PART_355)
	- PREDICT SR532 FROM NEW RATIO AND SR355
'''

def get_SR_ratio(filepath):
	return SR355, SR532, RATIO 

def get_beta_part(filepath):
	return BETA_PART_355, BETA_PART_532
	
class by_function
	def __init__(self, NAME, INPUT, FUNCTION)
		self.NAME = NAME
		self.INPUT = INPUT
		self.FUNCTION = FUNCTION

	def predict(self):
		PRED = F(INPUT)