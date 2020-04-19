
# general
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Geographical analysis
import geopandas as gpf #libspatialindex nees to be installed first
import json # library to handle JSON files
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import shapefile as shp
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import descartes

# accessibility analysis
import time
from pandana.loaders import osm
from pandana.loaders import pandash5

# modelling
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


#Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Set plot preference
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

print('Libraries imported.')




#Cleaning and Pre-Processing
raw_df = pd.read_csv('/home/Sina/ApartmentRentalOffersInBerlin/dataset/immo_data.csv')
print(f"The dataset contains {len(raw_df)} Apartment listing")
pd.set_option('display.max_columns', len(raw_df.columns)) # To view all columns
pd.set_option('display.max_rows', 100)
raw_df.head(3)


#droping initial columns
cols_to_drop = ["telekomHybridUploadSpeed", "picturecount", "telekomUploadSpeed",
                "geo_bln", "houseNumber", "geo_krs", "geo_plz", "regio3", "description",
                "facilities"]

df = raw_df.drop(cols_to_drop, axis=1)

#Columns with several NULL entries are dropped too.

df.isna().sum()

#filter columns for berlin
df = df[df["regio2"]=="Berlin"]

#sorting and re_indexing regarding to the price
df = df.sort_values(by=['totalRent'])
df = df.reset_index(drop=True)

#filter some columns between specific amount of values
df = df.query("totalRent >= 100").query("totalRent<10000")
df = df.query("baseRent >= 100").query("baseRent<10000")
df = df.query("livingSpace >= 10").query("livingSpace<500")
df = df.query("noRooms >= 0").query("noRooms<15")




# Replacing columns with f/t with 0/1
df.replace({False: 0, True: 1}, inplace=True)



# Plotting the distribution of numerical and boolean categories
df.hist(figsize=(20,20));