# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# %%
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # dataviz
import seaborn as sns # dataviz
from pandas.plotting import scatter_matrix

Rental= pd.read_csv("./dataset/immo_data.csv")

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
Rental.describe() #shows a summary of the numerical attributes


# %%
Berlin=Rental.loc[Rental["regio2"]=='Berlin']
print(Berlin)


# %%
corr_matrix = Berlin.corr()
corr_matrix["totalRent"].sort_values(ascending=False)


# %%
Berlin.corr()["baseRent"].sort_values(ascending=False)


# %%
Berlin.hist(bins=50, figsize=(15,15))
plt.show()


# %%
attributes = ["baseRent","totalRent","livingSpace", "serviceCharge", "noRooms","heatingCosts","picturecount"]
scatter_matrix(Berlin[attributes], figsize=(16, 12))
scatter_matrix


# %%
Berlin["totalRent"].describe()


# %%
plt.plot(np.log(Berlin["totalRent"]))


# %%
Berlin['totalRent'].hist(bins=30, range=(100,4000), grid=True, color='#86bf91')
plt.title('Distribution of Base Rents')
plt.xlabel('Total Rent')
plt.ylabel('Count')


# %%
Berlin.plot(kind="scatter", x="livingSpace", y="baseRent", alpha=0.1)


# %%
m=Berlin.groupby(['regio3'])['baseRent'].mean()
m.sort_values()


# %%
#droping initial columns
cols_to_drop = ["telekomHybridUploadSpeed", "picturecount", "telekomUploadSpeed",
                "geo_bln", "houseNumber", "geo_krs", "geo_plz", "regio3", "description",
                "facilities"]

Berlin = Berlin.drop(cols_to_drop, axis=1)

#Columns with several NULL entries are dropped too.

Berlin.isna().sum()

#filter columns for berlin
Berlin = Berlin[Berlin["regio2"]=="Berlin"]

#sorting and re_indexing regarding to the price
Berlin = Berlin.sort_values(by=['totalRent'])
Berlin = Berlin.reset_index(drop=True)

#filter some columns between specific amount of values
Berlin = Berlin.query("totalRent >= 100").query("totalRent<10000")
Berlin = Berlin.query("baseRent >= 100").query("baseRent<10000")
Berlin = Berlin.query("livingSpace >= 10").query("livingSpace<400")
Berlin = Berlin.query("noRooms >= 0").query("noRooms<15")




# Replacing columns with f/t with 0/1
Berlin.replace({False: 0, True: 1}, inplace=True)


# %%
#make a single binary variable to indicate if the apartment is refurbished/new
Berlin['refurbished'] = (Berlin.condition == 'refurbished') | (Berlin.condition == 'first_time_use') | (Berlin.condition == 'mint_condition') | (Berlin.condition == 'fully_renovated') | (Berlin.condition == 'first_time_use_after_refurbishment')

#make a binary variable to indicate if the rental property has good interior
Berlin['greatInterior'] = (Berlin.interiorQual == 'sophisticated') | (Berlin.interiorQual == 'luxury')

#make a binary variable to indicated if the rental property has good heating
Berlin['goodHeating'] = (Berlin.heatingType == 'central_heating') | (Berlin.heatingType == 'floor_heating') | (Berlin.heatingType == 'self_contained_central_heating')

#make a binary variable to identify rental ads from last year to factor in any inflationary effects.
Berlin['2018_ads'] = (Berlin.date == 'Sep18')

#transform totalRent into log(totalRent) to get a better distribution + better interpretive quality
Berlin['logRent'] = np.log(Berlin['totalRent'])


# %%
y_var = ['logRent']
X_var = ['balcony', 'hasKitchen', 'cellar', 'livingSpace', 'noRooms', 'garden',
         'refurbished', 'greatInterior', 'newlyConst',
         '2018_ads', 'lift']


#Berlin[X_var].replace({False: 0, True: 1}, inplace=True)
y = Berlin[y_var].values
X = Berlin[X_var].values

#print(X)
#print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=42)


# %%
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linearregression(xtrain, ytrain, xtest, ytest):
    linreg = LinearRegression()
    linreg.fit(xtrain, ytrain)
    y_pred = linreg.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred))

linearregression(X_train, y_train, X_test, y_test)


# %%

#RANDOM FOREST
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

#Best hyperparamters from the Random Search:
#minsamleaf: 30, maxfeat: 11, maxdepth: 24 

def randomforestreg(msl, mf, md, xtrain, ytrain, xtest, ytest):
    rfr_best = RandomForestRegressor(n_estimators=70, random_state=1111,
                                     max_depth=md, max_features=mf, min_samples_leaf=msl)
    rfr_best.fit(xtrain,ytrain)
    y_pred_rfr = rfr_best.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred_rfr))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred_rfr))
    
randomforestreg(30, 11, 24, X_train, y_train, X_test, y_test)


# %%
#GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingRegressor

#Best hyperparameters from Random Search:
#maxdepth: 16, minsamleaf: 117, n: 73, maxfeat: 10, lr: 0.07
def gradientboostingmachine(md, msl, n, mf, lr, xtrain, ytrain, xtest, ytest):
    gbm_best = GradientBoostingRegressor(n_estimators=n, random_state=1111,
                                         max_depth=md, max_features=mf, 
                                         min_samples_leaf=msl, learning_rate=lr
                                         )
    gbm_best.fit(xtrain, ytrain)
    y_pred_gbm = gbm_best.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred_gbm))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred_gbm))
    
gradientboostingmachine(16, 117, 73, 10, 0.07, X_train, y_train, X_test, y_test)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()


scores = cross_val_score(lin_reg, X_train, y_train,
                        scoring="neg_mean_squared_error", cv=10)

# find root mean squared error, scores is an array of negative numbers
rmse_scores = np.sqrt(-scores)

print("Mean:\t\t ", rmse_scores.mean(), "\nStandard Deviation:", rmse_scores.std())


# %%
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
forest_scores = cross_val_score(forest_reg, X_train, y_train,
                               scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)


# %%
print("Mean:\t\t ", forest_rmse_scores.mean(), "\nStandard Deviation:", forest_rmse_scores.std())


# %%
def stackedmodel(xtrain, ytrain, xtest, ytest):
    x_training, x_valid, y_training, y_valid = train_test_split(xtrain, ytrain,
                                                                test_size=0.5,
                                                                random_state=42)
    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=70, random_state=1111,
                                   max_depth=24, max_features=11, 
                                   min_samples_leaf=24)
    model3 = GradientBoostingRegressor(n_estimators=73, random_state=1111,
                                       max_depth=16, max_features=10, 
                                       min_samples_leaf=117, learning_rate=0.07)
    
    model1.fit(x_training, y_training)
    model2.fit(x_training, y_training)
    model3.fit(x_training, y_training)
    
    preds1 = model1.predict(x_valid)
    preds2 = model2.predict(x_valid)
    preds3 = model3.predict(x_valid)
    
    testpreds1 = model1.predict(xtest)
    testpreds2 = model2.predict(xtest)
    testpreds3 = model3.predict(xtest)
    
    stackedpredictions = np.column_stack((preds1, preds2, preds3))
    stackedtestpredictions = np.column_stack((testpreds1, testpreds2,
                                              testpreds3))
    
    metamodel = LinearRegression()
    metamodel.fit(stackedpredictions, y_valid)
    final_predictions = metamodel.predict(stackedtestpredictions)
    print('MAE:', metrics.mean_absolute_error(ytest, final_predictions))
    print('MSE:', metrics.mean_squared_error(ytest, final_predictions))

stackedmodel(X_train, y_train, X_test, y_test)


# %%



