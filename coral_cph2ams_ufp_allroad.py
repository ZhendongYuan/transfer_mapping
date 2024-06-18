#@author: Zhendong Yuan at 18 june 2024. 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import pearsonr
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from adapt.feature_based import CORAL

source_data =pd.read_csv('your source data', sep=',')
validation_data = pd.read_csv('your validation data', sep=',')
target_data = pd.read_csv("your target data")

# data manipulation on source data
Xs = source_data.values[:,3:]
Xs = preprocessing.normalize(Xs)
ys = source_data.values[:,1]

# check the mean of source measurements
print("mean of source measured: ")
print(ys.mean())

# data manipulation on target and validation data
Xt = target_data.values[:,1:]
Xt = preprocessing.normalize(Xt)

Xval = validation_data.values[:,3:]
Xval = preprocessing.normalize(Xval)
yval = validation_data.values[:,2]

print("mean of validation: ")
print(yval.mean())

# run the transfer learing methods
model = CORAL(Ridge(alpha=0.2), lambda_= 1e-6, Xt=Xt, random_state=0)
model.fit(Xs, ys)

# prediction accuracy be validation data
yval_pred = model.predict(Xval)

mae_coral = mean_absolute_error(yval, yval_pred)
print("MAE of coral:", mae_coral)

s_pmae = (mae_coral*len(yval))/sum(yval)
print("percentage MAE of coral:", s_pmae)

rmse_coral = math.sqrt(mean_squared_error(yval, yval_pred))
print("RMSE of coral:", rmse_coral)

s_prmse=(rmse_coral*len(yval))/sum(yval)
print("percentage RMSE of coral:", s_prmse)
temp = yval_pred.flatten()
corr_coeff, _ = pearsonr(yval, temp)
r2_coral = corr_coeff ** 2
print("R2 of coral:", r2_coral)

# save performance
acc_table = np.array([r2_coral, mae_coral, rmse_coral,s_pmae,s_prmse])
pd.DataFrame(acc_table).to_csv(
            "save performance table")
pd.DataFrame(yval_pred).to_csv(
            "save predicted value at validation sites")