import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

# statsmodels: Multivariate regression.
import statsmodels.api as sm

# Sklearn: Data normalisation, regression models and metrics.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

## Begin Data Handling  ##

# Read csv and store to pandas dataframe.
df = pd.read_csv('Exoplanet Discovery/cumulative.csv')

# List of unwanted dataframe columns(categoricals, system flags, indexes and null columns).
unwanted_cols = ['rowid', 'kepid', 'kepoi_name', 'kepler_name',
                 'koi_disposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
                 'koi_fpflag_ec', 'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname']

# Declare dependent and independent variables.
dependent_variable = 'koi_pdisposition'

independent_variables = ['koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
                         'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
                         'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq',
                         'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num',
                         'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
                         'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']


# Drop unwanted columns from dataframe.
df.drop(unwanted_cols, axis='columns', inplace=True)

# Remove all rows with NaN values.
df.dropna(inplace=True)

# Turn categoricals to ints in dependent variable.
df[dependent_variable].replace(
    ['CANDIDATE', 'FALSE POSITIVE'], [1, 2], inplace=True)

# Put (in)dependent variables in dataframes.
X = df[independent_variables]
y = df[dependent_variable]

##  End Data Handling  ##

##  Begin Multivariate regression  ##

## Multivariate Regression ##

# Split test/train data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.20, random_state=1)

# Train linear model
lm = sm.OLS(y_train, X_train).fit()

print(lm.summary())

y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

# Calculate Mean Abolute and Root Mean Square Errors.
mae = metrics.mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print('MAE = ', mae, '\n')
print('RMSE = ', rmse, '\n')
