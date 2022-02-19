import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('cumulative.csv')

##print(df)

# Drop unwanted columns (categorical or null).
df.drop(['rowid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname'], axis='columns', inplace=True)

# Remove all columns with NaN values.
df. dropna(inplace=True)


independent_variables = []

##print(df)

dependent_variable = 'koi_score'

independent_variables = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 
'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 
'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 
'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 
'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 
'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 
'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']

X = df[independent_variables]
y = df[dependent_variable]

# Split train and test data 20/80.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.20, random_state = 1)

##print(X_train)

# Create linear model
import statsmodels.api as sm
lm = sm.OLS(y_train, X_train).fit()

print(lm.summary())

y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print('MAE = ', mae, '\n')
print('RMSE = ', rmse, '\n')
