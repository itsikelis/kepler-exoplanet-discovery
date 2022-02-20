import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

# Read csv and store to pandas dataframe.
df = pd.read_csv('Exoplanet Discovery/cumulative.csv')

# List of unwanted dataframe columns(mostly categoricals, indexes and null columns).
unwanted_cols = ['kepid', 'rowid', 'kepoi_name', 'kepler_name',
                  'koi_pdisposition', 'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname']

# List of independent variables.
independent_variables = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
                         'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
                         'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
                         'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq',
                         'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num',
                         'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
                         'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']

# Dependent variable.
dependent_variable = 'koi_disposition'

# Drop unwanted columns from dataframe.
df.drop(unwanted_cols, axis='columns', inplace=True)

# Remove all rows with NaN values.
df. dropna(inplace=True)

# Replace categorical column koi_disposition with boolean values
df1 = df.koi_disposition.str.get_dummies()

##print(df)

# Put (in)dependent variables in dataframes.
X = df[independent_variables]
y = df[dependent_variable]

# Split test/train data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.20, random_state=1)

print(np.unique(y))

# Instantiate logistic regression model.
model = LogisticRegression()
model = model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate model's accuracy
score = model.score(X_test, y_test)

print(score)



