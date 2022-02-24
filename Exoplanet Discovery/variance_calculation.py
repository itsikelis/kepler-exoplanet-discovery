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

#########PCA###########

# Standardise data
X_std = StandardScaler().fit_transform(X)

# Create covariance matrix.
cov_mat = np.cov(X_std.T)
#print('Covariance matrix \n%s' % cov_mat)

# Calculate covariance eigenvalues.
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#print('Eigenvectors \n%s' % eig_vecs)
#print('\nEigenvalues \n%s' % eig_vals)

# Sort eigenvalues in decreasing order.
#eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
tot = sum(eig_vals)

# Calculate variance for every component.
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

# Calculate cumulative variance.
cum_var_exp = []
for i, val in enumerate(var_exp):
    if i == 0:
        cum_var_exp.append(val)
        print(cum_var_exp)
    else:
        cum_var_exp.append(cum_var_exp[i-1] + val)

# Plot diagram showing individual(decreasing order) and cummulative variance.
plt.figure(figsize=(10, 8))
plt.bar(range(36), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(36), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio (%)')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.tight_layout()
plt.show()