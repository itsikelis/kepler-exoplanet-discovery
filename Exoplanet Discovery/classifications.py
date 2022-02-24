import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Sklearn: Data normalisation, regression models and metrics.
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

##  Begin Important Variable Definitions  ##

PCA_COMPONENTS = 29

##  End Important Variable Definitions  ##



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
    ['CANDIDATE', 'FALSE POSITIVE'], [1, 0], inplace=True)

# Put (in)dependent variables in dataframes.
X = df[independent_variables]
y = df[dependent_variable]

##  End Data Handling  ##



##  Begin PCA  ##

# Standardise data
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=PCA_COMPONENTS)
pca.fit(X_std)
X_std = pca.transform(X_std)

##  End PCA  ##



##  Begin Logistic Regression with PCA data.  ##

# Split test/train data 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, train_size=.30, random_state=1)

# Instantiate logistic regression model.
model = LogisticRegression(max_iter=10000)
model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate model's accuracy.
print('\n \n Normalisation Logistic Regression Results: ')
score = model.score(X_test, y_test)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(
    y_test, model.predict(X_test)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n",
      metrics.classification_report(y_test, model.predict(X_test)))

# Plot the ROC curve.
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

##  End Logistic Regression with PCA data.  ##


print('Continue with Logistic Regression using normalisation on data. Press any key...')
input()


##  Begin Logistic Regression with Normalisation ##
# Split test/train data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.30, random_state=1)

# Print the maximum mean value of the sets before normalising.
print('Maximum mean values of train/test sets before normalisation: ')
print('Train set: ', max(X_train.mean(axis=0)))
print('Test set: ', max(X_test.mean(axis=0)))

# Normalise train data.
sc_train = StandardScaler()                 # Instantiate scaler object.
X_train = sc_train.fit_transform(X_train)   # Normalise data in X_train.

# Normalise test data.
sc_test = StandardScaler()                  # Instantiate scaler object.
X_test = sc_test.fit_transform(X_test)      # Normalise data in X_test.

# Print the maximum mean value of the sets after normalising.
print('Maximum mean values of train/test sets before normalisation: ')
print('Train set: ', max(X_train.mean(axis=0)))
print('Test set: ', max(X_test.mean(axis=0)))

# Instantiate logistic regression model.
model = LogisticRegression(max_iter=10000)
model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate model's accuracy.
print('\n \n Normalisation Logistic Regression Results: ')
score = model.score(X_test, y_test)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(
    y_test, model.predict(X_test)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n",
      metrics.classification_report(y_test, model.predict(X_test)))

# Plot the ROC curve.
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right") 
plt.show()

##  End Logistic Regression with Normalisation ##
