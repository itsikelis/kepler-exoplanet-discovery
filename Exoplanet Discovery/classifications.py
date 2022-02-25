import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Sklearn: Data normalisation, regression models and metrics.
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

##  Begin Important Variable Definitions  ##

TRAIN_SIZE = .40
PCA_COMPONENTS = 20
NUM_TREES = 100
K_FOLD = 5

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

# Turn categoricals to ints in dependent variable.
df[dependent_variable].replace(
    ['CANDIDATE', 'FALSE POSITIVE'], [1, 0], inplace=True)

# Drop unwanted columns from dataframe.
df.drop(unwanted_cols, axis='columns', inplace=True)

# Remove all rows with NaN values.
df.dropna(inplace=True)

# Put (in)dependent variables in dataframes.
X = df[independent_variables]
y = df[dependent_variable]

##  Instantiate Plot object for ROC curves.  ##
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

##  End Data Handling  ##



##  Begin PCA  ##

# Standardise data
X_std_pca = StandardScaler().fit_transform(X)

pca = PCA(n_components=PCA_COMPONENTS)
pca.fit(X_std_pca)
X_std_pca = pca.transform(X_std_pca)

##  End PCA  ##



##  Begin Logistic Regression with PCA data.  ##

# Split test/train data 70/30 (Using X_std.)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_std_pca, y, train_size=TRAIN_SIZE, random_state=1)

# Instantiate logistic regression model.
model = LogisticRegression(max_iter=10000)
model = model.fit(X_train_pca, y_train_pca)

y_train_pred = model.predict(X_train_pca)
y_test_pred = model.predict(X_test_pca)

# Calculate model's accuracy.
print('\n \n Normalisation Logistic Regression Results: ')
score = model.score(X_test_pca, y_test_pca)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(
    y_test_pca, model.predict_proba(X_test_pca)[:, 1])
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(
    y_test_pca, model.predict(X_test_pca)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n",
      metrics.classification_report(y_test_pca, model.predict(X_test_pca)))

# Plot the ROC curve.
plt.plot(fpr, tpr, label='ROC curve (PCA) (area = %0.2f)' % roc_auc)

##  End Logistic Regression with PCA data.  ##


print('Continue with Logistic Regression using normalisation on data. Press any key...')
input()


##  Begin Logistic Regression with Normalisation ##
# Split test/train data 80/20 (Using X.)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_SIZE, random_state=1)

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
plt.plot(fpr, tpr, label='ROC curve (Norm) (area = %0.2f)' % roc_auc)

##  End Logistic Regression with Normalisation ##

print('Continue with Support Vector Machines using normalisation on data. Press any key...')
input()

##  Begin SVM Classification  ##

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)

# Calculate model's accuracy.
score = svm.score(X_test, y_test)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y_test, svm.predict(X_test))
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(y_test, svm.predict(X_test)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n",metrics.classification_report(y_test, svm.predict(X_test)))

# Plot the ROC curve.
plt.plot(fpr, tpr, label='ROC curve (SVM) (area = %0.2f)' % roc_auc)

##  End SVM Classification  ##

print('Continue with k-Nearest Neighbours using normalisation on data. Press any key...')
input()

##  Begin kNN  ##

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

# Calculate model's accuracy.
score = knn.score(X_test, y_test)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y_test, knn.predict(X_test))
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(
    y_test, knn.predict(X_test)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n",
      metrics.classification_report(y_test, knn.predict(X_test)))

# Plot the ROC curve.
plt.plot(fpr, tpr, label='ROC curve (kNN) (area = %0.2f)' % roc_auc)

## End kNN  ##

print('Continue with Random Forests without using normalisation on data. Press any key...')
input()

##  Begin Random Forests  ##

# Resplit data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_SIZE, random_state=1)

forest = RandomForestClassifier(n_estimators=NUM_TREES, criterion='gini').fit(X_train, y_train)
forest.fit(X_train, y_train)

# Calculate model's accuracy.
print('\n \n Random Forest Results: ')
score = forest.score(X_test, y_test)
print('Model Accuracy (%):', 100*score)

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y_test, forest.predict_proba(X_test)[:, 1])
# Calculate AUC score.
roc_auc = metrics.auc(fpr, tpr)

# Print Confusion Matrix and Classification Report.
print("Confusion Matrix :\n", metrics.confusion_matrix(y_test, forest.predict(X_test)))
print("Classification Report (1: Candidate Exoplanet, 0: False Positive Measurement ) :\n", metrics.classification_report(y_test, forest.predict(X_test)))

plt.plot(fpr, tpr, label='ROC curve (RF) (area = %0.2f)' % roc_auc)
plt.legend(loc='best')
plt.show()

##  End Random Forests  ##