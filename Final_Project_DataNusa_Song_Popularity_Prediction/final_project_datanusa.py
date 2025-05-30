# -*- coding: utf-8 -*-
"""Final_Project_DataNusa.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sKFUP67uA-6uObVSLCNlnnSSVeM9JRQk

# Import relevant libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# Standard libraries for data analysis:
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from scipy.stats import norm, skew

import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,8]
from IPython.display import display

# sklearn modules for data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# sklearn modules for Model Selection
from sklearn import (
    svm, tree, linear_model, neighbors
)
from sklearn import (
    naive_bayes, ensemble,
    discriminant_analysis, gaussian_process
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# sklearn modules for Model Evaluation & Improvement
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    f1_score, precision_score,
    recall_score, fbeta_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, ShuffleSplit, KFold
)

# Standard libraries for data visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# %matplotlib inline
color = sns.color_palette()
import matplotlib.ticker as mtick
from IPython.display import display

# Miscellaneous Utilitiy Libraries
import random
from datetime import datetime
import string

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings
warnings.filterwarnings('ignore')

"""# Import dataset"""

# Install dependencies as needed:
!pip install kagglehub[pandas-datasets]

!mkdir ~/.kaggle
!mkdir -p ~/.config/kaggle  # Create the directory if it doesn't exist
!mv ~/.kaggle/kaggle.json ~/.config/kaggle/  # Move the kaggle.json file
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!echo '{"username":"finaltestds","key":"7c3e51f7230eadd8335720c4b69a0248"}' > ~/.kaggle/kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files('yasserh/song-popularity-dataset', path='download', unzip=True)

# Import the dataset
df = pd.read_csv('/content/download/song_data.csv')

# Read Dataset
display(df.head())

"""# Data Exploration


"""

df.describe()

df.dtypes

# Recheck Column Datatypes and Missing Values:

data_desc = (
    df
    .columns
    .to_series()
    .groupby(df.dtypes)
    .groups.items()
)

for dtype, columns in data_desc:
    print(f"{dtype}:")
    for col in columns:
        print(f"  - {col}")

# check missing values
df.isna().any()

df.drop(['song_name'], axis=1, inplace=True)
display(df.head())

target = 'song_popularity'
features = [i for i in df.columns if i not in [target]]

original_df = df.copy(deep=True)

print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))

df.info()

#Checking number of unique rows in each feature

nu = df[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0;

for i in range(df[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

"""# EDA"""

plt.figure(figsize=[8,4])
sns.distplot(df[target], color='blue',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution')
plt.show()

#Visualising the categorical features
import math # Import the math module

print('\033[1mVisualising Categorical Features:'.center(100))

n=2
plt.figure(figsize=[20,3*math.ceil(len(cf)/n)])

for i in range(len(cf)):
    if df[cf[i]].nunique()<=8:
        plt.subplot(math.ceil(len(cf)/n),n,i+1)
        sns.histplot(df[cf[i]])
    else:
        plt.subplot(2,1,2)
        sns.histplot(df[cf[i]])

plt.tight_layout()
plt.show()

#Visualising the numeric features

print('\033[1mNumeric Features Distribution'.center(100))

n=5

clr=['r','g','b','g','b','r']

plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    sns.distplot(df[nf[i]],hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255,255,255])/255))
plt.tight_layout()
plt.show()

plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()

"""# Data Preprocessing"""

#Removal of any Duplicate rows (if any)

counter = 0
rs,cs = original_df.shape

df.drop_duplicates(inplace=True)

df.shape

#Cek data kosong

df_empty = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
df_empty['Percentage'] = round(df_empty['Total Null Values']/df.shape[0],3)*100
print(df_empty)

#Convert categorical Columns to Numeric

df1 = df.copy()

df_convert = df_empty[df_empty['Percentage']!=0].index.values
df_convert1 = [i for i in cf if i not in df_convert]
#One-Hot Binay Encoding
oh=True
dm=True
for i in df_convert1:
    #print(i)
    if df1[i].nunique()==2:
        if oh==True: print("\033[1mOne-Hot Encoding on features:\033[0m")
        print(i);oh=False
        df1[i]=pd.get_dummies(df1[i], drop_first=True, prefix=str(i))
    if (df1[i].nunique()>2 and df1[i].nunique()<17):
        if dm==True: print("\n\033[1mDummy Encoding on features:\033[0m")
        print(i);dm=False
        df1 = pd.concat([df1.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df1[i], drop_first=True, prefix=str(i)))],axis=1)

df1.shape

#Removal of outlier:

df2 = df1.copy()

#features1 = [i for i in features if i not in ['CHAS','RAD']]
features1 = nf

for i in features1:
    Q1 = df2[i].quantile(0.25)
    Q3 = df2[i].quantile(0.75)
    IQR = Q3 - Q1
    df2 = df2[df2[i] <= (Q3+(1.5*IQR))]
    df2 = df2[df2[i] >= (Q1-(1.5*IQR))]
    df2 = df2.reset_index(drop=True)
display(df2.head())
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(df1.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df2.shape[0]))

#Final Dataset size after performing Preprocessing

df_final = df2.copy()
df.columns=[i.replace('-','_') for i in df.columns]
colors = ['b' , 'g']

plt.title('Final Dataset')
plt.pie([df_final.shape[0], original_df.shape[0]-df_final.shape[0]], radius = 1, labels=['Retained','Dropped'], counterclock=False,
        autopct='%1.2f%%', pctdistance=0.5, explode=[0,0.1], shadow=True , colors=colors, startangle=45)

plt.show()

print(f'\n\033[1mInference:\033[0m After the cleanup process, {original_df.shape[0]-df_final.shape[0]} samples were dropped, \
while retaining {round(100 - (df_final.shape[0]*100/(original_df.shape[0])),2)}% of the data.')

"""# Data Manipulation

## Target Variabel
"""

#  Log transformation of the target variable "song_popularity"
df_final['song_popularity'] = np.log(df_final['song_popularity'])

#Plot the distibution before and after transformation
fig, axes = plt.subplots(1,2,figsize=(10,4))
fig.suptitle("Target Variable Distribution of 'song_popularity' before and after")


#Before log transformation
p = sns.histplot(df_final['song_popularity'],
                 ax=axes[0], color='skyblue',
                 edgecolor="black", kde=True,
                 linewidth=2,
                 bins=30)
p.set_xlabel("song_popularity", fontsize=12)
p.set_ylabel("Effectivity", fontsize=12)
p.set_title("Before Log Transformation")

#After log transformation
q = sns.histplot(df_final['song_popularity'],
                 ax=axes[1], color='green',
                 edgecolor="black", kde=True,
                 linewidth=2,
                 bins=30)
q.set_xlabel("song_popularity", fontsize=12)
q.set_ylabel("Effectivity", fontsize=12)
q.set_title("After Log Transformation")

"""## Split data into train and test set and Standardization"""

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Select features for scaling (excluding the target variable 'song_popularity')
features = ['song_duration_ms', 'acousticness', 'danceability', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode',
            'speechiness', 'tempo', 'time_signature', 'audio_valence']

# Extract the target variable
target = 'song_popularity'

# Extract the features (X) and the target (y)
df_final = df.copy()
X = df_final [features]
y = df_final[target]

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the features and transform them
df_Song_data_scaled = pd.DataFrame(scaler.fit_transform(df_final[features]), columns=features, index=df_final.index)
df_Song_data_scaled['song_popularity'] = df_final['song_popularity']


# Features (X)
X = df_Song_data_scaled[features]
print(X.shape)

# Target (y)
y = df_Song_data_scaled.loc[:, "song_popularity"]
print(y.shape)

# Apply scaling to the features
# Assign df_final to song_data_scaled
song_data_scaled = df_final.copy()
song_data_scaled[features] = scaler.fit_transform(df_final[features])


# Display the first few rows of the scaled dataset
song_data_scaled.head()

# Visualization on 'Song_Popularity'

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot Empiric Cummulative Distribution
sorted_data = y.sort_values(ignore_index=True)
axs[0].scatter(x=sorted_data.index, y=sorted_data)
axs[0].set_title('Distribution of Song_Popularity')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Song_Popularity')

# Plot histogram of 'SalePrice'
axs[1].hist(y, bins=30, color='blue', alpha=0.7)
axs[1].set_title('Histogram for Song_Ppoularity')
axs[1].set_xlabel('Song_Popularity')
axs[1].set_ylabel('Frequency')

plt.tight_layout

from sklearn.model_selection import train_test_split
# Split into X_train and X_test (by stratifying on y)
# Stratify on a continuous variable by splitting it in bins
# Create the bins.
bins = np.linspace(min(y) + 0.5, max(y) - 0.5, 5)
y_binned = np.digitize(y, bins)

#Check for bins with only one sample
unique, counts = np.unique(y_binned, return_counts=True)
for bin_val, count in zip(unique, counts):
    if count < 2:
        print(f"Warning: Bin {bin_val} has only {count} sample(s). Consider adjusting bins.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_binned if all(counts >= 2) else None, shuffle=True
)

print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
print(f"\nX_test:{X_test.shape}\ny_test:{y_test.shape}")

from sklearn import preprocessing
# Standardize the data
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

# The same standardization is applied for X_test
df_test_new = std_scale.transform(X_test)

# Convert X, y and test data into dataframe
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
df_test_new = pd.DataFrame(df_test_new, columns=X.columns)

y_train = pd.DataFrame(y_train)
y_train = y_train.reset_index().drop("index", axis=1)

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index().drop("index", axis=1)

"""## Backward Stepwise Regression"""

import statsmodels.api as sm # Import the statsmodels library

def backward_regression(X, y, initial_list=[],
                        threshold_in=0.01, threshold_out=0.05,
                        verbose=True):
    """ To select features with backward stepwise regression

    Args:
    x -- feature values
    y -- target variable
    initial_list -- features header
    threshold_in -- pvalue threshold of features to keep
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit() # Use sm.OLS and sm.add_constant
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if p-values is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    print(f"\nSelected Features:\n{included}")
    return included

# Application of the backward regression function on our training data
Selected_Features = backward_regression(X_train, y_train)

# Keep the selected features only
X_train = X_train.loc[:, Selected_Features]
X_test = X_test.loc[:, Selected_Features]
df_test_new = df_test_new.loc[:, Selected_Features]

"""## Variance Inflation Factor"""

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])] # Change 'VIF' to 'VIF Factor'

# VIF results in a table
vif["features"] = X_train.columns
vif.round(2)

# Select features with high VIF
high_vif_list = vif[vif["VIF Factor"] > 10]["features"].tolist()

if len(high_vif_list) == 0:
    # print empty list if low multicolinearity
    print(f"None of the features have a high multicollinearity")
else:
    # print list of features with high multicolinearity
    print(f"List of features with high multicollinearity: {high_vif_list}")

"""## Cook distance"""

X_constant = sm.add_constant(X_train)

model = sm.OLS(y_train, X_constant)
lr = model.fit()

# Cook distance
np.set_printoptions(suppress=True)

# Create an instance of influence
influence = lr.get_influence()

# Get Cook's distance for each observation
cooks = influence.cooks_distance

# Result as a dataframe
cook_df = pd.DataFrame({"Cook_Distance": cooks[0], "p_value": cooks[1]})
cook_df.head()

cook_df.shape

# Remove the influential observation from X_train and y_train
influent_observation = cook_df[cook_df["p_value"] < 0.05].index.tolist()
print(f"Influential observations dropped: {influent_observation}")

# Drop these obsrevations
X_train = X_train.drop(X_train.index[influent_observation])
y_train = y_train.drop(y_train.index[influent_observation])

"""# Modelling

### Models and metrics selection
"""

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Let's define a function for each metrics
# R²
def rsqr_score(test, pred):
    """Calculate R squared score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        R squared score
    """
    r2_ = r2_score(test, pred)
    return r2_


# RMSE
def rmse_score(test, pred):
    """Calculate Root Mean Square Error score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        Root Mean Square Error score
    """
    rmse_ = np.sqrt(mean_squared_error(test, pred))
    return rmse_


# Print the scores
def print_score(test, pred):
    """Print calculated score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        print the regressor name
        print the R squared score
        print Root Mean Square Error score
    """

    print(f"- Regressor: {regr.__class__.__name__}")
    print(f"R²: {rsqr_score(test, pred)}")
    print(f"RMSE: {rmse_score(test, pred)}\n")

# Define regression models
MLR = LinearRegression()
dtr = DecisionTreeRegressor()
ridge = Ridge()
lasso = Lasso(alpha= 0.001)
elastic = ElasticNet(alpha= 0.001)
rdf = RandomForestRegressor()
xgboost = XGBRegressor()
lgbm = LGBMRegressor()

# Train models on X_train and y_train
for regr in [MLR, dtr, ridge, lasso, elastic, rdf, xgboost, lgbm]:
    # fit the corresponding model
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # Print the defined metrics above for each classifier
    print_score(y_test, y_pred)

# Define a function to evaluate models

Model_Evaluation = pd.DataFrame(np.zeros([7,4]),
                                columns=['Train-R2', 'Test-R2',
                                         'Train-RMSE','Test-RMSE'])

# Standardize X_train and assign it to X_train_std if necessary
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)  # Create X_train_std

rc = np.random.choice(X_train_std.loc[:,X_train_std.nunique()>=50].columns.values,3,replace=False)

# Evaluating models
def Evaluate(n, pred1,pred2):

    print('{}Training Set Metrics{}'.format('-'*20, '-'*20))
    print('\nR-Squared (R²)                 :', r2_score(y_train, pred1))
    print('Root Mean Squared Error (RMSE) :', np.sqrt(mean_squared_error(y_train, pred1)))

    print('\n\n{}Testing Set Metrics{}'.format('-'*20, '-'*20))
    print('\nR-Squared (R²)                 :', r2_score(y_test, pred2))
    print('Root Mean Squared Error (RMSE) :', np.sqrt(mean_squared_error(y_test, pred2)))

    Model_Evaluation.loc[n,'Train-R2']   = r2_score(y_train, pred1)
    Model_Evaluation.loc[n,'Test-R2']    = r2_score(y_test, pred2)
    Model_Evaluation.loc[n,'Train-RMSE'] = np.sqrt(mean_squared_error(y_train, pred1))
    Model_Evaluation.loc[n,'Test-RMSE']  = np.sqrt(mean_squared_error(y_test, pred2))

    # Plot Actual vs Prediction
    print('\n\n{}Actual vs Prediction{}'.format('-'*20, '-'*20))
    plt.figure()
    plt.scatter(y_train,pred1)
    plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], 'r--')
    plt.title('Actual vs Prediction')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.show()



"""## Lasso Regression"""

from sklearn.model_selection import GridSearchCV

# Define hyperparameters
alphas = np.logspace(-0.1, 1.0, 10).tolist()

tuned_parameters = {"alpha": alphas}

# Create a Ridge regression model
lasso = Lasso()

# Create GridSearchCV object
lasso_cv = GridSearchCV(
    estimator=lasso,
    param_grid=tuned_parameters,
    cv=10,
    n_jobs=-1,
    verbose=1,
)

# fit the GridSearch on train set
lasso_cv.fit(X_train, y_train)

# print best params and the corresponding R²
print(f"Best hyperparameters: {lasso_cv.best_params_}")
print(f"Best R² (train): {lasso_cv.best_score_}")

# Lasso Regressor with the best hyperparameters
lasso_mod = Lasso(alpha=lasso_cv.best_params_["alpha"])

# Fit the model on train set
lasso_mod.fit(X_train, y_train)

# Predict on test set
y_pred = lasso_mod.predict(X_test)

# Print the model name and evaluation metrics
print(f"- {lasso_mod.__class__.__name__}")
print(f"R²: {rsqr_score(y_test, y_pred)}")
print(f"RMSE: {rmse_score(y_test, y_pred)}")

# Save the model results into lists
model_list = []
r2_list = []
rmse_list = []

model_list.append(lasso_mod.__class__.__name__)
r2_list.append(round(rsqr_score(y_test, y_pred), 4))
rmse_list.append(round(rmse_score(y_test, y_pred), 4))

from sklearn.linear_model import Lasso
lasso_mod = lasso_cv.best_estimator_
pred1 = lasso_mod.predict(X_train_std)
pred2 = lasso_mod.predict(X_test)

Evaluate(0, pred1, pred2)



"""## Ridge Regression"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Define hyperparameters
alphas = np.logspace(-0.1, 1.0, 10).tolist()

# Define the parameter grid
tuned_parameters = {"alpha": alphas}

# GridSearchCV
ridge_cv = GridSearchCV(Ridge(), tuned_parameters, cv=10, n_jobs=-1, verbose=1)

# Fit the GridSearch on the training set
ridge_cv.fit(X_train, y_train)

# Print best hyperparameters and the corresponding R²
print(f"Best hyperparameters: {ridge_cv.best_params_}")
print(f"Best R² (train): {ridge_cv.best_score_}")

# Ridge Regressor with the best hyperparameters
ridge_mod = Ridge(alpha=ridge_cv.best_params_["alpha"])

# Fit the model on train set
ridge_mod.fit(X_train, y_train)

# Predict on test set
y_pred = ridge_mod.predict(X_test)

print(f"- {ridge_mod.__class__.__name__}")
print(f"R²: {rsqr_score(y_test, y_pred)}")
print(f"RMSE: {rmse_score(y_test, y_pred)}")

# Save the model results into lists
model_list = []
r2_list = []
rmse_list = []

model_list.append(ridge_mod.__class__.__name__)
r2_list.append(round(rsqr_score(y_test, y_pred), 4))
rmse_list.append(round(rmse_score(y_test, y_pred), 4))
rmse_list = []

model_list.append(ridge_mod.__class__.__name__)
r2_list.append(round(rsqr_score(y_test, y_pred), 4))
rmse_list.append(round(rmse_score(y_test, y_pred), 4))

from sklearn.linear_model import Ridge
ridge_mod = ridge_cv.best_estimator_
pred1 = ridge_mod.predict(X_train_std)
pred2 = ridge_mod.predict(X_test)

Evaluate(1, pred1, pred2)

"""## XGBoost"""

from xgboost import XGBRegressor
import xgboost as xgb

# Define hyperparameters
tuned_parameters = {"max_depth": [3],
                    "colsample_bytree": [0.3, 0.7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 500]}

# GridSearch
xgbr_cv = GridSearchCV(estimator=XGBRegressor(),
                       param_grid=tuned_parameters,
                       cv=5,
                       n_jobs=-1,
                       verbose=1)

# fit the GridSearch on train set
xgbr_cv.fit(X_train, y_train)

# print best params and the corresponding R²
print(f"Best hyperparameters: {xgbr_cv.best_params_}\n")
print(f"Best R²: {xgbr_cv.best_score_}")

# XGB Regressor with the best hyperparameters
xgbr_mod = XGBRegressor(seed=20,
                        colsample_bytree=xgbr_cv.best_params_["colsample_bytree"],
                        learning_rate=xgbr_cv.best_params_["learning_rate"],
                        max_depth=xgbr_cv.best_params_["max_depth"],
                        n_estimators=xgbr_cv.best_params_["n_estimators"])

# Fit the model on train set
xgbr_mod.fit(X_train, y_train)

# Predict on test set
y_pred = xgbr_mod.predict(X_test)

print(f"- {xgbr_mod.__class__.__name__}")
print(f"R²: {rsqr_score(y_test, y_pred)}")
print(f"RMSE: {rmse_score(y_test, y_pred)}")

from xgboost import XGBRegressor
import xgboost as xgb

xgbr_mod = xgbr_cv.best_estimator_
pred1 = xgbr_mod.predict(X_train_std)
pred2 = xgbr_mod.predict(X_test)

Evaluate(2, pred1, pred2)

"""## LightGBM Regression


"""

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

# GridSearchCV
lgb_cv = GridSearchCV(lgb.LGBMRegressor(), param_grid, cv=10, n_jobs=-1, verbose=1)

# Fit the GridSearch on train set
lgb_cv.fit(X_train, y_train)

# Print best params and the corresponding R²
print(f"Best hyperparameters: {lgb_cv.best_params_}")
print(f"Best R² (train): {lgb_cv.best_score_}")

# LightGBM Regressor with the best hyperparameters
lgb_mod = lgb.LGBMRegressor(
    num_leaves=lgb_cv.best_params_['num_leaves'],
    learning_rate=lgb_cv.best_params_['learning_rate'],
    n_estimators=lgb_cv.best_params_['n_estimators']
)

# Fit the model on train set
lgb_mod.fit(X_train, y_train)

# Predict on test set
y_pred = lgb_mod.predict(X_test)

# Print the model name and evaluation metrics
print(f"- {lgb_mod.__class__.__name__}")
print(f"R²: {rsqr_score(y_test, y_pred)}")
print(f"RMSE: {rmse_score(y_test, y_pred)}")

# Save the model results into lists
model_list = []
r2_list = []
rmse_list = []

model_list.append(lgb_mod.__class__.__name__)
r2_list.append(round(rsqr_score(y_test, y_pred), 4))
rmse_list.append(round(rmse_score(y_test, y_pred), 4))

import lightgbm as lgb

lgb_mod = lgb_cv.best_estimator_
pred1 = lgb_mod.predict(X_train_std)
pred2 = lgb_mod.predict(X_test)

Evaluate(3, pred1, pred2)



"""## Elasticnet"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
alphas = np.logspace(-0.1, 1.0, 10).tolist()
l1_ratios = np.arange(0.1, 1.0, 0.2)

tuned_parameters = {"alpha": alphas, "l1_ratio": l1_ratios}

# GridSearchCV
elastic_cv = GridSearchCV(ElasticNet(), tuned_parameters, cv=10, n_jobs=-1, verbose=1)

# Fit the model on trains set
elastic_cv.fit(X_train, y_train)

# Print best params and the corresponding R²
print(f"Best hyperparameters: {elastic_cv.best_params_}")
print(f"Best R² (train): {elastic_cv.best_score_}")
# GridSearchCV
elastic_cv = GridSearchCV(ElasticNet(), tuned_parameters, cv=10, n_jobs=-1, verbose=1)

# Fit the model on trains set
elastic_cv.fit(X_train, y_train)

# Print best params and the corresponding R²
print(f"Best hyperparameters: {elastic_cv.best_params_}")
print(f"Best R² (train): {elastic_cv.best_score_}")

# Elasticnet Regressor with the best hyperparameters
elastic_mod = ElasticNet(alpha=elastic_cv.best_params_["alpha"],
                         l1_ratio=elastic_cv.best_params_["l1_ratio"])

# Fit the model on train set
elastic_mod.fit(X_train, y_train)

# Predict on test set
y_pred = elastic_mod.predict(X_test)

print(f"- {elastic_mod.__class__.__name__}")
print(f"R²: {rsqr_score(y_test, y_pred)}")
print(f"RMSE: {rmse_score(y_test, y_pred)}")

#Save model results into list
model_list = []
r2_list = []
rmse_list = []

model_list.append(elastic_mod.__class__.__name__)
r2_list.append(round(rsqr_score(y_test, y_pred), 4))
rmse_list.append(round(rmse_score(y_test, y_pred), 4))

from sklearn.linear_model import ElasticNet

elastic_mod = elastic_cv.best_estimator_
pred1 = elastic_mod.predict(X_train_std)
pred2 = elastic_mod.predict(X_test)

Evaluate(4, pred1, pred2)

"""## Evaluation Model"""

EMC = Model_Evaluation.copy()
EMC = EMC.iloc[:5]
EMC.index = ['Lasso Linear Regression (LLR)',
             'Ridge Linear Regression (RLR)',
             'XGBoost Regression (XGBoostER)',
             'LightGBM Regression (LGBM)',
             'Elasticnet Regression (ELR)']
EMC

import pandas as pd

# hasil akhir
results = {
    "Model": ["Ridge", "Lasso", "XGBoost", "LightGBM"],
    "R2 (Test)": [0.91070892397036 , 0.9106699729520129, 0.9130605459213257, 0.8822771122927895],
    "RMSE (Test)": [0.11875812111598144, 0.11878402093488245, 0.11718385213896368, 0.13636086358837093]
}

# Buat dataframe
df_results = pd.DataFrame(results)

# Tampilkan hasil
print("Model Comparison on Test Set:")
print(df_results)

# Tampilkan model terbaik berdasarkan R²
best_model = df_results.loc[df_results["R2 (Test)"].idxmax()]
print("\nBest Model Based on R²:")
print(best_model)

# Tampilkan model terbaik berdasarkan RMSE
best_model_rmse = df_results.loc[df_results["RMSE (Test)"].idxmin()]
print("\nBest Model Based on RMSE:")
print(best_model_rmse)

import pickle

# Save model
with open('regressor.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save xgboost

with open('xgboost.pkl', 'wb') as file:
    pickle.dump(xgboost, file)