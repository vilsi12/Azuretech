# Import libraries
import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from azureml.core import Model

# Get the script arguments (regularization rate and training dataset ID)
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

# Set regularization hyperparameter (passed as an argument to the script)
reg = args.reg_rate
inputdata = args.training_dataset_id

# Get the experiment run context
run = Run.get_context()

ws = run.experiment.workspace

# get the input dataset by ID
dataset = Dataset.get_by_id(ws, id=inputdata)

# Get the training dataset
print("Loading Data...")
diabetes = dataset.to_pandas_dataframe()

# Separate features and labels
X, y = diabetes[['AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6']].values, diabetes['Y'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LinearRegression().fit(X_train,y_train)
# model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))


os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')




run.complete()
