# credit-card-fraud-detection
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used contains 284,807 transactions made by European cardholders, with only 492 labeled as fraudulent—highlighting a significant class imbalance problem.

Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Data Collection & processing
# load the data from csv file to Pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')
# printing the first 5 rows of the dataframe
credit_card_data.head()
# number of rows and columns
credit_card_data.shape
# getting some informations about the data
credit_card_data.info()
# check the number of missing values in each column
credit_card_data.isnull().sum()
# distribution of legit transaction & fraudulent transaction
credit_card_data['Class'].value_counts()

This Dataset is highly unbalanced

0->Normal Transaction
1->Fraudulent Transaction
# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
# statistical measures of the data
legit.Amount.describe()
fraud.Amount.describe()
# compare the values for both transactions
credit_card_data.groupby('Class').mean()

Under Sampling

Build a sample dataset containing similar distribution of normal transactions and fraudulent transactions

Number of Fraudulent Transactions->492
legit_sample = legit.sample(n=492)

Concatenating the two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

Splitting the data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
print(Y)

Split the data into training data & testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

Model Training

Logistic Regression
model = LogisticRegression()
#training the Logistic Regression model with training data
model.fit(X_train, Y_train)

Model Evaluation

Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
