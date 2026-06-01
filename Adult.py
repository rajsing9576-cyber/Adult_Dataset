import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
columns = [
    "age"
    "workclass"
    "fnlwgt"
    "education"
    "education-num"
    "marital-statuts"
    "occupation"
    "relationship"
    "race"
    "sex"
    "capital-gain"
    "capital-loss"
    "hours-per-week"
    "native-country"
    "income"
]




data = pd.read_csv(
    'adult.csv',
    header=None,
    names=columns,
    sep=";",
    skipinitialspace=True
)


# print first 5 rows
data.head()

# Info
data.info()

# check the missing values
data.isnull().sum()

# rename the categorical columns to numerical columns
le = LabelEncoder()
categorical_columns =[
    "workclass",
    "education",
    "marital-status",
    "occuaption",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "fnlwgt",
    "education-num"

]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])
    


# Create a histogram plot of all the numerical columns
   
data.hist(bins=50, figsize=(12,8))
plt.show()


# Droping the dataset of a marital-status columns
X = data.drop('marital-status', axis=1)
Y = data['marital-status']



print(X)
print(Y)

# Splitting the dataset into Training data and Testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shap)


# Support Vector Machione

SVC = SVC()

SVC.fit(X_train, Y_train)
train_data = SVC.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Training data accuracy in SVC model:", training_data_accuracy)
test_data = SVC.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Testing data accuracy on SVC Model:", testing_data_accuracy)



# LogisticRegression Model
LogisticRegression = LogisticRegression()
LogisticRegression.fit(X_train, Y_train)
train_data = LogisticRegression.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Training data accuarcy in LogisticRegression Model:", training_data_accuracy)
test_data = LogisticRegression.predict(X_test)
testing_data_accuracy = accuracy_score(X_test, test_data)
print("Testing data accuracy in LogisticRegression Model:", testing_data_accuracy)



# # RandomForestClassifier()
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)
train_data = RandomForestClassifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Training data accuracy in RandomForestClassifier Model:", training_data_accuracy)
test_data = RandomForestClassifier.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Testing data accuracy in RandomForestClassifier Model:", testing_data_accuracy)


# DecisionTreeClassifier()
DecisionTreeClassifier = DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_train, Y_train)
train_data = DecisionTreeClassifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Training data accuarcy in DecisionTreeClassifier Model:", training_data_accuracy)
test_data = DecisionTreeClassifier.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Testing data accuracy in DecisionTreeClassifier Model:", testing_data_accuracy)


