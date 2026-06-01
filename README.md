# Adult_Dataset
The Adult Dataset (also called the Census Income Dataset) is a popular machine learning dataset used for classification problems. The goal is to predict whether a person's annual income exceeds $50,000 based on demographic and employment-related attributes.
















Adult Income Prediction using Machine Learning
📌 Project Overview
This project uses the Adult Census Income Dataset to predict an individual's income category based on demographic and employment-related attributes. The notebook performs data preprocessing, feature encoding, model training, and evaluation using multiple machine learning algorithms.

The objective is to classify whether a person's income falls into a specific income category using supervised learning techniques.

📂 Dataset Information
The dataset contains information such as:

Age

Workclass

Final Weight (fnlwgt)

Education

Education Number

Marital Status

Occupation

Relationship

Race

Sex

Capital Gain

Capital Loss

Hours per Week

Native Country

Income

🛠️ Technologies Used
Python

NumPy

Pandas

Matplotlib

Scikit-learn

Jupyter Notebook

📊 Data Preprocessing
The following preprocessing steps were performed:

Loading the Adult Dataset.

Checking dataset structure and missing values.

Handling null values.

Data visualization using histograms.

Label Encoding categorical features.

Feature selection and target variable creation.

Train-test split for model training and testing.

🤖 Machine Learning Models Used
1. Support Vector Machine (SVM)
Classification model used for income prediction.

Trained and evaluated on training and testing datasets.

2. Logistic Regression
Linear classification algorithm.

Used for comparison with other models.

3. Random Forest Classifier
Ensemble learning method.

Provides robust performance through multiple decision trees.

4. Decision Tree Classifier
Tree-based classification algorithm.

Easy to interpret and visualize.

📈 Model Evaluation
The models were evaluated using:

Training Accuracy

Testing Accuracy

Accuracy Score

