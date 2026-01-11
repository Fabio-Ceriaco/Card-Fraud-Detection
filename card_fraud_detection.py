# Work Flow
# 1 Import Libraries
# 2 Load Data
# 3 Data Preprocessing
# 4 Feature Engineering
# 5 Split the Data
# 6 Model Building
# 7 Model Evaluation
# 8 Conclusion

# ==============================================================
# 1 Import Libraries
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    PrecisionRecallDisplay,
    precision_score,
    confusion_matrix,
)

# ==============================================================
# 2 Load Data
# ==============================================================

data_df = pd.read_csv("creditcard.csv")

# ==============================================================
# 3 Data Preprocessing
# ==============================================================

# Print the first few rows of the dataset
print(data_df.head())

# Print data shape
print(data_df.shape)

# Print data info
print(data_df.info())

# Print data description
print(data_df.describe())

# Check for missing values
print(data_df.isnull().sum())

# Print class Distribution
print(data_df["Class"].value_counts())


# As We can see that the dataset is highly unbalanced
# 0 --> Legitimate Transaction
# 1 --> Fraudulent Transaction
# When we have an unbalanced dataset, accuracy is not the best metric to evaluate the model
# We will use other metrics like F1 Score, Precision, Recall etc.
# Tho deal with imbalanced data, we will use Under-Sampling technique
# Under-Sampling is a technique where we reduce the number of majority
# class samples to match the number of minority class samples
# This helps in balancing the dataset and improving the model performance


# Separate the data for analysis

legit = data_df[data_df["Class"] == 0]

print(legit)

fraud = data_df[data_df["Class"] == 1]

print(fraud)

print(legit.shape)
print(fraud.shape)

# Statistical measures of the data

print(legit.Amount.describe())

print(fraud.Amount.describe())

# Compare the values for both transactions

print(data_df.groupby("Class").mean())

# How to deal with imbalanced data

# Under-Smampling
# Build a sample dataset containing similar distribution of Legit and Fraudulent Transactions

legit_sample = legit.sample(n=492)

print(legit_sample.shape)

# Concatenate the two dataframes

new_df = pd.concat([legit_sample, fraud], axis=0)

print(new_df.head())
print(new_df.tail())

print(new_df.shape)

# Shuffle the new dataframe
# Shuffling is important to avoid any bias in the data
# Reset the index after shuffling the dataframe
# frac=1 means we want to shuffle the entire dataframe the opposite of frac=0.5 which would shuffle only half the dataframe
# reset_index(drop=True) is used to reset the index of the dataframe after
# shuffling and drop=True is used to avoid adding the old index as a new column in the dataframe

new_df = new_df.sample(frac=1, random_state=2).reset_index(drop=True)

print(new_df.head())
print(new_df.tail())

# Check the class distribution of the new dataframe

print(new_df["Class"].value_counts())

# Statistical measures of the new dataframe
print(new_df.groupby("Class").mean())

# ==============================================================
# 4 Feature Engineering
# ==============================================================

X = new_df.drop(columns="Class", axis=1)

y = new_df["Class"]

# ==============================================================
# 5 Split the Data
# ==============================================================


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)


# ==============================================================
# 6 Model Building
# ==============================================================

model = LogisticRegression(max_iter=20000)

model.fit(X_train, y_train)

# ==============================================================
# 7 Model Evaluation
# ==============================================================

# Accuracy on Training Data

train_prediction = model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_prediction)

print("Accuracy on Training Data: ", train_accuracy)

# Accuracy on Test Data

test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)

print("Accuracy on Test Data: ", test_accuracy)

# Conlusions:
# The accuracy on training and test data is quite similar
# This indicates that the model is not overfitting or underfitting
# The model is generalizing well on the unseen data
# However, accuracy is not the best metric to evaluate the model
# We will use other metrics like F1 Score, Precision Score  and Precision-Recall Curve.

# Confusion Matrix
cm = confusion_matrix(y_test, test_prediction)

print("Confusion Matrix:\n", cm)


# Conclusion:
# Confusion Matrix : [[94  5][ 5 93]]
# True Positives (TP) = 93
# True Negatives (TN) = 94
# False Positives (FP) = 5
# False Negatives (FN) = 5
# The model is performing well as the number of TP and TN are high
# and the number of FP and FN are low.

# F1 Score on Test Data
f1 = f1_score(y_test, test_prediction)

print("F1 Score on Test Data: ", f1)

# Conclusion:
# F1_Score: 0.9489795918367347
# The F1 Score is quite high, indicating that the model is performing well
# F1 Score is the harmonic mean of Precision and Recall
# It is a better metric to evaluate the model when we have imbalanced data
# A high F1 Score indicates that the model has low false positives and low false negatives.

# Precision-Recall Curve
display = PrecisionRecallDisplay.from_estimator(
    model,
    X_test,
    y_test,
    name="Logistic Regression",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set_title("Precision-Recall Curve")
plt.show()


# CConclusion:
# As we can see from the Precision-Recall Curve, the model is performing well
# as the curve is well above the chance level.
# Chance level is the performance of a random classifier. It can be calculated as the ratio of positive samples to the total samples in the dataset.
# In this case, since the dataset is balanced after under-sampling, the chance level is 0.5.
# A good model should have a Precision-Recall curve that is well above the chance level.

# Precision Score

precision = precision_score(y_test, test_prediction)

print("Precision Score on Test Data: ", precision)

# Conclusion:
# Precision Score: 0.9489795918367347
# The Precision Score is quite high, indicating that the model is performing well
# Precision is the ratio of true positives to the sum of true positives and false positives
# A high Precision Score indicates that the model has low false positives.

# Verify Overfitting or Underfitting

if train_accuracy > test_accuracy:

    print(
        f"The model is overfitting for {round(train_accuracy - test_accuracy, 2)} difference."
    )
elif test_accuracy > train_accuracy:
    print(
        f"The model is underfitting for {round(test_accuracy - train_accuracy, 2)} difference."
    )
else:
    print("The model is neither overfitting nor underfitting.")

# ==============================================================
# 8 Conclusion
# ==============================================================

# The model has been successfully built and evaluated.
# Further improvements can be made by trying different algorithms,
# tuning hyperparameters, and using advanced techniques for handling imbalanced data.
# The Logistic Regression model achieved good accuracy and F1 score on the test data.

# Prediction System

input_data = (
    0.0,
    -1.3598071336738,
    -0.0727811733098497,
    2.53634673796914,
    1.37815522427443,
    -0.338320769942518,
    0.462387777762292,
    0.239598554061257,
    0.0986979012610507,
    0.363786969611213,
    0.0907941719789316,
    -0.551599533260813,
    -0.617800855762348,
    -0.991389847235408,
    -0.311169353699879,
    1.46817697209427,
    -0.470400525136192,
    0.207971241929242,
    0.0257905801985893,
    0.403992960255733,
    0.251412098239705,
    -0.018306777944153,
    0.277837575558899,
    -0.110473910188767,
    0.0669280749146731,
    0.128539358273528,
    -0.189114843888824,
    0.133558376740387,
    -0.0210530534538215,
    149.62,
)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshapped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshapped)

if prediction[0] == 0:
    print("The transaction is Legitimate.")
else:
    print("The transaction is Fraudulent.")
