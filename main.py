import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model

# Read in the data
data = pd.read_csv("student-mat.csv", sep=";")

# Pick the features
data = data[["G1", "G2", "studytime", "absences", "failures", "schoolsup", "famsup", "higher", "internet", "paid", "traveltime", "G3"]]

# Get the features into the right format
data.famsup = data.famsup.eq('yes').mul(1)
data.schoolsup = data.schoolsup.eq('yes').mul(1)
data.higher = data.higher.eq('yes').mul(1)
data.internet = data.internet.eq('yes').mul(1)
data.paid = data.paid.eq('yes').mul(1)

# Make the X and y arrays
predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split the data into training and test data (7:3 training-test split)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)

# Create the linear regression model
linRegModel = linear_model.LinearRegression()

# Train the model
linRegModel.fit(x_train, y_train)

# Display the coefficient and intercept of the model
print("Coefficient Vector: ", linRegModel.coef_)
print("Intecept: ", linRegModel.intercept_)

# Score the model
accuracy = linRegModel.score(x_test, y_test)

# Display the accuracy of the model
print("Accuracy: ", accuracy)
