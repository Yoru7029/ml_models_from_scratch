import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import linear_model

from logistic_regression import model

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Train model
sklearn_model = linear_model.LogisticRegression().fit(X_train, y_train)
my_model = model.LogisticRegression(solver="sgd").fit(X_train, y_train)
m = y_test.shape[0]

# Predict
y_pred = sklearn_model.predict(X_test)
y_proba = sklearn_model.predict_proba(X_test)
print("Sklearn model")
print("- Accuracy:", np.sum(y_test == y_pred) / m)
print("- Log loss:", log_loss(y_test, y_proba), "\n")

y_pred = my_model.predict(X_test)
y_proba = my_model.predict_proba(X_test)
print("My model (SGD):")
print("- Accuracy:", np.sum(y_test == y_pred) / m)
print("- Log loss", log_loss(y_test, y_proba), "\n")

my_model = model.LogisticRegression().fit(X_train, y_train)
y_pred = my_model.predict(X_test)
y_proba = my_model.predict_proba(X_test)
print("My model (MLE):")
print("- Accuracy:", np.sum(y_test == y_pred) / m)
print("- Log loss", log_loss(y_test, y_proba))