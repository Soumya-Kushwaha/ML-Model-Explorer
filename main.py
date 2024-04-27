import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

st.title("ML Model Master")

st.subheader("""
Explore different classifiers and their optimized hyperparameters
""")


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        dataset = datasets.load_iris()

    elif dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()

    else:
        dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target
    return X, y


def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max_depth", 2, 15)
        params["max_depth"] = max_depth

    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    elif clf_name == "Gradient Boosting":
        max_depth = st.sidebar.slider("Max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    elif clf_name == "Naive Bayes":
        pass

    return params


def get_classifier(clf_name, params):
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["C"], max_iter=2000)

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)

    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["n_estimators"],
                                         max_depth=params["max_depth"], random_state=1234)

    elif clf_name == "Naive Bayes":
        clf = GaussianNB()

    return clf


# Setup (Dataset and Classifier options)
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "KNN", "SVM", "Decision Tree",
                                                             "Random Forest", "Gradient Boosting", "Naive Bayes"))
params = add_parameter_ui(classifier_name)

X, y = get_dataset(dataset_name)
clf = get_classifier(classifier_name, params)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train the classifier and predict on the test data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.sidebar.write("Shape of Dataset :", X.shape)
st.sidebar.write("Number of Classes : ", len(np.unique(y)))

# Calculate and display performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.sidebar.write(f"Classifier      : {classifier_name}")
st.sidebar.write(f"Accuracy Score  : {accuracy}")
st.sidebar.write(f"Precision Score : {precision}")
st.sidebar.write(f"Recall Score    : {recall}")
st.sidebar.write(f"F1 Score        : {f1}")

# Display classification report
st.write("#### Classification Report: ####")
report = classification_report(y_test, y_pred)
st.text(report)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
st.write("#### Confusion Matrix: ####")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d', ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
st.pyplot(fig)
