import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc)

st.markdown("<h1 style='color: #ff6600;'>ML MODEL EXPLORER</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size: 20px;'>Explore different classifiers and their optimized hyperparameters</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


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
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    elif clf_name == "Naive Bayes":
        clf = GaussianNB()
    return clf


# Calculate performance metrics
def calculate_metrics(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, precision, recall, f1, y_pred


# Function to show classifier and dataset information
def show_dataset_info(shape, classes):
    st.sidebar.write("<h4 style='color: #ff6600;'>DATASET INFO</h4>", unsafe_allow_html=True)
    st.sidebar.write(f"Shape of Dataset: {shape}")
    st.sidebar.write(f"Number of Classes: {classes}")


# Display Performance Metrics
def show_metrics(accuracy, precision, recall, f1):
    st.sidebar.write("<h4 style='color: #ff6600;'>PERFORMANCE SCORES</h4>", unsafe_allow_html=True)
    st.sidebar.write(f"Accuracy Score  : {accuracy:.3f}")
    st.sidebar.write(f"Precision Score : {precision:.3f}")
    st.sidebar.write(f"Recall Score    : {recall:.3f}")
    st.sidebar.write(f"F1 Score        : {f1:.3f}")


# Classification Report
def show_classification_report(y_test, y_pred):
    st.markdown("<h4 style='color: #3366ff;'>Classification Report:</h4>", unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})
    st.dataframe(report_df)


# Confusion Matrix
def show_confusion_matrix(y_test, y_pred):
    st.write("<h4 style='color: #3366ff;'>Confusion Matrix:</h4>", unsafe_allow_html=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    st.pyplot(plt.gcf())


# ROC Curve
def roc_curve(clf, X_test, y_test):
    if hasattr(clf, "predict_proba"):
        st.write("<h4 style='color: #3366ff;'>ROC Curve:</h4>", unsafe_allow_html=True)
        y_prob = clf.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="red")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt.gcf())
        else:
            st.write("Cannot plot ROC curve for multi-class classification.")
    else:
        st.write("This classifier does not support probability predictions.")


# Setup (Dataset and Classifier options)
dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine Dataset"])
classifier_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest", "Gradient Boosting", "Naive Bayes"])

params = add_parameter_ui(classifier_name)
submit_button = st.sidebar.button("Predict")

# Submit Button
if submit_button:
    X, y = get_dataset(dataset_name)
    clf = get_classifier(classifier_name, params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)

    shape = X.shape
    classes = len(np.unique(y))
    show_dataset_info(shape, classes)

    accuracy, precision, recall, f1, y_pred = calculate_metrics(clf, X_test, y_test)
    show_metrics(accuracy, precision, recall, f1)

    show_classification_report(y_test, y_pred)
    show_confusion_matrix(y_test, y_pred)
    roc_curve(clf, X_test, y_test)
