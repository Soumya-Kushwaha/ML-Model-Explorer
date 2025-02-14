import streamlit as st
from data.datasets import get_dataset
from models.classifiers import add_parameter_ui, get_classifier
from utils.metrics import calculate_metrics, show_dataset_info, show_metrics, show_classification_report, show_confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split
import numpy as np

st.markdown("<h1 style='color: #ff6600;'>ML MODEL EXPLORER</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size: 20px;'>Explore different classifiers and their optimized hyperparameters</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Setup (Dataset and Classifier options)
dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine"])
classifier_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest", "XGBoost"])

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
    plot_roc_curve(clf, X_test, y_test)