import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc)
import numpy as np

def calculate_metrics(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, precision, recall, f1, y_pred

def show_dataset_info(shape, classes):
    st.sidebar.write("<h4 style='color: #ff6600;'>DATASET INFO</h4>", unsafe_allow_html=True)
    st.sidebar.write(f"Shape of Dataset: {shape}")
    st.sidebar.write(f"Number of Classes: {classes}")

def show_metrics(accuracy, precision, recall, f1):
    st.sidebar.write("<h4 style='color: #ff6600;'>PERFORMANCE SCORES</h4>", unsafe_allow_html=True)
    st.sidebar.write(f"Accuracy Score  : {accuracy:.3f}")
    st.sidebar.write(f"Precision Score : {precision:.3f}")
    st.sidebar.write(f"Recall Score    : {recall:.3f}")
    st.sidebar.write(f"F1 Score        : {f1:.3f}")

def show_classification_report(y_test, y_pred):
    st.markdown("<h4 style='color: #3366ff;'>Classification Report:</h4>", unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})
    st.dataframe(report_df)

def show_confusion_matrix(y_test, y_pred):
    st.write("<h4 style='color: #3366ff;'>Confusion Matrix:</h4>", unsafe_allow_html=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    st.pyplot(plt.gcf())

def plot_roc_curve(clf, X_test, y_test):
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            st.write("<h4 style='color: #3366ff;'>ROC Curve:</h4>", unsafe_allow_html=True)
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
        st.write("This classifier does not support probability predictions.")