import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

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
    elif clf_name == "XGBoost":
        max_depth = st.sidebar.slider("Max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate

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
    elif clf_name == "XGBoost":
        clf = XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], learning_rate=params["learning_rate"], random_state=1234)

    return clf