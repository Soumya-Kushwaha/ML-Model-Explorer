import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("ML Model Master")

st.write("""
## Explore different classifiers and their optimized hyperparameters
### Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "KNN", "SVM", "Decision Tree",
                                                             "Random Forest", "Gradient Boosting", "Naive Bayes"))


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


X, y = get_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


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


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C = params["C"])

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


clf = get_classifier(classifier_name, params)

# Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show()
st.pyplot(fig)
