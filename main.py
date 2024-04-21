import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    fbeta_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline

st.title("Early Detection of Heart Disease Based on Underlying Symptoms")

dataset_path = 'Dataset/Heart.csv'
df = pd.read_csv(dataset_path)
st.write("preview of the dataset:")
st.dataframe(df.head())

normalization_method = st.sidebar.radio(
    "Select a normalization technique:",
    ("Z-Score", "Min-Max")
)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Random Forest', 'AdaBoost', 'SVM', 'Decision Tree')
)

params = {}
if classifier_name == 'Random Forest':
    st.sidebar.subheader("Random Forest parameters")
    params["criterion"] = st.sidebar.selectbox("Select Criterion", ["gini", "entropy"])
    params["max_depth"] = st.sidebar.slider("Max Depth", 2, 10)
    params["n_estimators"] = st.sidebar.slider("N Estimators", 10, 100)
elif classifier_name == 'AdaBoost':
    st.sidebar.subheader("AdaBoost parameters")
    params["n_estimators"] = st.sidebar.slider("N Estimators", 50, 100)
elif classifier_name == 'SVM':
    st.sidebar.subheader("SVM parameters")
    params["C"] = st.sidebar.slider("C", 0.01, 1.0)
elif classifier_name == 'Decision Tree':
    st.sidebar.subheader("Decision Tree parameters")
    params["max_depth"] = st.sidebar.slider("Max Depth", 2, 15)


def plot_pca_and_show_eigen(df_selected):
    if normalization_method == "Z-Score":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df_standardized = scaler.fit_transform(df_selected)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_standardized)
    explained_variance = pca.explained_variance_ratio_

    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    loadings = eigenvectors.T * np.sqrt(eigenvalues)

    loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=df_selected.columns)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap([explained_variance], annot=True, fmt=".2%", cmap="Blues", ax=ax[0])
    ax[0].set_title('PCA Explained Variance')
    ax[0].set_xlabel('Principal Components')
    ax[0].set_yticklabels([''], rotation=360)

    # PCA scatter plot
    ax[1].scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].set_title('2D PCA Plot')
    st.pyplot(fig)

    # Display the eigenvalues and eigenvectors
    st.write("Eigenvalues:", eigenvalues)
    st.write("Eigenvectors (Loadings):")
    st.dataframe(loadings_df)


if st.sidebar.checkbox('Show PCA'):
    st.write("PCA Analysis")
    # Select the features excluding the target
    attributes = df.drop('target', axis=1).columns
    plot_pca_and_show_eigen(df[attributes])


def create_pipeline(normalization_method, classifier_name, params):
    if normalization_method == "Z-Score":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    if classifier_name == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=params.get("n_estimators", 10),
                                            max_depth=params.get("max_depth"),
                                            criterion=params.get("criterion", "gini"),
                                            random_state=0)
    elif classifier_name == 'AdaBoost':
        classifier = AdaBoostClassifier(n_estimators=params.get("n_estimators", 50),
                                        random_state=0)
    elif classifier_name == 'SVM':
        classifier = SVC(C=params.get("C", 1.0))
    elif classifier_name == 'Decision Tree':
        classifier = DecisionTreeClassifier(max_depth=params.get("max_depth"),
                                            random_state=0)

    pipeline = Pipeline([
        ('normalizer', scaler),
        ('classifier', classifier)
    ])
    return pipeline


def train_and_evaluate_model(pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    return y_test, prediction


if st.button("Train Model"):
    X = df.drop('target', axis=1)
    y = df['target']
    pipeline = create_pipeline(normalization_method, classifier_name, params)
    y_test, predictions = train_and_evaluate_model(pipeline, X, y)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    f2 = fbeta_score(y_test, predictions, beta=2)

    conf_matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Displaying metrics
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write(f"F2 Score: {f2:.2f}")
