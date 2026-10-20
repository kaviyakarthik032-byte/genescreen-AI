
import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
import sklearn
import numpy as np

st.set_page_config(
    page_title="GeneScreen AI",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 GeneScreen AI")
st.subheader("Genetic Disorder Subclass Screening System")

st.write("Python version:", sys.version)
st.write("Scikit-learn version:", sklearn.__version__)
st.write("Joblib version:", joblib.__version__)
st.write("NumPy version:", np.__version__)

st.write(
    "Research demonstration for predicting a dataset label "
    "using machine learning. This is not a medical diagnosis."
)

MODEL_DIR = "models"


# Test saved files separately to identify loading errors
st.markdown("### Model File Diagnostics")

required_files = [
    "linear_svm.joblib",
    "polynomial_svm.joblib",
    "rbf_svm.joblib",
    "label_encoder.joblib",
    "feature_schema.json"
]

files_ok = True

for filename in required_files:
    path = os.path.join(MODEL_DIR, filename)

    if os.path.isfile(path):
        st.success(
            f"{filename} found "
            f"({os.path.getsize(path)} bytes)"
        )
    else:
        st.error(f"{filename} is missing")
        files_ok = False


@st.cache_resource
def load_files():
    models = {}

    model_files = {
        "Linear SVM": "linear_svm.joblib",
        "Polynomial SVM": "polynomial_svm.joblib",
        "RBF SVM": "rbf_svm.joblib"
    }

    for model_name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        models[model_name] = joblib.load(path)

    with open(
        os.path.join(MODEL_DIR, "feature_schema.json"),
        "r"
    ) as file:
        features = json.load(file)

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.joblib")
    )

    return models, features, encoder


# Load models
try:
    if not files_ok:
        st.stop()

    models, features, encoder = load_files()

    st.success("All models loaded successfully!")

except Exception as error:
    st.error("Model loading failed.")
    st.write("Exact error details:")
    st.exception(error)

    st.info(
        "Please share this error message so the model "
        "compatibility issue can be fixed."
    )
    st.stop()


# Prediction interface
st.markdown("## Enter Sample Details")

age = st.number_input(
    "Patient Age",
    min_value=0,
    max_value=120,
    value=10
)

blood_count = st.number_input(
    "Blood cell count (mcL)",
    min_value=0.0,
    max_value=20.0,
    value=4.7
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

maternal_gene = st.selectbox(
    "Maternal gene",
    ["Yes", "No", "Unknown"]
)

paternal_gene = st.selectbox(
    "Paternal gene",
    ["Yes", "No", "Unknown"]
)

inherited = st.selectbox(
    "Inherited from father",
    ["Yes", "No", "Unknown"]
)

symptoms = {}

for i in range(1, 6):
    symptoms[f"Symptom {i}"] = st.selectbox(
        f"Symptom {i}",
        [0, 1],
        index=0
    )

model_choice = st.selectbox(
    "Choose SVM model",
    list(models.keys())
)

if st.button("Predict Subclass", type="primary"):

    try:
        sample = pd.DataFrame(
            0,
            index=[0],
            columns=features
        )

        # Fill numerical inputs only if columns exist
        if "Patient Age" in sample.columns:
            sample["Patient Age"] = age

        if "Blood cell count (mcL)" in sample.columns:
            sample["Blood cell count (mcL)"] = blood_count

        categorical_values = {
            "Gender": gender,
            "Maternal gene": maternal_gene,
            "Paternal gene": paternal_gene,
            "Inherited from father": inherited
        }

        for column, value in categorical_values.items():
            encoded_column = f"{column}_{value}"

            if encoded_column in sample.columns:
                sample.loc[0, encoded_column] = 1

        for column, value in symptoms.items():
            if column in sample.columns:
                sample.loc[0, column] = value

        prediction = models[model_choice].predict(sample)[0]

        subclass = encoder.inverse_transform(
            [prediction]
        )[0]

        st.markdown("### Prediction Result")
        st.info(f"Predicted dataset subclass: **{subclass}**")

        st.warning(
            "This is an experimental research demonstration, "
            "not a medical diagnosis. Its predictions may be "
            "inaccurate. Consult a qualified healthcare "
            "professional for medical assessment."
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
