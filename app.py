import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
import sklearn

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
st.write(
    "Research demonstration for predicting a dataset label "
    "using machine learning. This is not a medical diagnosis."
)

# Load saved models and feature information
MODEL_DIR = "models"

@st.cache_resource
def load_files():
    models = {
        "Linear SVM": joblib.load(
            os.path.join(MODEL_DIR, "linear_svm.joblib")
        ),
        "Polynomial SVM": joblib.load(
            os.path.join(MODEL_DIR, "polynomial_svm.joblib")
        ),
        "RBF SVM": joblib.load(
            os.path.join(MODEL_DIR, "rbf_svm.joblib")
        )
    }

    with open(os.path.join(MODEL_DIR, "feature_schema.json")) as f:
        features = json.load(f)

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.joblib")
    )

    return models, features, encoder

try:
    models, features, encoder = load_files()

    st.success("Models loaded successfully!")

    st.markdown("### Enter the sample details")

    # The trained model expects 82 encoded features.
    # Ask for the original dataset's feature values in a simple form.
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=10)
    blood_count = st.number_input(
        "Blood cell count (mcL)", min_value=0.0,
        max_value=20.0, value=4.7
    )
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    maternal_gene = st.selectbox("Maternal gene", ["Yes", "No", "Unknown"])
    paternal_gene = st.selectbox("Paternal gene", ["Yes", "No", "Unknown"])
    inherited = st.selectbox(
        "Inherited from father", ["Yes", "No", "Unknown"]
    )

    symptoms = {}
    for i in range(1, 6):
        symptoms[f"Symptom {i}"] = st.selectbox(
            f"Symptom {i}", [0, 1], index=0
        )

    model_choice = st.selectbox(
        "Choose SVM model", list(models.keys())
    )

    if st.button("Predict Subclass", type="primary"):
        # Create an all-zero feature row, then fill the fields provided.
        sample = pd.DataFrame(0, index=[0], columns=features)

        sample["Patient Age"] = age
        sample["Blood cell count (mcL)"] = blood_count

        # Match the categorical one-hot columns created during training.
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
        subclass = encoder.inverse_transform([prediction])[0]

        st.markdown("### Prediction result")
        st.info(f"Predicted dataset subclass: **{subclass}**")
        st.warning(
            "This is an experimental machine-learning prediction, "
            "not a diagnosis. Consult a qualified healthcare professional "
            "for medical assessment."
        )

except Exception as e:
    st.error("Could not load the saved model files.")
    st.write("Exact error:")
    st.exception(e)

    st.write("Files found in repository:")
    st.write(os.listdir("."))
    st.write("Files inside models folder:")
    if os.path.isdir("models"):
        st.write(os.listdir("models"))
    else:
        st.error("The models folder was not found.")
