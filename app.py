
import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(
    page_title="GeneScreen AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

.stApp {
    background:#F3F6FA;
    color:#18283B;
    font-family:'DM Sans',sans-serif;
}
[data-testid="stSidebar"] {background:#101D32;}
[data-testid="stSidebar"] * {color:#F4F8FC;}
.hero {
    background:linear-gradient(120deg,#101D32,#1B4556);
    border-radius:20px;
    padding:30px;
    color:white;
    margin-bottom:24px;
}
.hero-label {
    color:#6DE0CD;
    font-weight:700;
    font-size:11px;
    letter-spacing:2px;
}
.hero h1 {
    color:white;
    font-family:'Manrope',sans-serif;
    font-size:36px;
    font-weight:800;
}
.hero p {color:#D4E1EB;line-height:1.7;}
.section-title {
    font-family:'Manrope',sans-serif;
    font-size:22px;
    font-weight:800;
    margin:10px 0 5px;
}
.section-subtitle {
    color:#718096;
    font-size:13px;
    margin-bottom:18px;
}
.metric,.panel {
    background:white;
    border:1px solid #E0E7EF;
    border-radius:15px;
    padding:20px;
    margin-bottom:16px;
}
.metric-label {color:#718096;font-size:12px;}
.metric-value {
    font-family:'Manrope',sans-serif;
    font-size:26px;
    font-weight:800;
    margin-top:10px;
}
.panel-title {font-size:16px;font-weight:800;}
.panel-text {color:#718096;font-size:13px;line-height:1.8;}
.result {
    background:#DDF7F0;
    border:1px solid #B6E6D9;
    border-radius:17px;
    padding:24px;
    margin:18px 0;
}
.result-value {
    color:#123E42;
    font-size:30px;
    font-weight:800;
    overflow-wrap:anywhere;
}
.notice {
    background:#FFF7E6;
    border:1px solid #F0DCA9;
    color:#765719;
    border-radius:12px;
    padding:15px;
    font-size:12px;
    line-height:1.8;
    margin-top:20px;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background:#0D9488;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:700;
    min-height:45px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def hero(label, title, description):
    st.markdown(
        f'<div class="hero"><div class="hero-label">{label}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True
    )


def section(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric(label, value):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )


def disclaimer():
    st.markdown("""
    <div class="notice"><b>Research demonstration only.</b>
    This application predicts dataset labels. It is not a validated
    medical screening or diagnostic tool. Never use predictions
    to make healthcare decisions.</div>
    """, unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model_files = {
        "Linear SVM": "linear_svm.joblib",
        "Polynomial SVM": "polynomial_svm.joblib",
        "RBF SVM": "rbf_svm.joblib"
    }

    loaded = {}
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing model: {filename}")
        if os.path.getsize(path) < 1000:
            raise ValueError(f"{filename} may be incomplete.")
        loaded[name] = joblib.load(path)

    with open(
        os.path.join(MODEL_DIR, "feature_schema.json"),
        "r", encoding="utf-8"
    ) as f:
        features = json.load(f)

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.joblib")
    )
    return loaded, features, encoder


try:
    models, features, encoder = load_models()
except Exception as error:
    st.title("🧬 GeneScreen AI")
    st.error("Unable to load the saved models.")
    st.exception(error)
    st.stop()


# ---------------- NAVIGATION ----------------
# IMPORTANT: Set the target BEFORE creating the radio widget.
if "nav_target" in st.session_state:
    st.session_state["page"] = st.session_state.pop("nav_target")

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

with st.sidebar:
    st.markdown("## 🧬 GeneScreen AI")
    st.caption("GENETIC RESEARCH PLATFORM")
    st.divider()
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Screening Demo",
         "Model Performance", "About Project"],
        key="page",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("🟢 **Models loaded**")
    st.caption("Three SVM models available for research demonstration.")


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    hero(
        "GENETIC RESEARCH • MACHINE LEARNING",
        "Explore genetic data.<br>Understand predictions.",
        "An educational demonstration of Support Vector Machine "
        "models predicting subclass labels from a research dataset."
    )

    section("Workspace overview", "Choose an area to explore.")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("AVAILABLE MODELS", "03")
    with c2:
        metric("TRAINING FEATURES", len(features))
    with c3:
        metric("PLATFORM", "Research")

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="panel">
        <div class="panel-title">🧪 Screening Demo</div>
        <div class="panel-text">
        Enter fictional sample values and test a saved model.
        </div></div>
        """, unsafe_allow_html=True)

        if st.button("Open Screening Demo →",
                     type="primary", use_container_width=True):
            st.session_state["nav_target"] = "Screening Demo"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="panel">
        <div class="panel-title">📊 Model Performance</div>
        <div class="panel-text">
        Compare previously recorded evaluation metrics.
        </div></div>
        """, unsafe_allow_html=True)

        if st.button("View Model Results →",
                     use_container_width=True):
            st.session_state["nav_target"] = "Model Performance"
            st.rerun()

    st.markdown("""
    <div class="panel">
    <div class="panel-title">How it works</div>
    <div class="panel-text">
    01 · Enter sample details<br><br>
    02 · Select an SVM model<br><br>
    03 · View the predicted dataset label<br><br>
    04 · Interpret the result cautiously
    </div></div>
    """, unsafe_allow_html=True)

    disclaimer()


# ---------------- SCREENING DEMO ----------------
elif page == "Screening Demo":
    hero(
        "INTERACTIVE WORKSPACE",
        "Screening Demo",
        "Use fictional values to explore model behavior. "
        "This form is for software demonstration only."
    )

    st.info(
        "Only a subset of training features is entered here. "
        "Remaining features are set to zero, so predictions "
        "may be unreliable."
    )

    with st.form("screening_form"):
        section("01 · Sample information")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Patient age", 0, 120, 10
            )
        with c2:
            blood_count = st.number_input(
                "Blood cell count (mcL)",
                min_value=0.0, max_value=20.0,
                value=4.7, step=0.1
            )

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox(
                "Gender", ["Female", "Male", "Other"]
            )
            maternal_gene = st.selectbox(
                "Maternal gene", ["Unknown", "Yes", "No"]
            )
        with c2:
            paternal_gene = st.selectbox(
                "Paternal gene", ["Unknown", "Yes", "No"]
            )
            inherited = st.selectbox(
                "Inherited from father", ["Unknown", "Yes", "No"]
            )

        st.divider()
        section("02 · Symptom indicators")

        symptom_cols = st.columns(5)
        symptoms = {}
        for i, col in enumerate(symptom_cols, start=1):
            with col:
                symptoms[f"Symptom {i}"] = st.selectbox(
                    f"Symptom {i}", [0, 1], index=0
                )

        st.divider()
        section("03 · Model selection")

        model_choice = st.selectbox(
            "Prediction model", list(models.keys())
        )

        submitted = st.form_submit_button(
            "Run Demo Prediction →",
            type="primary", use_container_width=True
        )

    if submitted:
        try:
            sample = pd.DataFrame(
                0, index=[0], columns=features
            )

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
                encoded = f"{column}_{value}"
                if encoded in sample.columns:
                    sample.loc[0, encoded] = 1

            for column, value in symptoms.items():
                if column in sample.columns:
                    sample.loc[0, column] = value

            prediction = models[model_choice].predict(sample)[0]
            subclass = encoder.inverse_transform([prediction])[0]

            st.markdown(
                '<div class="result">'
                '<div class="result-label">DEMO PREDICTION</div>'
                f'<div class="result-value">{subclass}</div>'
                f'<div class="panel-text">Model: {model_choice}</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "This is a dataset label prediction, not a diagnosis. "
                "Many trained features are missing from this form."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

    disclaimer()


# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    hero(
        "MODEL EVALUATION",
        "Model Performance",
        "Previously recorded holdout results. "
        "These metrics do not establish clinical reliability."
    )

    results = pd.DataFrame({
        "Model": ["Linear SVM", "Polynomial SVM", "RBF SVM"],
        "Accuracy": [0.3940, 0.3534, 0.3969],
        "Macro Precision": [0.2203, 0.2811, 0.2601],
        "Macro Recall": [0.2566, 0.1960, 0.2314],
        "Macro F1": [0.2331, 0.1870, 0.2255]
    })

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("LINEAR SVM ACCURACY", "39.40%")
    with c2:
        metric("POLYNOMIAL SVM ACCURACY", "35.34%")
    with c3:
        metric("RBF SVM ACCURACY", "39.69%")

    section("Holdout evaluation", "Previously recorded model scores.")

    display = results.copy()
    for column in [
        "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
    ]:
        display[column] = (
            display[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Accuracy and Macro F1 comparison")
    st.bar_chart(results.set_index("Model")[["Accuracy", "Macro F1"]])

    st.warning(
        "The recorded scores are limited. These models are not "
        "validated for medical screening."
    )
    disclaimer()


# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    hero(
        "ABOUT THE PROJECT",
        "GeneScreen AI",
        "A student research demonstration for predicting "
        "genetic-disorder subclass labels."
    )

    section("Project overview")

    st.markdown("""
    <div class="panel">
    <div class="panel-title">Objective</div>
    <div class="panel-text">
    Explore a machine-learning workflow using saved SVM classifiers
    to predict subclass labels from selected input features.
    </div><br>
    <div class="panel-title">Algorithms</div>
    <div class="panel-text">
    • Linear Support Vector Machine<br>
    • Polynomial Support Vector Machine<br>
    • RBF Support Vector Machine
    </div><br>
    <div class="panel-title">Technology stack</div>
    <div class="panel-text">
    Python · Streamlit · Pandas · NumPy · Scikit-learn · Joblib
    </div></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
    <div class="panel-title">Known limitations</div>
    <div class="panel-text">
    • Earlier holdout accuracy was approximately 35–40%.<br><br>
    • The form does not collect every training feature.<br><br>
    • This is not a validated medical screening tool.<br><br>
    • Predictions must not guide healthcare decisions.
    </div></div>
    """, unsafe_allow_html=True)

    disclaimer()

  
