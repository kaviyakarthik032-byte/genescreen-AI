
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="GeneScreen AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ================= FRONTEND DESIGN =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F3F6FA;
    color: #18283B;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #101D32;
}

[data-testid="stSidebar"] * {
    color: #F4F8FC;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 4px;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: white;
    padding-top: 12px;
}

.brand-caption {
    color: #A8BED1;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

.hero {
    background: linear-gradient(120deg, #101D32, #1B4556);
    border: 1px solid #29495B;
    border-radius: 20px;
    padding: 30px;
    color: white;
    margin-bottom: 24px;
}

.hero-label {
    color: #6DE0CD;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 2px;
}

.hero h1 {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(27px, 4vw, 38px);
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #D4E1EB;
    line-height: 1.7;
    font-size: 14px;
    max-width: 720px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #18283B;
    margin: 10px 0 5px;
}

.section-subtitle {
    color: #718096;
    font-size: 13px;
    margin-bottom: 18px;
}

.metric {
    background: white;
    border: 1px solid #E0E7EF;
    border-radius: 15px;
    padding: 20px;
    min-height: 110px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #18283B;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 26px;
    margin-top: 10px;
}

.panel {
    background: white;
    border: 1px solid #E0E7EF;
    border-radius: 15px;
    padding: 21px;
    margin-bottom: 16px;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: #18283B;
    margin-bottom: 8px;
}

.panel-text {
    color: #718096;
    font-size: 13px;
    line-height: 1.8;
}

.result {
    background: linear-gradient(120deg, #DDF7F0, #FFFFFF);
    border: 1px solid #B6E6D9;
    border-radius: 17px;
    padding: 24px;
    margin: 18px 0;
}

.result-label {
    color: #0B796D;
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 1.5px;
}

.result-value {
    color: #123E42;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 30px;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.notice {
    background: #FFF7E6;
    border: 1px solid #F0DCA9;
    color: #765719;
    border-radius: 12px;
    padding: 15px;
    font-size: 12px;
    line-height: 1.8;
    margin-top: 20px;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: #0D9488;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: #08786F;
    color: white;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 9px;
}

@media (max-width: 700px) {
    .hero {
        padding: 22px;
    }
    .panel {
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================
def show_hero(label, title, description):
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-label">{label}</div>'
        f'<h1>{title}</h1>'
        f'<p>{description}</p>'
        f'</div>',
        unsafe_allow_html=True
    )


def show_section(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def show_metric(label, value):
    st.markdown(
        f'<div class="metric">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def show_disclaimer():
    st.markdown("""
    <div class="notice">
        <b>Research demonstration only.</b> This application
        predicts labels from a dataset. It is not a validated
        medical screening or diagnostic tool. Predictions may
        be wrong and must not guide healthcare decisions.
    </div>
    """, unsafe_allow_html=True)


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    model_paths = {
        "Linear SVM": "linear_svm.joblib",
        "Polynomial SVM": "polynomial_svm.joblib",
        "RBF SVM": "rbf_svm.joblib"
    }

    loaded_models = {}

    for name, filename in model_paths.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing model file: {path}"
            )

        if os.path.getsize(path) < 1000:
            raise ValueError(
                f"{filename} looks incomplete. "
                "Please upload the original model file."
            )

        loaded_models[name] = joblib.load(path)

    schema_path = os.path.join(
        MODEL_DIR, "feature_schema.json"
    )

    with open(schema_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    encoder_path = os.path.join(
        MODEL_DIR, "label_encoder.joblib"
    )

    label_encoder = joblib.load(encoder_path)

    return loaded_models, feature_names, label_encoder


try:
    models, features, encoder = load_models()
except Exception as error:
    st.title("🧬 GeneScreen AI")
    st.error("Unable to load the saved models.")
    st.write("Error details:")
    st.exception(error)
    st.info(
        "Check the files inside the models folder and "
        "the package versions in requirements.txt."
    )
    st.stop()


# ================= SIDEBAR NAVIGATION =================
with st.sidebar:
    st.markdown(
        '<div style="font-size:34px;">🧬</div>'
        '<div class="brand">GeneScreen AI</div>'
        '<div class="brand-caption">GENETIC RESEARCH PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("WORKSPACE")

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Screening Demo",
            "Model Performance",
            "About Project"
        ],
        key="page",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div style="color:#6DE0CD;font-weight:700;">● Models loaded</div>'
        '<div style="color:#B6C8D9;font-size:12px;line-height:1.7;">'
        'Three SVM models are available for research demonstration.'
        '</div>',
        unsafe_allow_html=True
    )


# ================= DASHBOARD =================
if page == "Dashboard":

    show_hero(
        "GENETIC RESEARCH • MACHINE LEARNING",
        "Explore genetic data.<br>Understand predictions.",
        "An educational demonstration of Support Vector Machine "
        "models predicting subclass labels from a research dataset."
    )

    show_section(
        "Workspace overview",
        "Choose an area to explore."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric("AVAILABLE MODELS", "03")

    with col2:
        show_metric("TRAINING FEATURES", str(len(features)))

    with col3:
        show_metric("PLATFORM", "Research")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">🧪 Screening Demo</div>
            <div class="panel-text">
                Enter fictional sample values and test a saved model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Screening Demo →",
            type="primary",
            use_container_width=True
        ):
            st.session_state.page = "Screening Demo"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📊 Model Performance</div>
            <div class="panel-text">
                Compare the previously recorded evaluation metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "View Model Results →",
            use_container_width=True
        ):
            st.session_state.page = "Model Performance"
            st.rerun()

    st.markdown("""
    <div class="panel">
        <div class="panel-title">How it works</div>
        <div class="panel-text">
            <b>01</b> &nbsp; Enter sample details<br><br>
            <b>02</b> &nbsp; Select an SVM model<br><br>
            <b>03</b> &nbsp; View the predicted dataset label<br><br>
            <b>04</b> &nbsp; Interpret the result cautiously
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_disclaimer()


# ================= SCREENING DEMO =================
elif page == "Screening Demo":

    show_hero(
        "INTERACTIVE WORKSPACE",
        "Screening Demo",
        "Use fictional values to explore the model's behavior. "
        "This form is for software demonstration only."
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Before you start</div>
        <div class="panel-text">
            The form below captures only a small subset of the
            features used during training. Remaining features are
            set to zero, so the output may be unreliable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("screening_form"):

        show_section(
            "01 · Sample information",
            "Basic values for the demonstration."
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Patient age",
                min_value=0,
                max_value=120,
                value=10
            )

        with col2:
            blood_count = st.number_input(
                "Blood cell count (mcL)",
                min_value=0.0,
                max_value=20.0,
                value=4.7,
                step=0.1
            )

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"]
            )

            maternal_gene = st.selectbox(
                "Maternal gene",
                ["Unknown", "Yes", "No"]
            )

        with col2:
            paternal_gene = st.selectbox(
                "Paternal gene",
                ["Unknown", "Yes", "No"]
            )

            inherited = st.selectbox(
                "Inherited from father",
                ["Unknown", "Yes", "No"]
            )

        st.divider()

        show_section(
            "02 · Symptom indicators",
            "Select binary demo values."
        )

        symptom_cols = st.columns(5)
        symptoms = {}

        for i, column in enumerate(symptom_cols, start=1):
            with column:
                symptoms[f"Symptom {i}"] = st.selectbox(
                    f"Symptom {i}",
                    [0, 1],
                    index=0
                )

        st.divider()

        show_section(
            "03 · Model selection",
            "Choose the saved classifier to test."
        )

        model_choice = st.selectbox(
            "Prediction model",
            list(models.keys())
        )

        submitted = st.form_submit_button(
            "Run Demo Prediction →",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            sample = pd.DataFrame(
                0,
                index=[0],
                columns=features
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

            st.markdown(
                '<div class="result">'
                '<div class="result-label">DEMO PREDICTION</div>'
                f'<div class="result-value">{subclass}</div>'
                f'<div class="panel-text">Model: {model_choice}</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "This is only a predicted dataset label, not a "
                "diagnosis. The form omits many trained features, "
                "and the earlier test accuracy was limited."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

    show_disclaimer()


# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    show_hero(
        "MODEL EVALUATION",
        "Model Performance",
        "Previously recorded holdout evaluation results. "
        "These metrics do not establish clinical reliability."
    )

    results = pd.DataFrame({
        "Model": [
            "Linear SVM",
            "Polynomial SVM",
            "RBF SVM"
        ],
        "Accuracy": [0.3940, 0.3534, 0.3969],
        "Macro Precision": [0.2203, 0.2811, 0.2601],
        "Macro Recall": [0.2566, 0.1960, 0.2314],
        "Macro F1": [0.2331, 0.1870, 0.2255]
    })

    show_section(
        "Holdout evaluation",
        "Recorded scores from the earlier model comparison."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        show_metric("LINEAR SVM ACCURACY", "39.40%")

    with c2:
        show_metric("POLYNOMIAL SVM ACCURACY", "35.34%")

    with c3:
        show_metric("RBF SVM ACCURACY", "39.69%")

    st.markdown("<br>", unsafe_allow_html=True)

    display_results = results.copy()

    for column in [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ]:
        display_results[column] = (
            display_results[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Accuracy and Macro F1 comparison")

    st.bar_chart(
        results.set_index("Model")[["Accuracy", "Macro F1"]]
    )

    st.warning(
        "The scores are limited, especially macro-F1. "
        "The models are not validated for medical screening."
    )

    show_disclaimer()


# ================= ABOUT PROJECT =================
elif page == "About Project":

    show_hero(
        "ABOUT THE PROJECT",
        "GeneScreen AI",
        "A student research demonstration for predicting "
        "genetic-disorder subclass labels."
    )

    show_section("Project overview")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Objective</div>
        <div class="panel-text">
            Explore a machine-learning workflow that uses
            saved Support Vector Machine classifiers to predict
            subclass labels from selected input features.
        </div>
        <br>
        <div class="panel-title">Algorithms</div>
        <div class="panel-text">
            • Linear Support Vector Machine<br>
            • Polynomial Support Vector Machine<br>
            • RBF Support Vector Machine
        </div>
        <br>
        <div class="panel-title">Technology stack</div>
        <div class="panel-text">
            Python · Streamlit · Pandas · NumPy · Scikit-learn · Joblib
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Known limitations</div>
        <div class="panel-text">
            • Earlier holdout accuracy was approximately 35–40%.<br><br>
            • The current form does not collect every training feature.<br><br>
            • The system predicts dataset labels and is not a validated
            medical screening tool.<br><br>
            • Predictions must not guide healthcare decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_disclaimer()
