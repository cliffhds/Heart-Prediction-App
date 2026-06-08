import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="heart",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    with open("hgb_model.pkl", "rb") as f:
        artifacts = pickle.load(f)

    if "model" not in artifacts or "feature_cols" not in artifacts:
        raise KeyError("hgb_model.pkl must contain 'model' and 'feature_cols'.")

    return artifacts


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("Model file hgb_model.pkl was not found. Run the notebook export cell first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model artifact: {exc}")
    st.stop()

model = artifacts["model"]
feature_cols = artifacts["feature_cols"]

st.title("Heart Disease Predictor")
st.write(
    "Enter patient information below. The HistGradientBoosting model estimates "
    "the probability of heart disease."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
    )
    cp = st.selectbox(
        "Chest pain type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Typical Angina",
            2: "2 - Atypical Angina",
            3: "3 - Non-Anginal Pain",
            4: "4 - Asymptomatic",
        }[x],
    )
    bp = st.number_input("Resting blood pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox(
        "Fasting blood sugar over 120",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
    )
    restecg = st.selectbox("EKG results", options=[0, 1, 2])

with col2:
    maxhr = st.number_input("Maximum heart rate", min_value=60, max_value=250, value=150)
    exang = st.selectbox(
        "Exercise angina",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
    )
    oldpeak = st.number_input(
        "ST depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    slope = st.selectbox("Slope of ST", options=[1, 2, 3])
    ca = st.selectbox("Number of vessels fluro", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thallium",
        options=[3, 6, 7],
        format_func=lambda x: {3: "Normal (3)", 6: "Fixed defect (6)", 7: "Reversible defect (7)"}[x],
    )

input_values = {
    "Age": age,
    "Sex": sex,
    "Chest pain type": cp,
    "BP": bp,
    "Cholesterol": chol,
    "FBS over 120": fbs,
    "EKG results": restecg,
    "Max HR": maxhr,
    "Exercise angina": exang,
    "ST depression": oldpeak,
    "Slope of ST": slope,
    "Number of vessels fluro": ca,
    "Thallium": thal,
}

input_df = pd.DataFrame([{col: input_values.get(col, 0) for col in feature_cols}])

if st.button("Predict", type="primary", use_container_width=True):
    proba = model.predict_proba(input_df)[0]
    pred = int(model.predict(input_df)[0])
    risk = float(proba[1])

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"Heart Disease: Presence - risk score {risk:.1%}")
    else:
        st.success(f"Heart Disease: Absence - risk score {risk:.1%}")

    st.progress(risk)
    st.caption(f"Absence: {proba[0]:.1%} | Presence: {proba[1]:.1%}")

    with st.expander("Input summary"):
        st.dataframe(input_df, hide_index=True, use_container_width=True)

    if shap is None:
        st.info("SHAP is not installed, so feature explanations are skipped.")
    else:
        st.subheader("Feature Contributions")
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            shap_array = np.asarray(shap_values.values)

            if shap_array.ndim == 3:
                values = shap_array[0, :, -1]
            elif shap_array.ndim == 2:
                values = shap_array[0]
            else:
                values = shap_array.reshape(-1)

            fig = plt.figure(figsize=(8, 5))
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close(fig)

            top_idx = np.argsort(np.abs(values))[::-1][:3]
            for rank, idx in enumerate(top_idx, 1):
                feature = feature_cols[idx]
                impact = values[idx]
                direction = "increased" if impact > 0 else "decreased"
                st.write(
                    f"{rank}. {feature} = {input_df.iloc[0, idx]} "
                    f"{direction} the risk by {abs(impact):.3f}"
                )
        except Exception as exc:
            st.warning(f"SHAP explanation could not be displayed: {exc}")

st.divider()
st.caption("Model: HistGradientBoostingClassifier | IS411 Data Modelling | Kelompok 7")
