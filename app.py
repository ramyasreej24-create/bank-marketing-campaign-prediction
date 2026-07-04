"""
Streamlit app: Bank Term Deposit Subscription Predictor
=========================================================
Loads the trained "without_duration" pipeline (realistic, pre-call model)
and lets a user enter customer/campaign details to get a subscription
prediction + probability.

Run locally:
    streamlit run app.py

Expects `best_model_without_duration.pkl` in the same folder (or set
MODEL_PATH below / via the MODEL_PATH environment variable).
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_without_duration.pkl")

st.set_page_config(
    page_title="Term Deposit Subscription Predictor",
    page_icon="🏦",
    layout="centered",
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🏦 Term Deposit Subscription Predictor")
st.write(
    "Predict whether a customer is likely to subscribe to a term deposit "
    "**before making the call** — using only information available prior to "
    "contact (no call-duration data, to avoid leakage)."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Make sure "
        "`best_model_without_duration.pkl` is in the app directory, or set "
        "the `MODEL_PATH` environment variable."
    )
    st.stop()

model = load_model(MODEL_PATH)

with st.form("customer_form"):
    st.subheader("Customer & Campaign Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        job = st.selectbox(
            "Job",
            ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
             "retired", "self-employed", "services", "student", "technician",
             "unemployed", "unknown"],
        )
        marital = st.selectbox("Marital status", ["married", "single", "divorced"])
        education = st.selectbox(
            "Education", ["primary", "secondary", "tertiary", "unknown"]
        )
        default = st.selectbox("Has credit in default?", ["no", "yes"])
        balance = st.number_input(
            "Average yearly balance (euros)", value=1000, step=100
        )
        housing = st.selectbox("Has housing loan?", ["no", "yes"])
        loan = st.selectbox("Has personal loan?", ["no", "yes"])

    with col2:
        contact = st.selectbox(
            "Contact communication type", ["cellular", "telephone", "unknown"]
        )
        day = st.number_input(
            "Last contact day of month", min_value=1, max_value=31, value=15
        )
        month = st.selectbox(
            "Last contact month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
             "oct", "nov", "dec"],
        )
        campaign = st.number_input(
            "Number of contacts this campaign", min_value=1, value=1
        )
        previous = st.number_input(
            "Number of contacts before this campaign", min_value=0, value=0
        )
        pdays = st.number_input(
            "Days since last contact (-1 = never contacted before)",
            min_value=-1, value=-1,
        )
        poutcome = st.selectbox(
            "Outcome of previous campaign",
            ["unknown", "failure", "other", "success"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    was_contacted_before = 1 if pdays != -1 else 0

    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "was_contacted_before": was_contacted_before,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Yes — likely to subscribe" if proba >= 0.5 else "No — unlikely to subscribe"

    st.subheader("Result")
    st.metric("Subscription probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.success(prediction)
    else:
        st.info(prediction)

    st.caption(
        "Note: threshold is 0.5 by default. Adjust based on the relative cost "
        "of a wasted call vs. a missed subscriber."
    )

st.divider()
st.caption(
    "Model: Gradient Boosting trained on the UCI Bank Marketing dataset, "
    "excluding call `duration` to avoid data leakage (duration is only known "
    "after a call happens, so it can't inform who to call)."
)
