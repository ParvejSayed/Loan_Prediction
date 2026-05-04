import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Loan Approval Prediction")
st.markdown("Enter the applicant's details below to check loan eligibility.")
st.divider()

with st.form("loan_form"):
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        married = st.selectbox("Married", ["Yes", "No"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.divider()
    st.subheader("Financial Information")

    col4, col5 = st.columns(2)

    with col4:
        applicant_income = st.number_input(
            "Applicant Income (₹/month)", min_value=1, value=5000, step=500
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands ₹)", min_value=1.0, value=120.0, step=10.0
        )
        credit_history = st.selectbox(
            "Credit History",
            options=[1.0, 0.0],
            format_func=lambda x: "Good (meets guidelines)" if x == 1.0 else "Bad (does not meet guidelines)",
        )

    with col5:
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹/month)", min_value=0.0, value=0.0, step=500.0
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=12.0, value=360.0, step=12.0
        )

    st.divider()
    submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

if submitted:
    payload = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    with st.spinner("Analyzing your application..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                if result["prediction"] == 1:
                    st.success(f"✅ {result['message']}")
                    st.balloons()
                else:
                    st.error(f"❌ {result['message']}")

                with st.expander("View API Response"):
                    st.json(result)

            elif response.status_code == 422:
                st.warning(f"Validation Error: {response.json().get('detail', 'Invalid input')}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the prediction API. "
                "Make sure the FastAPI server is running: `uvicorn main:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server may be overloaded.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

st.divider()
st.caption("Powered by Gradient Boosting · FastAPI · Streamlit")
