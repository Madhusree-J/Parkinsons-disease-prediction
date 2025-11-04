import streamlit as st
import pandas as pd
import joblib
import json
import os




st.set_page_config(
    page_title="Parkinson’s Disease Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #ffffff, #d9e6ff);
        font-family: "Poppins", sans-serif;
    }
    h1 {
        color: #004aad;
        text-align: center;
        font-size: 42px !important;
        font-weight: bold;
    }
    h5 {
        color: #0052cc;
        text-align: center;
        font-style: italic;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 50px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)





st.title("🩺 Parkinson’s Disease Detection from Voice Features")
st.markdown("<h5 style='text-align:center;'>An AI-based Health Prediction System</h5>", unsafe_allow_html=True)
st.write("---")

# Model Information Section
with st.expander("ℹ️ About this Model"):
    st.write("""
    This AI model analyzes biomedical voice features to detect early signs of Parkinson’s Disease.
    It uses a **Random Forest Classifier**, which is an ensemble of decision trees trained on the 
    Parkinson’s dataset from **Kaggle**. 

    Each data record represents a person’s voice sample with acoustic parameters such as:
    - Jitter, Shimmer (variations in pitch and amplitude)
    - Frequency and amplitude ratios
    - Harmonics-to-noise ratio (HNR)
    - Nonlinear measures like DFA and RPDE

    These features help the model learn differences between healthy and Parkinson’s-affected voices.
    """)


# Load model and features
model_path = "code/rf_parkinsons_v1.joblib"
feat_path = "code/selected_features.json"




# Load model safely
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        features = json.load(f)
    return model, features

model, features = load_model()

# File upload
st.subheader("📂 Upload Voice Feature Dataset")
uploaded_file = st.file_uploader("Choose a CSV file (voice features only)", type=["csv"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:")
    st.dataframe(data.head())

    # Check for required columns
    missing = [f for f in features if f not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        # Predict with progress spinner
        with st.spinner('🔍 Analyzing voice features... Please wait...'):
            preds = model.predict(data[features])

        results = ["🩺 Parkinson’s" if p == 1 else "✅ Healthy" for p in preds]

        # Show results
        # Show results beautifully
        data["Prediction"] = results
        st.subheader("📊 Prediction Results:")

        # Display overall summary
        parkinsons_count = results.count("🩺 Parkinson’s")
        healthy_count = results.count("✅ Healthy")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🩺 Parkinson’s Detected", parkinsons_count)
        with col2:
            st.metric("✅ Healthy", healthy_count)

        st.write(" ")

        # Show styled dataframe
        st.dataframe(data.style.set_properties(**{
            'background-color': '#f0f6ff',
            'color': '#000000',
            'border-color': 'gray'
        }))

        # Optional: Success message
        if parkinsons_count > 0:
            st.error("⚠️ Parkinson’s detected in one or more records.")
        else:
            st.success("🎉 All uploaded voices classified as Healthy.")



        # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="parkinsons_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file with the required voice measurement features.")


st.write("---")
st.markdown(
    """
    <div class='footer'>
        Developed by <b>Madhu Sree</b> | B.Tech IT | Internship Project <br>
        Vivekanandha College of Engineering for Women
    </div>
    """,
    unsafe_allow_html=True
)

