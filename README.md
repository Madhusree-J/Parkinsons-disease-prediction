# 🧠 Parkinson’s Disease Prediction

This project predicts the presence of **Parkinson’s Disease** using **voice measurement features**.  
It applies **Machine Learning (Random Forest Classifier)** to analyze biomedical voice data and classify whether a person is healthy or affected.

---

## 📘 Project Overview
- Built an AI-based health prediction system using **Python** and **Streamlit**.  
- Trained a **Random Forest model** on voice feature datasets.  
- Created a **web app interface** that allows users to upload a CSV file and view prediction results instantly.  
- The app displays whether the patient has **Parkinson’s Disease** or is **Healthy** based on their voice data.

---

## ⚙️ Technologies Used
- **Python 3.10+**
- **Libraries:** Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn
- **Tools:** JupyterLab, VS Code, GitHub

---

## 📁 Project Structure
Parkinsons-Disease-Prediction/
│
├── app/ → Streamlit Web App
│ └── app.py
│
├── code/ → Trained Model Files
│ ├── rf_parkinsons_v1.joblib
│ └── selected_features.json
│
├── data/ → Dataset Files
│ └── parkinsons.csv
│
├── notebooks/ → Jupyter Notebooks
│ ├── 01_data_exploration.ipynb
│ └── 02_modeling.ipynb
│
├── slides/ → Project Presentation
│
└── README.md


---

## 🚀 How to Run the Project

1. **Open the terminal** inside your project folder.  
2. **Activate the virtual environment**  
   ```bash
   venv\Scripts\activate
3. Run the Streamlit app
   streamlit run app/app.py
4. Once the app opens in your browser:
   Upload a CSV file containing the voice measurement features.
   View prediction results (🩺 Parkinson’s / ✅ Healthy).

**📊 Dataset**

The project uses the Parkinson’s dataset from the UCI Machine Learning Repository.
It includes biomedical voice measurements from patients and healthy individuals.
Dataset link: https://archive.ics.uci.edu/ml/datasets/parkinsons

**Acknowledgment**
This project was developed as part of the Mini Project for college under the theme AI for Health Prediction.
It demonstrates the use of Machine Learning and Streamlit for early detection of Parkinson’s Disease.
