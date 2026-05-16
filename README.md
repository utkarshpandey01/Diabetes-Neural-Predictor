# 🩺 Diabetes Neural Predictor

A high-performance, full-stack Machine Learning web application designed to classify and predict diabetes risk patterns. This project features a predictive model served via a Flask backend and wrapped in an interactive, responsive frontend user interface.

🚀 **Live Deployment:** https://diabetes-neural-predictor.onrender.com

---

## 🖥️ Project Preview

* **Responsive Dashboard:** A clean, accessible web interface for inputting health metrics.
* **Instant Evaluation:** Real-time risk assessment driven by an optimized classification engine.
* **Scale-Independent Inference:** Seamless preprocessing using automated pipeline scaling to ensure accuracy.

---

## 🧠 Machine Learning Engine

The core predictive engine is trained on a comprehensive diabetes health dataset (including key indicators from data pools like the IDMIA diabetes tracking structures). 

### Model Architecture
* **Algorithm:** Logistic Regression (Binary Classification)
* **Preprocessing:** Features are normalized using `StandardScaler` to handle variances in metric bounds (e.g., Blood Pressure, Glucose levels, BMI).
* **Artifacts:** The trained classification model and data scaler are serialized into optimized `.pkl` pipelines for lightweight, low-latency production inference.

---

## 🛠️ Tech Stack & Ecosystem

| Component | Technologies Used |
| :--- | :--- |
| **Backend & Core** | Python, Flask, Gunicorn |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Jupyter Notebook |
| **Frontend UI** | HTML5, CSS3, JavaScript (Modern Responsive Layout) |
| **Deployment & DevOps** | Git, GitHub, Render Cloud Platform |

---

## 📁 Repository Architecture

```text
Diabetes-Prediction-System-Project-1/
├── templates/
│   └── index.html          # Interactive frontend user interface
├── app.py                  # Production Flask application backend router
├── requirements.txt        # Production environment dependencies
├── Diabetes_prediction.pkl # Serialized Logistic Regression model binary
├── scaler.pkl              # Serialized StandardScaler preprocessing pipeline
└── Diabetes_prediction.ipynb # Experimental Jupyter Notebook (EDA & Training)
