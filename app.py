from flask import Flask, request, render_template
import pickle
import numpy as np
import time  # For the loading delay
import os

app = Flask(__name__)

# Load the model and the scaler
# Ensure these files are in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # This creates an absolute path directly to your files regardless of the terminal location
    model_path = os.path.join(BASE_DIR, "Diabetes_prediction.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    print("✅ Model & Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():


    # Artificial delay to let the user see the cool loading animation
    time.sleep(2) 

    try:
        # Extract features using the exact names from your final HTML
        feature_list = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        # 1. Convert to 2D array
        raw_features = np.array([feature_list])
        
        # 2. Scale the features (Must match the training process)
        scaled_features = scaler.transform(raw_features)


        print("\n=== AI MODEL DEBUG LOG ===")
        print(f"1. Raw Input Array: {raw_features}")
        print(f"2. Scaled Input Array: {scaled_features}")
        
        prediction = model.predict(scaled_features)
        print(f"3. Model Output Array: {prediction}\n")
        
        # 3. Predict
        prediction = model.predict(scaled_features)
        
        # 4. Result formatting
        output = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        output = "Error in Processing"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    # use_reloader=False is best for Jupyter Notebook environments
    app.run(debug=True, use_reloader=False)