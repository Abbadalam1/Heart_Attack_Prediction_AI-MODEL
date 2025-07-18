import sys
import json
import joblib
import pandas as pd

# Load model
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    print(json.dumps({'error': 'Model file not found'}))
    sys.exit(1)

# Get input from command line
try:
    input_data = json.loads(sys.argv[1])
except (IndexError, json.JSONDecodeError):
    print(json.dumps({'error': 'Invalid input data'}))
    sys.exit(1)

# Prepare input to match UCI dataset features
# CHANGE HERE: Adjust features to match your model inputs. UCI dataset expects 14 features.
# Since we only have age, troponin, ecgHeartRate, fill others with defaults.
input_dict = {
    'age': int(input_data.get('age', 50)),  # Default age: 50 if not provided
    'sex': 1,  # Default: male (1=male, 0=female)
    'cp': 0,   # Default: no chest pain
    'trestbps': 120,  # Default: resting blood pressure
    'chol': 200,      # Default: cholesterol
    'fbs': 0,         # Default: fasting blood sugar < 120 mg/dl
    'restecg': 1 if int(input_data.get('ecgHeartRate', 70)) > 100 else 0,  # Fixed: Use Python's conditional expression
    'thalach': int(input_data.get('ecgHeartRate', 70)),  # Max heart rate
    'exang': 0,       # Default: no exercise-induced angina
    'oldpeak': float(input_data.get('troponin', 0)) / 1000,  # Approximate troponin effect
    'slope': 2,       # Default: flat slope
    'ca': 0,          # Default: no major vessels colored
    'thal': 2         # Default: normal
}

# Convert to DataFrame
try:
    input_df = pd.DataFrame([input_dict])
except Exception as e:
    print(json.dumps({'error': f'Input processing failed: {str(e)}'}))
    sys.exit(1)

# Make prediction
try:
    prediction = model.predict(input_df)
    print(json.dumps({'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk'}))
except Exception as e:
    print(json.dumps({'error': f'Prediction failed: {str(e)}'}))
    sys.exit(1)