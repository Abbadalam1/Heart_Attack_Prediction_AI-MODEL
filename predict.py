import sys
import json
import pandas as pd
import joblib
import numpy as np

def generate_plans(prediction, age, country):
    if prediction == "High Risk":
        diet = "Low-sodium, heart-healthy diet with fruits, vegetables, whole grains, and lean proteins. Avoid fried foods and processed sugars."
        exercise = "Moderate aerobic exercise like walking for 20-30 min, 5 days a week, as approved by a doctor."
        yoga = "Gentle yoga (e.g., Hatha or chair yoga) for 15 min daily to reduce stress and improve circulation."
        stress_management = "Practice meditation and deep breathing for 10-15 min daily to manage stress."
    else:
        diet = "Balanced diet with whole grains, fruits, vegetables, and lean proteins. Limit saturated fats."
        exercise = "Regular exercise like brisk walking or cycling for 30 min, 5 days a week."
        yoga = "Beginner yoga (e.g., Vinyasa) for 20 min daily to maintain flexibility."
        stress_management = "Mindfulness practices for 10 min daily to promote mental well-being."

    # Adjust plans based on age
    if int(age) >= 80:
        exercise = "Light exercise like walking or stretching for 15 min daily, as per doctor's advice."
        yoga = "Chair yoga or gentle stretching for 10 min daily to maintain mobility."

    # Adjust plans based on country
    if country.lower() == "india":
        diet += " Include millets and turmeric in meals for anti-inflammatory benefits."
        yoga += " Consider traditional Indian yoga practices like Surya Namaskar if suitable."

    return {
        "diet": diet,
        "exercise": exercise,
        "yoga": yoga,
        "stressManagement": stress_management
    }

def main():
    try:
        # Read input JSON
        input_json = sys.argv[1]
        input_data = json.loads(input_json)

        # Extract input features
        age = float(input_data.get('age', 50))
        gender = float(input_data.get('gender', 1))  # 1=male, 0=female
        heart_rate = float(input_data.get('heartRate', 70))
        systolic_bp = float(input_data.get('systolicBloodPressure', 120))
        diastolic_bp = float(input_data.get('diastolicBloodPressure', 80))
        blood_sugar = float(input_data.get('bloodSugar', 100))
        ck_mb = float(input_data.get('ckMb', 1.0))
        troponin = float(input_data.get('troponin', 0.01))
        country = input_data.get('country', 'India')

        # Load model and scaler
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Prepare input as DataFrame with feature names
        features = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 
                    'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
        input_features = pd.DataFrame([[age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin]], 
                                      columns=features)
        input_scaled = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_label = "High Risk" if prediction == 1 else "Low Risk"

        # Generate tailored plans
        plans = generate_plans(prediction_label, age, country)

        # Output result
        result = {
            "prediction": prediction_label,
            "plans": plans
        }
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()