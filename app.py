from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('personality_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature names as they appear in the dataset (for display)
feature_names = [
    'social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection',
    'group_comfort', 'party_liking', 'listening_skill', 'empathy', 'creativity',
    'organization', 'leadership', 'risk_taking', 'public_speaking_comfort',
    'curiosity', 'routine_preference', 'excitement_seeking', 'friendliness',
    'emotional_stability', 'planning', 'spontaneity', 'adventurousness',
    'reading_habit', 'sports_interest', 'online_social_usage', 'travel_desire',
    'gadget_usage', 'work_style_collaborative', 'decision_speed', 'stress_handling'
]

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from the form
        input_values = []
        for name in feature_names:
            val = request.form.get(name)
            if val is None or val.strip() == '':
                return render_template('index.html', error=f"Missing value for {name}", features=feature_names)
            input_values.append(float(val))

        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)

        # Scale the input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', error=str(e), features=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
