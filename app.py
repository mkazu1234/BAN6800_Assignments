from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('xgboost_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data['input']
        
        # Ensure the correct number of features
        if len(input_data) != 23:  # Adjust this according to your model's feature requirements
            return jsonify({"error": "Input data must have 23 features."}), 400

        # Make a prediction
        prediction = model.predict([input_data])  # Ensure input_data is in the correct format
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return the error message for debugging

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)