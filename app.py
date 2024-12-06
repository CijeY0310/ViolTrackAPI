from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}}, supports_credentials=True)

app = Flask(__name__)

# Load models
model_paths = {
    'arima_model': os.path.join(os.path.dirname(__file__), 'arima_model3.pkl'),
    'logistic_regression_model': os.path.join(os.path.dirname(__file__), 'logistic_regression_model2.pkl'),
    'program_encoder': os.path.join(os.path.dirname(__file__), 'program_encoder2.pkl')
}

models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = joblib.load(f)

@app.route('/')
def home():
    return 'Server API is running'

@app.route('/predict/arima', methods=['POST'])
def predict_arima(): 
    try:
        data = request.json
        steps = data.get('steps', 12)

        arima_model = models['arima_model']
        predictions = arima_model.forecast(steps=steps)
        predictions_rounded = predictions.round().tolist()

        last_date = data.get('last_date', pd.Timestamp.now())
        future_dates = pd.date_range(start=last_date, periods=steps, freq='W').strftime('%Y-%m-%d').tolist()

        response = {
            "predictions": {
                "dates": future_dates,
                "values": predictions_rounded
            }
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    try:
        data = request.json

        # Input data
        program = data['Program']
        year_level = data['Year_Level']
        offense = data['Offense']

        # Prepare data for encoding and prediction
        input_data = pd.DataFrame({
            'Program': [program],
            'Year_Level': [year_level],
            'Offense': [offense]
        })

        # Encode the 'Program' feature using the encoder
        encoder = models['program_encoder']
        encoded_program = encoder.transform(input_data[['Program']]).toarray()
        encoded_program_df = pd.DataFrame(encoded_program, columns=encoder.get_feature_names_out(['Program']))

        # Combine encoded program with other features
        input_data_encoded = pd.concat([input_data.drop(['Program'], axis=1), encoded_program_df], axis=1)
        input_data_encoded = input_data_encoded.reindex(columns=models['logistic_regression_model'].feature_names_in_, fill_value=0)

        # Make prediction
        logistic_model = models['logistic_regression_model']
        probability = logistic_model.predict_proba(input_data_encoded)[0][1]
        prediction = logistic_model.predict(input_data_encoded)[0]

        # Response
        result = {
            'predicted_reoffend_status': int(prediction),
            'probability_of_reoffending': round(probability, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/predict/encoder', methods=['POST'])
def predict_encoder():
    try:
        data = request.json

        # Input program to encode
        program = data['Program']
        input_data = pd.DataFrame({'Program': [program]})

        # Encode using preloaded encoder
        encoder = models['program_encoder']
        encoded_program = encoder.transform(input_data[['Program']]).toarray()
        encoded_program_df = pd.DataFrame(encoded_program, columns=encoder.get_feature_names_out(['Program']))

        # Convert to dictionary for JSON response
        encoded_result = encoded_program_df.to_dict(orient='records')[0]
        return jsonify(encoded_result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
@app.route('/predict/logistic_bulk', methods=['POST'])
def predict_logistic_bulk():
    try:
        data_list = request.json  # Expecting a list of dictionaries

        # Verify that data_list is a list of dictionaries
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            return jsonify({"error": "Input data must be a list of dictionaries"}), 400

        results = []

        for data in data_list:
            # Extract data
            program = data['Program']
            year_level = data['Year_Level']
            offense = data['Offense']

            # Prepare data for encoding and prediction
            input_data = pd.DataFrame({
                'Program': [program],
                'Year_Level': [year_level],
                'Offense': [offense]
            })

            # Encode the 'Program' feature using the encoder
            encoder = models['program_encoder']
            encoded_program = encoder.transform(input_data[['Program']]).toarray()
            encoded_program_df = pd.DataFrame(encoded_program, columns=encoder.get_feature_names_out(['Program']))

            # Combine encoded program with other features
            input_data_encoded = pd.concat([input_data.drop(['Program'], axis=1), encoded_program_df], axis=1)
            input_data_encoded = input_data_encoded.reindex(columns=models['logistic_regression_model'].feature_names_in_, fill_value=0)

            # Make prediction
            logistic_model = models['logistic_regression_model']
            probability = logistic_model.predict_proba(input_data_encoded)[0][1]
            prediction = logistic_model.predict(input_data_encoded)[0]

            # Append each result to the list
            results.append({
                'input': data,
                'predicted_reoffend_status': int(prediction),
                'probability_of_reoffending': round(probability, 2)
            })

        # Return all predictions as a JSON array
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, Nice your API is now WORKING!'})

if __name__ == '__main__':
    app.run(port=5000)
