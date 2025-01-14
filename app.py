from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import traceback
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}}, supports_credentials=True)

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

@app.route('/train/arima', methods=['POST'])
def train_arima():
    try:
        # Expecting new training data in JSON format
        data = request.json
        if 'data' not in data or 'date_column' not in data or 'value_column' not in data:
            return jsonify({"error": "Input data must include 'data', 'date_column', and 'value_column' fields"}), 400

        # Extract and process training data
        training_data = pd.DataFrame(data['data'])
        date_column = data['date_column']
        value_column = data['value_column']

        if date_column not in training_data.columns or value_column not in training_data.columns:
            return jsonify({"error": f"Columns '{date_column}' and '{value_column}' must exist in the input data"}), 400

        # Convert date column to datetime
        try:
            training_data[date_column] = pd.to_datetime(
                training_data[date_column], 
                format="%Y-%m-%d",  # Replace with the actual date format
                errors="coerce"
            )
        except Exception as e:
            return jsonify({"error": f"Date parsing error: {str(e)}"}), 400

        # Drop rows with invalid or missing dates
        training_data.dropna(subset=[date_column], inplace=True)
        training_data.set_index(date_column, inplace=True)
        training_data = training_data.resample('W').sum()  # Resample to weekly data

        # Train a new ARIMA model
        order = data.get('order', (5, 1, 0))  # Default order
        arima_model = ARIMA(training_data[value_column], order=order)
        model_fit = arima_model.fit()

        # Save the retrained model
        with open(model_paths['arima_model'], 'wb') as f:
            joblib.dump(model_fit, f)

        # Update the loaded model in memory
        models['arima_model'] = model_fit

        response = {
            "message": "ARIMA model retrained and saved successfully.",
            "aic": model_fit.aic,
            "bic": model_fit.bic
        }

        # Optionally, return predictions after retraining
        steps = data.get('steps', 12)
        predictions = model_fit.forecast(steps=steps).round().tolist()
        last_date = training_data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='W')[1:].strftime('%Y-%m-%d').tolist()

        response["predictions"] = {
            "dates": future_dates,
            "values": predictions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


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

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, Nice your API is now WORKING!'})

if __name__ == '__main__':
    app.run(port=5000)
