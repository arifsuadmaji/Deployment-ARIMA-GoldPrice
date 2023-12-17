from flask import Flask, render_template, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import numpy as np

app = Flask(__name__)

# Placeholder for the dataframe
df = pd.DataFrame()

# Placeholder for the ARIMA model
results = None

# Placeholder for prediction results
prediction_df = pd.DataFrame(columns=['Date', 'Forecasting Rate'])

# Define function to make predictions
def make_arima_prediction(steps):
    global results, df, prediction_df

    if results is None:
        raise ValueError("ARIMA model is not trained. Please train the model first.")
    
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean

    # Generate future dates for the forecast
    future_dates = pd.date_range(start=df.index[-1], periods=steps + 1, freq='D')[1:]

    # Create a DataFrame for the forecast with formatted dates
    forecast_df = pd.DataFrame({'Waktu': future_dates.strftime('%Y-%m-%d'), 'Harga Emas': forecast_mean.values})
    
    # Update the global prediction_df
    prediction_df = forecast_df

    return forecast_df

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle data upload
@app.route('/upload', methods=['POST'])
def upload():
    global df
    # Get the uploaded file
    file = request.files['file']
    
    # Check if the file is of CSV type
    if file and file.filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(file, index_col='Date', parse_dates=['Date'])
        df = df['USD']
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Please upload a valid CSV file.'})

# Define route to handle model training
@app.route('/train', methods=['POST'])
def train():
    global df, results
    # Train ARIMA model
    auto_arima_model = pm.auto_arima(df, seasonal=False, stepwise=True, trace=True, suppress_warnings=True, error_action="ignore")
    order = auto_arima_model.get_params()['order']
    results = ARIMA(df, order=order).fit()
    
    print("Model trained successfully")  # Add this line

    return jsonify({'status': 'success'})


# Define route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the number of steps from the form
    steps = int(request.form['steps'])

    # Make ARIMA prediction
    arima_prediction = make_arima_prediction(steps)

    # Convert the prediction to JSON
    prediction_json = arima_prediction.to_json(orient='records')

    return jsonify({'prediction': prediction_json})

if __name__ == '__main__':
    app.run(debug=True)