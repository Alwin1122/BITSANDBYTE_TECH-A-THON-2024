from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import os

# Initialize the Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your_jwt_secret_key')  # Use environment variable or default value

# Initialize extensions
db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Stock model
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True)
    price = db.Column(db.Float)

# Define the Stock schema using SQLAlchemyAutoSchema
class StockSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Stock
        load_instance = True

# Authentication Endpoint
@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    if username == "admin" and password == "password":
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    return jsonify({"msg": "Bad username or password"}), 401

# Fetch Current Stock Data
@app.route('/api/stocks/<symbol>', methods=['GET'])
@jwt_required()
def get_stock(symbol):
    api_key = '50LVZX35U7MLT1NX'  # Replace with your actual API key
    response = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}')
    
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (5min)' in data:
            latest_time = list(data['Time Series (5min)'].keys())[0]
            latest_data = data['Time Series (5min)'][latest_time]
            return jsonify({
                "symbol": symbol,
                "price": float(latest_data['1. open'])
            })
    return jsonify({"error": "Stock data not found"}), 404

# Fetch Historical Stock Data
@app.route('/api/stocks/historical/<symbol>', methods=['GET'])
@jwt_required()
def get_historical_data(symbol):
    api_key = '50LVZX35U7MLT1NX'  # Replace with your actual API key
    response = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}')
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "Historical data not found"}), 404

# Predict Future Stock Price
@app.route('/api/stocks/predict/<symbol>', methods=['GET'])
@jwt_required()
def predict_stock(symbol):
    api_key = '50LVZX35U7MLT1NX'  # Replace with your actual API key
    response = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}')
    if response.status_code != 200:
        return jsonify({"error": "Historical data not found"}), 404

    historical_data = response.json()
    if 'Time Series (Daily)' not in historical_data:
        return jsonify({"error": "Invalid data format"}), 400

    df = pd.DataFrame.from_dict(historical_data['Time Series (Daily)'], orient='index')
    df = df[['4. close']].astype(float)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date', '4. close': 'close'}, inplace=True)

    if 'date' not in df.columns or 'close' not in df.columns:
        return jsonify({"error": "Invalid data format"}), 400

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['close'].values

    model = LinearRegression()
    model.fit(X, y)

    next_time_step = np.array([[len(df)]])
    prediction = model.predict(next_time_step)

    return jsonify({'predicted_price': prediction[0]})

# Swagger UI setup for API documentation
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_bp = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Stock Prediction API"}
)
app.register_blueprint(swaggerui_bp, url_prefix=SWAGGER_URL)

# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    response = {
        "error": str(e),
        "type": type(e).__name__
    }
    return jsonify(response), 500

# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database and tables
    app.run(debug=True)
