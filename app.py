from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({'message': 'ML Flask API is running! 🚀'})

# House Price Prediction
@app.route('/predict/house-price', methods=['POST'])
def predict_house_price():
    data = request.get_json()
    area = float(data.get('area', 1000))
    bedrooms = float(data.get('bedrooms', 2))
    age = float(data.get('age', 5))
    price = (area * 150) + (bedrooms * 10000) - (age * 500)
    price = max(50000, price)
    return jsonify({
        'predicted_price': round(price, 2),
        'currency': 'USD',
        'message': f'Predicted house price for {area} sq ft'
    })

# Sentiment Analysis
@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'happy', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointing', 'sad']
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    if pos_count > neg_count:
        sentiment = 'Positive 😊'
        confidence = round(70 + (pos_count * 5), 1)
    elif neg_count > pos_count:
        sentiment = 'Negative 😞'
        confidence = round(70 + (neg_count * 5), 1)
    else:
        sentiment = 'Neutral 😐'
        confidence = 60.0
    return jsonify({
        'sentiment': sentiment,
        'confidence': min(confidence, 99.0),
        'text': text
    })

# Customer Segmentation
@app.route('/predict/segment', methods=['POST'])
def predict_segment():
    data = request.get_json()
    income = float(data.get('income', 50))
    spending = float(data.get('spending', 50))

    # Recreate the 5 cluster centers from Mall Customers dataset
    # Centers: [income, spending_score]
    cluster_centers = [
        [55, 49],   # Cluster 0 - Average Customers
        [87, 82],   # Cluster 1 - High Income High Spenders
        [26, 77],   # Cluster 2 - Low Income High Spenders
        [26, 20],   # Cluster 3 - Low Income Low Spenders
        [88, 17],   # Cluster 4 - High Income Low Spenders
    ]

    cluster_info = [
        {'type': '🎯 Average Customers', 'strategy': 'Maintain engagement', 'color': 'secondary'},
        {'type': '💰 High Income High Spenders', 'strategy': 'VIP treatment', 'color': 'success'},
        {'type': '⚠️ Low Income High Spenders', 'strategy': 'Offer discounts', 'color': 'warning'},
        {'type': '✅ Low Income Low Spenders', 'strategy': 'Budget deals', 'color': 'info'},
        {'type': '💸 High Income Low Spenders', 'strategy': 'Premium offers', 'color': 'primary'},
    ]

    # Find nearest cluster
    distances = []
    for center in cluster_centers:
        dist = np.sqrt((income - center[0])**2 + (spending - center[1])**2)
        distances.append(dist)

    cluster = int(np.argmin(distances))
    info = cluster_info[cluster]

    return jsonify({
        'cluster': cluster,
        'customer_type': info['type'],
        'strategy': info['strategy'],
        'color': info['color'],
        'income': income,
        'spending_score': spending
    })

# Fraud Detection
@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    amount = float(data.get('amount', 0))
    hour = int(data.get('hour', 12))
    distance = float(data.get('distance', 0))

    risk_score = 0

    if amount > 5000:
        risk_score += 40
    elif amount > 1000:
        risk_score += 20
    elif amount > 500:
        risk_score += 10

    if hour >= 0 and hour <= 5:
        risk_score += 30
    elif hour >= 22:
        risk_score += 20

    if distance > 500:
        risk_score += 30
    elif distance > 100:
        risk_score += 15

    if risk_score >= 60:
        result = 'Fraudulent 🚨'
        color = 'danger'
    elif risk_score >= 30:
        result = 'Suspicious ⚠️'
        color = 'warning'
    else:
        result = 'Legitimate ✅'
        color = 'success'

    return jsonify({
        'prediction': result,
        'risk_score': min(risk_score, 99),
        'color': color,
        'amount': amount
    })

# Sales Forecasting
@app.route('/predict/sales', methods=['POST'])
def predict_sales():
    data = request.get_json()
    month = int(data.get('month', 1))
    prev_sales = float(data.get('prev_sales', 10000))
    marketing = float(data.get('marketing', 5000))

    # Seasonal factors
    seasonal = [1.0, 0.9, 0.95, 1.0, 1.1, 1.2, 1.1, 1.0, 1.05, 1.1, 1.3, 1.5]
    season_factor = seasonal[month - 1]

    # Forecast formula
    forecast = (prev_sales * 1.05 + marketing * 0.3) * season_factor
    growth = ((forecast - prev_sales) / prev_sales) * 100

    return jsonify({
        'forecasted_sales': round(forecast, 2),
        'growth_percent': round(growth, 2),
        'month': month,
        'season_factor': season_factor
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)