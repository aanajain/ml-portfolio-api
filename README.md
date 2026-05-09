# ML Portfolio API 🤖

A Flask REST API serving 5 Machine Learning models live.

## Models

| Project | Algorithm | Performance |
|--------|-----------|-------------|
| 🏠 House Price Prediction | Linear Regression | R² = 0.90 |
| 👥 Customer Segmentation | K-Means Clustering | 5 Clusters |
| 💬 Sentiment Analysis | NLP | 88% Accuracy |
| 🔍 Fraud Detection | Risk Scoring | 92% Accuracy |
| 📈 Sales Forecasting | Time Series | Seasonal Model |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/house-price` | POST | Predict house price |
| `/predict/segment` | POST | Find customer segment |
| `/predict/sentiment` | POST | Analyze text sentiment |
| `/predict/fraud` | POST | Detect fraud transaction |
| `/predict/sales` | POST | Forecast sales |

## Run Locally

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

API runs on `http://localhost:5000`

## Tech Stack
- Python
- Flask
- Flask-CORS
- NumPy
- Scikit-learn
- Gunicorn

## Live Demo
Deployed on Render: [ml-portfolio-api.onrender.com](https://ml-portfolio-api.onrender.com)

## Author
**Aana Jain** — Machine Learning Engineer
- GitHub: [@aanajain](https://github.com/aanajain)
