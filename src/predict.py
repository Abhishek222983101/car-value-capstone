import warnings
warnings.filterwarnings("ignore")
import pickle
import os
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

# --- HYBRID IMPORT FIX ---
# This allows the script to work both inside Docker (src.schema) 
# and locally on your machine (schema)
try:
    from src.schema import CarInput
except ImportError:
    from schema import CarInput
# -------------------------

# 1. Load Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.bin')

# Fallback for Docker path (if running from root)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('models', 'model.bin')

with open(MODEL_PATH, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('car-price')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Validate (Pydantic)
        valid_data = CarInput(**data)
        
        # 2. Preprocess
        # use model_dump() to fix the warning
        car_dict = valid_data.model_dump()
        
        # Vectorize
        X = dv.transform([car_dict])
        
        # 3. Create DMatrix (Required for XGBoost)
        features = dv.get_feature_names_out().tolist()
        dtest = xgb.DMatrix(X, feature_names=features)
        
        # 4. Predict
        y_pred = model.predict(dtest)[0]
        price = float(np.expm1(y_pred)) # Reverse Log Transform

        return jsonify({
            'predicted_price_usd': round(price, 2),
            'model': 'xgboost_tuned',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)