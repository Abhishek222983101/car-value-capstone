import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

input_file = 'models/model.bin' 

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('car-price')

@app.route('/predict', methods=['POST'])
def predict():
    car = request.get_json()

    # Normalize
    car_normalized = {}
    for key, value in car.items():
        if isinstance(value, str):
            car_normalized[key] = value.lower().replace(' ', '_')
        else:
            car_normalized[key] = value

    # 1. Transform to Matrix
    X = dv.transform([car_normalized])
    
    # 2. Get Feature Names (THE MISSING LINK)
    # We must explicitly pass these so XGBoost knows the columns match.
    features = list(dv.get_feature_names_out())
    
    # 3. Create DMatrix with Names
    dX = xgb.DMatrix(X, feature_names=features)

    # 4. Predict
    y_pred = model.predict(dX)
    
    # 5. Convert to Dollars
    actual_price = np.expm1(y_pred)

    result = {
        'predicted_price_usd': float(actual_price[0]),
        'model': 'xgboost_tuned'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)