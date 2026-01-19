import pickle
import numpy as np
import xgboost as xgb  # <--- NEW IMPORT
from flask import Flask, request, jsonify

input_file = 'models/model.bin' 

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('car-price')

@app.route('/predict', methods=['POST'])
def predict():
    car = request.get_json()

    car_normalized = {}
    for key, value in car.items():
        if isinstance(value, str):
            car_normalized[key] = value.lower().replace(' ', '_')
        else:
            car_normalized[key] = value

    # 1. Transform to Matrix
    X = dv.transform([car_normalized])
    
    # 2. Create DMatrix (THE FIX)
    # The model crashes without this!
    features = dv.get_feature_names_out()
    dX = xgb.DMatrix(X, feature_names=features)

    # 3. Predict using the DMatrix
    y_pred = model.predict(dX)
    
    actual_price = np.expm1(y_pred)

    result = {
        'predicted_price_usd': float(actual_price[0]),
        'model': 'xgboost_tuned'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)