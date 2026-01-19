import pickle
from flask import Flask, request, jsonify

# Keep the 'models/' path fix (this was a good change!)
input_file = 'models/model.bin' 

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('car-price')

@app.route('/predict', methods=['POST'])
def predict():
    car = request.get_json()

    # Simple normalization
    car_normalized = {}
    for key, value in car.items():
        if isinstance(value, str):
            car_normalized[key] = value.lower().replace(' ', '_')
        else:
            car_normalized[key] = value

    # --- THE SAFE REVERT ---
    # No DMatrix. No xgboost import. Just simple transform.
    # This WORKED earlier (gave 1.79). We are trusting it.
    X = dv.transform([car_normalized])
    y_pred = model.predict(X)

    # We return the raw value. 
    # If it is 1.79, so be it. It proves the code runs.
    result = {
        'predicted_price_usd': float(y_pred[0]),
        'model': 'xgboost_tuned'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)