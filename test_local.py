import requests

# URL of your Flask app
url = 'http://localhost:9696/predict'

# Valid data that matches your schema
data = {
    "make": "bmw",
    "model": "3_series_gran_turismo",
    "year": 2017,
    "engine_hp": 240.0,
    "engine_cylinders": 4.0,
    "transmission_type": "automatic",
    "vehicle_style": "4dr_hatchback",
    "highway_mpg": 34,
    "city_mpg": 23
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=data)
    print("Response Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)