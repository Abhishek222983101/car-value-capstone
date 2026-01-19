import requests

# Your LIVE Render URL
url = 'https://car-value-capstone.onrender.com/predict'

# The car data
car = {
    "make": "bmw",
    "model": "3_series",
    "year": 2018,
    "engine_hp": 248.0,
    "engine_cylinders": 4.0,
    "transmission_type": "automatic",
    "vehicle_style": "sedan",
    "highway_mpg": 34,
    "city_mpg": 23
}

print(f"🚀 Sending request to: {url}")
response = requests.post(url, json=car)

if response.status_code == 200:
    print("✅ SUCCESS! Status Code: 200")
    print("🚗 Predicted Price:", response.json())
else:
    print("❌ FAILED.", response.text)
