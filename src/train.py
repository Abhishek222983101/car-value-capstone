import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'car_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.bin')

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# 2. CLEANING
df.columns = df.columns.str.lower().str.replace(' ', '_')
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Filter features (Using the CORRECT name 'transmission_type')
features = ['make', 'model', 'year', 'engine_hp', 'engine_cylinders', 'transmission_type', 'vehicle_style', 'highway_mpg', 'city_mpg']
df = df[features + ['msrp']].dropna()

# Log Transform Target
df['msrp'] = np.log1p(df['msrp'])

# 3. SPLIT DATA (60/20/20)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train.msrp.values
y_val = df_val.msrp.values
y_test = df_test.msrp.values

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

# 4. VECTORIZE
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_val = dv.transform(df_val.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))

print("Data ready. Training models...")

# --- MODEL 1: RIDGE REGRESSION ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
# FIX: Use np.sqrt for RMSE
rmse_r = np.sqrt(mean_squared_error(y_val, ridge.predict(X_val)))
print(f"[1] Ridge RMSE: {rmse_r:.4f}")

# --- MODEL 2: RANDOM FOREST ---
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
# FIX: Use np.sqrt for RMSE
rmse_rf = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
print(f"[2] Random Forest RMSE: {rmse_rf:.4f}")

# --- MODEL 3: XGBOOST (With Tuning Loop) ---
print("[3] Tuning XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.get_feature_names_out().tolist())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.get_feature_names_out().tolist())

best_eta = 0.3
best_rmse = 999
best_model = None

# Explicit Tuning Loop
for eta in [0.3, 0.1, 0.05]:
    params = {
        'eta': eta, 'max_depth': 6, 'min_child_weight': 1,
        'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 0
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred = model.predict(dval)
    # FIX: Use np.sqrt for RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"   -> ETA={eta}: RMSE={rmse:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_eta = eta
        best_model = model

print(f"WINNER: XGBoost (eta={best_eta}) with RMSE: {best_rmse:.4f}")

# 5. FINAL SAVE
print(f"Saving model to {MODEL_PATH}...")
# Train on full train set before saving
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.msrp.values
del df_full_train['msrp']

X_full_train = dv.fit_transform(df_full_train.to_dict(orient='records'))
d_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())

final_params = {
    'eta': best_eta, 'max_depth': 6, 'min_child_weight': 1,
    'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 0
}
final_model = xgb.train(final_params, d_full_train, num_boost_round=100)

with open(MODEL_PATH, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)
print("Done!")