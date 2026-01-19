import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import os

# 1. Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'model.bin')
images_dir = os.path.join(base_dir, 'images')

# 2. Load the model
with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# 3. Create images folder if missing
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 4. Generate and Save Plot
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=10, height=0.5, title="XGBoost Feature Importance (Top 10)")
output_path = os.path.join(images_dir, 'feature_importance.png')
plt.savefig(output_path, bbox_inches='tight')
print(f"Chart saved to {output_path}")
