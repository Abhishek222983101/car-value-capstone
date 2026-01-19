import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'data', 'car_data.csv'))
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.rename(columns={'msrp': 'price'}, inplace=True)

images_dir = os.path.join(base_dir, 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 2. Save Price Distribution (The Long Tail)
plt.figure(figsize=(6, 4))
sns.histplot(df.price, bins=50)
plt.title('Distribution of Prices (Long Tail)')
plt.xlabel('Price')
plt.ylabel('Count')
plt.savefig(os.path.join(images_dir, 'price_dist.png'), bbox_inches='tight')
print("Saved price_dist.png")

# 3. Save Log Distribution (The Normalization)
plt.figure(figsize=(6, 4))
sns.histplot(np.log1p(df.price), bins=50)
plt.title('Log Distribution of Prices (Normalized)')
plt.xlabel('Log(Price)')
plt.ylabel('Count')
plt.savefig(os.path.join(images_dir, 'price_log_dist.png'), bbox_inches='tight')
print("Saved price_log_dist.png")
