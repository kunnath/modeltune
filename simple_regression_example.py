"""
Simple Regression Example
==========================
Predict house prices based on size and number of bedrooms.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate sample data
n_samples = 200
house_size = np.random.randint(800, 3500, n_samples)  # sq ft
bedrooms = np.random.randint(1, 6, n_samples)

# Price formula: base + size factor + bedroom factor + noise
base_price = 50000
price = (
    base_price + 
    (house_size * 100) + 
    (bedrooms * 15000) + 
    np.random.normal(0, 20000, n_samples)  # Add noise
)

# Create DataFrame
df = pd.DataFrame({
    'Size_sqft': house_size,
    'Bedrooms': bedrooms,
    'Price': price
})

print("=" * 50)
print("HOUSE PRICE PREDICTION")
print("=" * 50)
print("\nDataset Overview:")
print(df.head(10))
print(f"\nDataset Statistics:")
print(df.describe())

# Prepare features and target
X = df[['Size_sqft', 'Bedrooms']].values
y = df['Price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train model
print("\n" + "=" * 50)
print("TRAINING MODEL...")
print("=" * 50)
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"R² Score: {r2:.4f} (1.0 is perfect)")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")

# Display model coefficients
print("\nModel Equation:")
print(f"Price = ${model.intercept_:,.2f}")
print(f"      + ${model.coef_[0]:,.2f} × Size(sqft)")
print(f"      + ${model.coef_[1]:,.2f} × Bedrooms")

# Make predictions for new houses
print("\n" + "=" * 50)
print("PREDICT PRICES FOR NEW HOUSES")
print("=" * 50)

new_houses = [
    [1500, 2],  # 1500 sqft, 2 bedrooms
    [2500, 3],  # 2500 sqft, 3 bedrooms
    [3000, 4],  # 3000 sqft, 4 bedrooms
]

for house in new_houses:
    predicted_price = model.predict([house])[0]
    print(f"House: {house[0]} sqft, {house[1]} bedrooms")
    print(f"  Predicted Price: ${predicted_price:,.2f}\n")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='black')
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Actual vs Predicted Prices')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Price vs Size
axes[1, 0].scatter(df['Size_sqft'], df['Price'], alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('House Size (sqft)')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title('Price vs House Size')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Price vs Bedrooms
bedroom_avg = df.groupby('Bedrooms')['Price'].mean()
axes[1, 1].bar(bedroom_avg.index, bedroom_avg.values, 
               color='skyblue', edgecolor='black')
axes[1, 1].set_xlabel('Number of Bedrooms')
axes[1, 1].set_ylabel('Average Price ($)')
axes[1, 1].set_title('Average Price by Bedrooms')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'regression_results.png'")

print("\n" + "=" * 50)
print("DONE! 🎉")
print("=" * 50)
