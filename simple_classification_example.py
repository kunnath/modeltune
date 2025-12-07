"""
Simple Classification Example
==============================
Predict whether a customer will buy a product based on their age and salary.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 300
ages = np.random.randint(18, 70, n_samples)
salaries = np.random.randint(20000, 120000, n_samples)

# Create target: buy product (1) or not (0)
# Rule: More likely to buy if age > 30 AND salary > 50000
purchase = ((ages > 30) & (salaries > 50000)).astype(int)
# Add some noise
noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
purchase[noise_indices] = 1 - purchase[noise_indices]

# Create DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Salary': salaries,
    'Purchase': purchase
})

print("=" * 50)
print("CUSTOMER PURCHASE PREDICTION")
print("=" * 50)
print(f"\nDataset Overview:")
print(df.head(10))
print(f"\nTotal Samples: {len(df)}")
print(f"Purchases: {purchase.sum()} ({purchase.sum()/len(df)*100:.1f}%)")
print(f"No Purchase: {len(df) - purchase.sum()} ({(len(df) - purchase.sum())/len(df)*100:.1f}%)")

# Prepare features and target
X = df[['Age', 'Salary']].values
y = df['Purchase'].values

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train the model
print("\n" + "=" * 50)
print("TRAINING MODEL...")
print("=" * 50)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy: {accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Purchase', 'Purchase']))

# Feature importance
importances = model.feature_importances_
print("\nFeature Importance:")
print(f"  Age: {importances[0]:.4f}")
print(f"  Salary: {importances[1]:.4f}")

# Make a prediction for a new customer
print("\n" + "=" * 50)
print("PREDICT FOR NEW CUSTOMER")
print("=" * 50)
new_customer = [[35, 65000]]  # 35 years old, $65,000 salary
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)[0]

print(f"Customer: Age {new_customer[0][0]}, Salary ${new_customer[0][1]:,}")
print(f"Prediction: {'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'}")
print(f"Confidence: {max(probability):.2%}")

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Data distribution
plt.subplot(1, 2, 1)
colors = ['red' if p == 0 else 'green' for p in y]
plt.scatter(df['Age'], df['Salary'], c=colors, alpha=0.6, edgecolors='black')
plt.xlabel('Age')
plt.ylabel('Salary ($)')
plt.title('Customer Purchase Behavior\n(Red=No, Green=Yes)')
plt.grid(True, alpha=0.3)

# Plot 2: Feature importance
plt.subplot(1, 2, 2)
plt.bar(['Age', 'Salary'], importances, color=['skyblue', 'lightcoral'], edgecolor='black')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'classification_results.png'")

print("\n" + "=" * 50)
print("DONE! 🎉")
print("=" * 50)
