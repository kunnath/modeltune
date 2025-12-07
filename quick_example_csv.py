"""
Quick Example: Using Auto Model Selector with Your CSV Data
=============================================================
"""

from auto_model_selector import AutoModelSelector
import pandas as pd
import numpy as np

print("="*70)
print("  QUICK EXAMPLE: AUTO MODEL SELECTOR WITH CSV DATA")
print("="*70)

# Example 1: Load from CSV and predict
print("\n\n" + "="*70)
print("EXAMPLE 1: Classification from CSV")
print("="*70)
print("\nScenario: Predict customer churn from CSV file\n")

# Simulate loading from CSV (in real use, replace with your file)
print("# In your code:")
print("df = pd.read_csv('customers.csv')")
print()

# Create sample data to simulate CSV
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 120000, 100),
    'tenure_months': np.random.randint(1, 60, 100),
    'total_purchases': np.random.randint(0, 50, 100),
    'will_churn': np.random.choice([0, 1], 100)
})

print("Dataset preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# Separate features and target
X = df.drop('will_churn', axis=1)
y = df['will_churn']

print("\n# Step 1: Create selector")
print("selector = AutoModelSelector()")

print("\n# Step 2: Fit (automatically detects and selects best model)")
print("selector.fit(X, y)")

# Actually run it
selector = AutoModelSelector(verbose=False)  # Silent for this example
selector.fit(X, y)

print(f"\n✅ Done! Best model selected: {selector.best_model_name}")

# Make predictions
print("\n# Step 3: Predict for new customers")
print("new_customers = [[35, 50000, 24, 10], [60, 80000, 48, 30]]")
print("predictions = selector.predict(new_customers)")

new_customers = [[35, 50000, 24, 10], [60, 80000, 48, 30]]
predictions = selector.predict(new_customers)

print(f"\nPredictions: {predictions}")
print("Interpretation: 0 = Won't churn, 1 = Will churn")

# Example 2: With your own CSV
print("\n\n" + "="*70)
print("EXAMPLE 2: Using Your Own CSV File")
print("="*70)

code_example = '''
import pandas as pd
from auto_model_selector import AutoModelSelector

# Step 1: Load your data
df = pd.read_csv('your_data.csv')

# Step 2: Prepare features and target
X = df.drop('target_column_name', axis=1)  # All columns except target
y = df['target_column_name']                # Your target/label column

# Step 3: Let the system work its magic!
selector = AutoModelSelector()
selector.fit(X, y)

# Step 4: Make predictions on new data
new_data = pd.read_csv('new_data.csv')
predictions = selector.predict(new_data)

print(f"Predictions: {predictions}")

# Optional: Get the best model for deployment
best_model = selector.get_best_model()

# Optional: Save the model
import joblib
joblib.dump(best_model, 'my_best_model.pkl')
'''

print(code_example)

# Example 3: Clustering (no labels)
print("\n" + "="*70)
print("EXAMPLE 3: Clustering (No Labels Needed)")
print("="*70)

code_clustering = '''
import pandas as pd
from auto_model_selector import AutoModelSelector

# Load data without labels
df = pd.read_csv('customer_behavior.csv')

# Use all columns as features (no target needed!)
X = df[['annual_spending', 'visit_frequency', 'purchase_size']]

# System automatically detects it's clustering
selector = AutoModelSelector()
selector.fit(X)  # Notice: NO y parameter!

# Get cluster assignments
clusters = selector.predict(X)
df['segment'] = clusters

# Analyze segments
print(df.groupby('segment').mean())
'''

print(code_clustering)

print("\n" + "="*70)
print("KEY POINTS")
print("="*70)
print("""
✅ THREE SIMPLE STEPS:
   1. selector = AutoModelSelector()
   2. selector.fit(X, y)  # or just fit(X) for clustering
   3. predictions = selector.predict(new_data)

✅ SYSTEM AUTOMATICALLY:
   - Detects problem type (classification/regression/clustering)
   - Selects appropriate models
   - Trains and evaluates all models
   - Recommends the best one
   - Explains why

✅ WORKS WITH:
   - Pandas DataFrames
   - NumPy arrays
   - CSV files (after pd.read_csv)
   - Any numeric data

✅ NO NEED TO:
   - Manually try different models
   - Know which model to use
   - Write evaluation code
   - Compare results manually

The system does all the hard work for you! 🚀
""")

print("\n" + "="*70)
print("TRY IT NOW")
print("="*70)
print("""
Option 1: Run the main example
   python auto_model_selector.py

Option 2: Use with your CSV
   python
   >>> from auto_model_selector import AutoModelSelector
   >>> import pandas as pd
   >>> df = pd.read_csv('your_file.csv')
   >>> X = df.drop('target', axis=1)
   >>> y = df['target']
   >>> selector = AutoModelSelector()
   >>> selector.fit(X, y)

Option 3: Read the complete guide
   cat AUTO_MODEL_GUIDE.md
""")

print("\n" + "="*70)
print("✅ YOU'RE READY TO GO!")
print("="*70)
