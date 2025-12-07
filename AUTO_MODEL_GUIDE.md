# 🤖 Auto Model Selector - User Guide

## What is it?

An intelligent system that **automatically** analyzes your dataset and selects the best machine learning model. No need to manually test different models!

## How it Works

```
Your Data → Auto Detection → Model Selection → Training → Best Model ✨
```

1. **Auto-Detection**: Determines if your problem is classification, regression, or clustering
2. **Smart Selection**: Chooses appropriate models based on dataset size, features, and characteristics
3. **Comprehensive Testing**: Trains and evaluates multiple models
4. **Best Pick**: Recommends the model with best performance

## Quick Start

### Basic Usage

```python
from auto_model_selector import AutoModelSelector

# Create selector
selector = AutoModelSelector()

# Fit with your data (it does everything automatically!)
selector.fit(X, y)  # For supervised learning
# OR
selector.fit(X)     # For clustering (no y needed)

# Make predictions
predictions = selector.predict(new_data)
```

### With Your CSV File

```python
import pandas as pd
from auto_model_selector import AutoModelSelector

# Load your data
df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop('target_column', axis=1)
y = df['target_column']

# Let the system choose the best model
selector = AutoModelSelector()
selector.fit(X, y)

# Use the best model
predictions = selector.predict(X_new)
```

## What the System Detects

### Problem Type Detection

| Your Data | System Detects |
|-----------|----------------|
| Labels with few unique values (< 20) | **Classification** |
| Labels with many unique values | **Regression** |
| No labels provided | **Clustering** |

### Dataset Characteristics

The system analyzes:
- **Sample size**: Small (< 1000) vs Large (≥ 1000)
- **Features**: Number of input variables
- **Dimensionality**: Low vs High dimensional
- **Balance**: For classification, checks class distribution
- **Missing values**: Detects if data has gaps

### Model Selection Logic

**For Classification:**
- Small dataset → Logistic Regression, Decision Tree, KNN, Naive Bayes
- Large dataset → Random Forest, Gradient Boosting
- All datasets → Cross-validation for reliability

**For Regression:**
- Small dataset → Linear Regression, Ridge, KNN
- Large dataset → Random Forest, Gradient Boosting
- High-dimensional → Lasso Regression

**For Clustering:**
- All datasets → K-Means (most popular)
- Any size → DBSCAN (handles arbitrary shapes)
- Small dataset → Hierarchical Clustering

## Complete Examples

### Example 1: Classification (Customer Churn)

```python
from auto_model_selector import AutoModelSelector
import pandas as pd

# Your customer data
df = pd.read_csv('customers.csv')
X = df[['age', 'income', 'tenure', 'purchases']]
y = df['will_churn']  # 0 or 1

# Automatic model selection
selector = AutoModelSelector()
selector.fit(X, y)

# Predict for new customers
new_customers = [[35, 50000, 12, 5], [28, 35000, 6, 2]]
predictions = selector.predict(new_customers)
print(f"Churn predictions: {predictions}")

# Get the best model for deployment
best_model = selector.get_best_model()
```

**System Output:**
```
🔍 ANALYZING DATASET...
✓ Few unique values (2) → CLASSIFICATION problem

📊 Dataset Characteristics:
  Samples: 1000
  Features: 4
  Size: Large
  Classes: 2
  Balance: Balanced

🤖 SELECTING CANDIDATE MODELS...
✓ Evaluated 5 models

🏆 BEST MODEL RECOMMENDATION
✨ Selected Model: Random Forest
📝 Reason: Highest cross-validation score: 0.9250 | Test accuracy: 0.9200
```

### Example 2: Regression (House Prices)

```python
from auto_model_selector import AutoModelSelector
import pandas as pd

# Your housing data
df = pd.read_csv('houses.csv')
X = df[['size_sqft', 'bedrooms', 'age', 'location_score']]
y = df['price']

# Automatic model selection
selector = AutoModelSelector()
selector.fit(X, y)

# Predict house prices
new_houses = [[2000, 3, 10, 8.5], [1500, 2, 5, 7.0]]
predicted_prices = selector.predict(new_houses)
print(f"Predicted prices: ${predicted_prices}")
```

**System Output:**
```
🔍 ANALYZING DATASET...
✓ Many unique values (500) → REGRESSION problem

📊 Dataset Characteristics:
  Samples: 500
  Features: 4
  Size: Small

🏆 BEST MODEL RECOMMENDATION
✨ Selected Model: Random Forest
📝 Reason: Best R² score: 0.8750 | Lowest RMSE: 25000.00
```

### Example 3: Clustering (Customer Segmentation)

```python
from auto_model_selector import AutoModelSelector
import pandas as pd

# Your customer behavior data (no labels!)
df = pd.read_csv('customer_behavior.csv')
X = df[['annual_spending', 'visit_frequency', 'avg_purchase']]

# Automatic clustering (no y needed!)
selector = AutoModelSelector()
selector.fit(X)  # Notice: no y parameter

# Get cluster assignments
clusters = selector.predict(X)
df['segment'] = clusters

# Analyze segments
print(df.groupby('segment').mean())
```

**System Output:**
```
🔍 ANALYZING DATASET...
✓ No labels detected → CLUSTERING problem

📊 Dataset Characteristics:
  Samples: 300
  Features: 3
  Size: Small

🏆 BEST MODEL RECOMMENDATION
✨ Selected Model: K-Means
📝 Reason: Identified 3 clusters | Silhouette score: 0.6500
```

## Advanced Usage

### Custom Test Size

```python
# Use 30% of data for testing instead of default 20%
selector = AutoModelSelector()
selector.fit(X, y, test_size=0.3)
```

### Silent Mode

```python
# No verbose output
selector = AutoModelSelector(verbose=False)
selector.fit(X, y)
```

### Access All Results

```python
selector = AutoModelSelector()
selector.fit(X, y)

# Get all model results
import pandas as pd
results_df = pd.DataFrame(selector.results)
print(results_df)

# Get best model name
print(f"Best model: {selector.best_model_name}")
```

## Understanding the Output

### Classification Metrics

- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we catch?
- **F1-Score**: Balance between precision and recall
- **CV Score**: Cross-validation accuracy (most reliable)

### Regression Metrics

- **R² Score**: How well model explains variance (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **CV Score**: Cross-validated R² score

### Clustering Metrics

- **N Clusters**: Number of groups found
- **Silhouette Score**: How well-separated clusters are (-1 to 1, higher is better)
- **Inertia**: Within-cluster sum of squares (lower is better, for K-Means)

## Tips for Best Results

### Data Preparation

1. **Remove missing values** or fill them:
   ```python
   df = df.dropna()  # or
   df = df.fillna(df.mean())
   ```

2. **Encode categorical variables**:
   ```python
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['category'] = le.fit_transform(df['category'])
   ```

3. **Use numeric features only**:
   - The system expects numeric data
   - Convert text/categories to numbers first

### When to Use Each Problem Type

**Classification** (Predict categories):
- Customer will churn: Yes/No
- Email is spam: Spam/Not Spam
- Disease diagnosis: Healthy/Sick
- Product rating: 1-5 stars

**Regression** (Predict numbers):
- House price: $200,000
- Sales forecast: 150 units
- Temperature: 72°F
- Stock price: $45.50

**Clustering** (Find patterns):
- Customer segmentation
- Anomaly detection
- Market basket analysis
- Document grouping

## Troubleshooting

### "No labels detected → CLUSTERING"
**Issue**: System thinks it's clustering when you want classification/regression  
**Solution**: Make sure you're passing `y` parameter: `selector.fit(X, y)`

### "ValueError: could not convert string to float"
**Issue**: Your data has text/categorical values  
**Solution**: Encode categorical variables first

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_column'] = le.fit_transform(df['category_column'])
```

### Low Accuracy
**Solutions**:
1. Get more training data
2. Feature engineering (create better features)
3. Check for data quality issues
4. Try hyperparameter tuning on the selected model

### "All models performed poorly"
**Possible reasons**:
1. Data quality issues (missing values, outliers)
2. Features don't correlate with target
3. Need more data
4. Problem is too complex for traditional ML

## Comparison: Manual vs Automatic

### Manual Approach (Traditional)

```python
# You have to do all this manually:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# ... import many more

# Try model 1
model1 = LogisticRegression()
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

# Try model 2
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
score2 = model2.score(X_test, y_test)

# ... try many more models
# ... compare all results manually
# ... pick the best one
```

### Automatic Approach (Our System)

```python
# Everything happens automatically!
from auto_model_selector import AutoModelSelector

selector = AutoModelSelector()
selector.fit(X, y)  # Done! Best model selected.

predictions = selector.predict(new_data)
```

**Time saved**: Hours → Seconds  
**Models tested**: 5-7 models automatically  
**Best practices**: Built-in (cross-validation, scaling, etc.)

## Integration with Your Projects

### Save the Best Model

```python
import joblib

selector = AutoModelSelector()
selector.fit(X_train, y_train)

# Save for later use
best_model = selector.get_best_model()
joblib.dump(best_model, 'best_model.pkl')

# Load and use later
loaded_model = joblib.load('best_model.pkl')
predictions = loaded_model.predict(new_data)
```

### Use in Web Applications

```python
from flask import Flask, request, jsonify
from auto_model_selector import AutoModelSelector
import pandas as pd

app = Flask(__name__)

# Train once at startup
selector = AutoModelSelector()
selector.fit(training_X, training_y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = pd.DataFrame([data])
    prediction = selector.predict(X)
    return jsonify({'prediction': int(prediction[0])})

app.run()
```

## Real-World Use Cases

1. **E-commerce**: Customer churn prediction
2. **Finance**: Credit risk assessment
3. **Healthcare**: Disease prediction
4. **Marketing**: Customer segmentation
5. **Real Estate**: Price prediction
6. **Manufacturing**: Quality control
7. **Retail**: Demand forecasting

## Next Steps

1. ✅ Run the example: `python auto_model_selector.py`
2. ✅ Try with sample data
3. ✅ Use your own dataset
4. ✅ Integrate into your project
5. ✅ Share results!

## Support

Need help? Check:
- Example code in `auto_model_selector.py`
- Run examples: `python auto_model_selector.py`
- Review error messages carefully
- Ensure data is numeric and clean

---

**Happy Modeling!** 🚀

*The system does the hard work, you focus on insights!*
