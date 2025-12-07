# 📋 Machine Learning Cheat Sheet

## 🎯 Model Selection Guide

### Use Case → Model Type

| What do you want to predict? | Model Type | Examples |
|------------------------------|------------|----------|
| Categories (Yes/No, Types) | **Classification** | Spam detection, Image recognition, Disease diagnosis |
| Numbers (Prices, Values) | **Regression** | House prices, Stock prices, Temperature |
| Natural groups in data | **Clustering** | Customer segmentation, Anomaly detection |

## 🤖 Popular Models Quick Reference

### Classification Models

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **Logistic Regression** | Binary classification, baseline | Fast, interpretable, works well with linear data | Limited to linear relationships |
| **Decision Tree** | Non-linear data, need interpretability | Easy to understand, handles non-linear data | Prone to overfitting |
| **Random Forest** | High accuracy needed, robust model | Very accurate, handles missing data well | Slow, less interpretable |
| **SVM** | High-dimensional data, clear margin | Effective in high dimensions | Slow on large datasets |
| **KNN** | Small datasets, simple approach | Simple, no training needed | Slow prediction, sensitive to scale |
| **Naive Bayes** | Text classification, fast predictions | Very fast, works with little data | Assumes feature independence |

### Regression Models

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **Linear Regression** | Linear relationships, baseline | Fast, interpretable, simple | Only linear relationships |
| **Polynomial Regression** | Curved relationships | Captures non-linear patterns | Can overfit easily |
| **Ridge/Lasso** | Many features, regularization needed | Prevents overfitting | Requires tuning |
| **Decision Tree Regressor** | Non-linear data, interpretability | Handles non-linearity well | Can overfit |
| **Random Forest Regressor** | High accuracy needed | Very accurate, robust | Slow, less interpretable |

### Clustering Models

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **K-Means** | Spherical clusters, known # clusters | Fast, simple | Must specify K, sensitive to outliers |
| **DBSCAN** | Irregular shapes, unknown # clusters | Finds arbitrary shapes, handles noise | Struggles with varying densities |
| **Hierarchical** | Need dendrogram, unknown # clusters | No need to specify K | Slow on large datasets |

## 📊 Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)      # Overall correctness
precision = precision_score(y_true, y_pred)    # Of predicted positives, how many correct?
recall = recall_score(y_true, y_pred)          # Of actual positives, how many caught?
f1 = f1_score(y_true, y_pred)                  # Harmonic mean of precision and recall
```

**When to use what:**
- **Accuracy**: Balanced dataset, all errors equally important
- **Precision**: False positives are costly (e.g., spam detection)
- **Recall**: False negatives are costly (e.g., disease detection)
- **F1-Score**: Need balance between precision and recall

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)       # Average squared error
rmse = np.sqrt(mse)                             # Root mean squared error
mae = mean_absolute_error(y_true, y_pred)      # Average absolute error
r2 = r2_score(y_true, y_pred)                  # R-squared (0-1, higher better)
```

**When to use what:**
- **RMSE**: Penalize large errors more (sensitive to outliers)
- **MAE**: All errors treated equally (robust to outliers)
- **R²**: How well model explains variance (0-1, 1 is perfect)

## 🔧 Common Code Snippets

### 1. Load and Split Data

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. Scale Features (Important!)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to scale:**
- ✅ SVM, KNN, Neural Networks, Logistic Regression
- ❌ Decision Trees, Random Forests (tree-based models)

### 3. Train a Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 4. Make Predictions

```python
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # For classification
```

### 5. Evaluate Model

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions))
```

### 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2%}")
```

### 8. Save and Load Models

```python
import joblib

# Save model
joblib.dump(model, 'my_model.pkl')

# Load model
loaded_model = joblib.load('my_model.pkl')
```

## 🎨 Visualization Templates

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### Feature Importance

```python
importances = model.feature_importances_
feature_names = ['feature1', 'feature2', ...]

plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### Learning Curve

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

## 🐛 Common Issues & Solutions

### Issue: Low Accuracy

**Solutions:**
1. Get more data
2. Feature engineering (create better features)
3. Try different models
4. Tune hyperparameters
5. Handle class imbalance (if classification)

### Issue: Overfitting (High training accuracy, low test accuracy)

**Solutions:**
1. Use cross-validation
2. Reduce model complexity (lower max_depth, fewer features)
3. Add regularization (Ridge, Lasso)
4. Get more training data
5. Use dropout (for neural networks)

### Issue: Underfitting (Low accuracy on both training and test)

**Solutions:**
1. Use more complex model
2. Add more features
3. Reduce regularization
4. Train longer
5. Remove outliers

### Issue: Model is too slow

**Solutions:**
1. Use simpler model (e.g., Logistic Regression instead of Random Forest)
2. Reduce number of features (feature selection)
3. Use fewer samples (if dataset is huge)
4. Use `n_jobs=-1` to parallelize
5. Consider approximate methods

## 📚 Quick Imports

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Train-test split
from sklearn.model_selection import train_test_split, cross_val_score

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
```

## 🎓 Best Practices

1. **Always split your data** - Never train and test on the same data
2. **Use cross-validation** - More reliable than single train-test split
3. **Scale features** - Especially for distance-based models
4. **Start simple** - Try simple models before complex ones
5. **Visualize everything** - Plots reveal insights numbers can't
6. **Handle missing data** - Impute or remove missing values
7. **Check for class imbalance** - Use appropriate techniques if imbalanced
8. **Document experiments** - Keep track of what works
9. **Validate assumptions** - Check if data meets model assumptions
10. **Use version control** - Track your code and models

## 🚀 Typical Workflow

```
1. Define Problem
   ↓
2. Collect Data
   ↓
3. Explore Data (EDA)
   ↓
4. Clean Data
   ↓
5. Feature Engineering
   ↓
6. Split Data (Train/Test)
   ↓
7. Choose Model
   ↓
8. Train Model
   ↓
9. Make Predictions
   ↓
10. Evaluate Performance
   ↓
11. Tune Hyperparameters
   ↓
12. Compare Models
   ↓
13. Select Best Model
   ↓
14. Deploy Model
```

## 📖 Resources

- **Scikit-learn Docs**: https://scikit-learn.org
- **Kaggle Learn**: https://www.kaggle.com/learn
- **Google ML Crash Course**: https://developers.google.com/machine-learning/crash-course
- **Fast.ai**: https://www.fast.ai

---

**Remember**: Practice makes perfect! Start with simple examples and gradually tackle more complex problems. 🎯
