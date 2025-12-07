# 🏋️ Machine Learning Practice Exercises

Complete these exercises to solidify your understanding of machine learning concepts.

## Level 1: Beginner Exercises

### Exercise 1.1: Modify Train-Test Split
**Goal**: Understand how data split affects model performance

**Task**:
1. Open `simple_classification_example.py`
2. Change `test_size=0.2` to `test_size=0.3`
3. Run the script and note the accuracy
4. Try with `test_size=0.1` and `test_size=0.4`

**Question**: How does the split ratio affect model performance?

---

### Exercise 1.2: Hyperparameter Tuning
**Goal**: Learn how parameters affect model behavior

**Task**:
1. Open `simple_classification_example.py`
2. Modify `RandomForestClassifier` parameters:
   ```python
   model = RandomForestClassifier(
       n_estimators=50,    # Try 50, 100, 200
       max_depth=5,        # Try 3, 5, 10, None
       random_state=42
   )
   ```
3. Run with different combinations

**Question**: Which parameter has the biggest impact on accuracy?

---

### Exercise 1.3: Feature Importance Analysis
**Goal**: Understand which features are most predictive

**Task**:
1. Use the iris dataset from sklearn
2. Train a Random Forest
3. Plot feature importance
4. Remove the least important feature and retrain

**Starter Code**:
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

# Your code here
```

---

## Level 2: Intermediate Exercises

### Exercise 2.1: Handle Imbalanced Data
**Goal**: Learn to deal with imbalanced classes

**Task**:
1. Create an imbalanced dataset (90% class 0, 10% class 1)
2. Train a model and observe accuracy
3. Try these solutions:
   - Use `class_weight='balanced'` in the model
   - Oversample minority class
   - Use F1-score instead of accuracy

**Starter Code**:
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],  # Imbalanced
    random_state=42
)

# Your solution here
```

---

### Exercise 2.2: Cross-Validation Deep Dive
**Goal**: Compare single split vs. cross-validation

**Task**:
1. Create a dataset with 500 samples
2. Evaluate model using:
   - Single train-test split (80-20)
   - 5-fold cross-validation
   - 10-fold cross-validation
3. Compare the stability of results

**Questions**:
- Which method gives more reliable estimates?
- What are the tradeoffs?

---

### Exercise 2.3: Feature Engineering
**Goal**: Create new features to improve model performance

**Task**:
1. Load the house price data from `simple_regression_example.py`
2. Create new features:
   - `price_per_sqft = price / size`
   - `size_category` (small, medium, large)
   - `bedroom_per_1000sqft = bedrooms / (size / 1000)`
3. Train models with and without new features
4. Compare R² scores

---

### Exercise 2.4: Model Comparison
**Goal**: Systematically compare different models

**Task**:
Create a comparison for regression models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

Compare using:
- R² Score
- RMSE
- Training time

---

## Level 3: Advanced Exercises

### Exercise 3.1: Hyperparameter Grid Search
**Goal**: Find optimal hyperparameters automatically

**Task**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Implement grid search
# Find best parameters
# Compare before and after tuning
```

---

### Exercise 3.2: Pipeline Creation
**Goal**: Build an end-to-end ML pipeline

**Task**:
Create a pipeline that:
1. Handles missing values
2. Scales features
3. Selects best features
4. Trains model
5. Makes predictions

**Starter Code**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])

# Your code here
```

---

### Exercise 3.3: Ensemble Methods
**Goal**: Combine multiple models for better predictions

**Task**:
1. Train 3 different models (Logistic Regression, Random Forest, SVM)
2. Create an ensemble by:
   - Voting (majority vote)
   - Averaging probabilities
   - Stacking (train a meta-model)
3. Compare ensemble vs. individual models

---

### Exercise 3.4: Learning Curves
**Goal**: Diagnose overfitting/underfitting

**Task**:
```python
from sklearn.model_selection import learning_curve

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot and interpret
# Is the model overfitting or underfitting?
```

---

## Level 4: Real-World Projects

### Project 4.1: Credit Card Fraud Detection
**Goal**: Build a complete fraud detection system

**Requirements**:
1. Handle imbalanced classes (fraud is rare)
2. Feature engineering from transaction data
3. Multiple model comparison
4. High recall (catch all fraud)
5. Reasonable precision (avoid false alarms)

**Metrics to optimize**: F1-score, Recall

---

### Project 4.2: Customer Churn Prediction
**Goal**: Predict which customers will leave

**Dataset Features**:
- Customer demographics
- Usage patterns
- Service history
- Support interactions

**Tasks**:
1. Exploratory Data Analysis
2. Feature engineering
3. Model selection
4. Interpret results (which factors drive churn?)
5. Business recommendations

---

### Project 4.3: Demand Forecasting
**Goal**: Predict product demand for inventory planning

**Requirements**:
1. Time series features (day, week, month, season)
2. Lag features (yesterday's sales, last week)
3. Handle missing data
4. Multiple regression models
5. Cross-validation with time series split

---

### Project 4.4: Image Classification
**Goal**: Classify images using traditional ML

**Tasks**:
1. Extract features from images (HOG, color histograms)
2. Try different classifiers
3. Optimize for accuracy
4. Handle class imbalance if present

**Dataset**: Use MNIST, Fashion-MNIST, or CIFAR-10

---

## Bonus Challenges

### Challenge 1: AutoML
Create a function that:
- Takes X, y as input
- Tries multiple models automatically
- Tunes hyperparameters
- Returns the best model

### Challenge 2: Feature Selection
Implement different feature selection methods:
- Filter methods (correlation, chi-square)
- Wrapper methods (RFE)
- Embedded methods (Lasso)

Compare their impact on model performance.

### Challenge 3: Anomaly Detection
Build an anomaly detection system:
- Use clustering (unusual points far from clusters)
- Use Isolation Forest
- Compare methods

### Challenge 4: Multi-class Classification
Extend binary classification to multi-class:
- One-vs-Rest strategy
- One-vs-One strategy
- Direct multi-class models

---

## Solutions & Tips

### General Tips:
1. **Start Simple**: Begin with basic models, add complexity gradually
2. **Visualize Often**: Plot your data, results, errors
3. **Document Everything**: Keep notes on what works
4. **Iterate**: ML is iterative - try, learn, improve
5. **Compare Fairly**: Use same data splits, same metrics

### Where to Find Datasets:
- **Sklearn**: `sklearn.datasets` (built-in)
- **Kaggle**: https://www.kaggle.com/datasets
- **UCI ML Repository**: https://archive.ics.uci.edu/ml
- **OpenML**: https://www.openml.org

### Evaluation Checklist:
- [ ] Split data properly (train/test)
- [ ] Scale features if needed
- [ ] Use appropriate metrics
- [ ] Compare multiple models
- [ ] Check for overfitting
- [ ] Validate with cross-validation
- [ ] Test on holdout set

---

## Progress Tracker

Mark your progress:

**Level 1 (Beginner)**:
- [ ] Exercise 1.1: Train-Test Split
- [ ] Exercise 1.2: Hyperparameter Tuning
- [ ] Exercise 1.3: Feature Importance

**Level 2 (Intermediate)**:
- [ ] Exercise 2.1: Imbalanced Data
- [ ] Exercise 2.2: Cross-Validation
- [ ] Exercise 2.3: Feature Engineering
- [ ] Exercise 2.4: Model Comparison

**Level 3 (Advanced)**:
- [ ] Exercise 3.1: Grid Search
- [ ] Exercise 3.2: Pipelines
- [ ] Exercise 3.3: Ensembles
- [ ] Exercise 3.4: Learning Curves

**Level 4 (Projects)**:
- [ ] Project 4.1: Fraud Detection
- [ ] Project 4.2: Churn Prediction
- [ ] Project 4.3: Demand Forecasting
- [ ] Project 4.4: Image Classification

**Bonus**:
- [ ] Challenge 1: AutoML
- [ ] Challenge 2: Feature Selection
- [ ] Challenge 3: Anomaly Detection
- [ ] Challenge 4: Multi-class Classification

---

**Good luck with your exercises!** 💪

Remember: The goal is to **understand**, not just to complete. Take your time, experiment, and learn from mistakes!
