"""
Model Comparison Script
========================
Compare multiple models and choose the best one for your task.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import multiple models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import time

print("=" * 70)
print("MODEL COMPARISON FRAMEWORK")
print("=" * 70)

# Generate sample dataset
print("\n📊 Generating sample dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Dataset ready: {len(X_train)} training, {len(X_test)} testing samples")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Store results
results = []

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MODELS")
print("=" * 70)

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Measure training time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Score': cv_mean,
        'CV Std': cv_std,
        'Training Time (s)': training_time
    })
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   ⏱️  Time: {training_time:.4f}s")

# Create results DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL RESULTS (Sorted by Accuracy)")
print("=" * 70)
print("\n" + df_results.to_string(index=False))

# Find best model
best_model = df_results.iloc[0]
print("\n" + "=" * 70)
print("🏆 BEST MODEL")
print("=" * 70)
print(f"Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"F1-Score: {best_model['F1-Score']:.4f}")
print(f"Cross-Validation: {best_model['CV Score']:.4f} (±{best_model['CV Std']:.4f})")
print(f"Training Time: {best_model['Training Time (s)']:.4f}s")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy comparison
axes[0, 0].barh(df_results['Model'], df_results['Accuracy'], color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_xlim(0, 1)
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Precision, Recall, F1-Score
x = np.arange(len(df_results))
width = 0.25
axes[0, 1].bar(x - width, df_results['Precision'], width, label='Precision', color='lightcoral', edgecolor='black')
axes[0, 1].bar(x, df_results['Recall'], width, label='Recall', color='lightgreen', edgecolor='black')
axes[0, 1].bar(x + width, df_results['F1-Score'], width, label='F1-Score', color='lightyellow', edgecolor='black')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Detailed Metrics Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Cross-validation scores with error bars
axes[1, 0].barh(df_results['Model'], df_results['CV Score'], 
                xerr=df_results['CV Std'],
                color='lightblue', edgecolor='black', capsize=5)
axes[1, 0].set_xlabel('Cross-Validation Score')
axes[1, 0].set_title('Cross-Validation Performance (5-Fold)')
axes[1, 0].set_xlim(0, 1)
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Training time comparison
axes[1, 1].barh(df_results['Model'], df_results['Training Time (s)'], 
                color='lightgreen', edgecolor='black')
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_title('Training Time Comparison')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'model_comparison_results.png'")

# Save results to CSV
df_results.to_csv('model_comparison_results.csv', index=False)
print("✅ Results saved to 'model_comparison_results.csv'")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("\n1. Best Accuracy: " + df_results.iloc[0]['Model'])
print("2. Fastest Training: " + df_results.loc[df_results['Training Time (s)'].idxmin(), 'Model'])
print("3. Most Stable (CV): " + df_results.loc[df_results['CV Std'].idxmin(), 'Model'])

print("\n💡 TIP: Consider the trade-off between:")
print("   - Accuracy: How well the model performs")
print("   - Speed: How fast it trains and predicts")
print("   - Stability: How consistent it is (low CV standard deviation)")

print("\n" + "=" * 70)
print("DONE! 🎉")
print("=" * 70)
