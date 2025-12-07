"""
Intelligent Auto-Model Selector
================================
Automatically analyzes your dataset and selects the best machine learning model.

Features:
- Automatic problem type detection (classification, regression, clustering)
- Smart model selection based on dataset characteristics
- Comprehensive model comparison
- Best model recommendation with reasoning
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, List, Any
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

warnings.filterwarnings('ignore')


class AutoModelSelector:
    """
    Intelligent system that automatically detects problem type and selects
    the best machine learning model for your dataset.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.problem_type = None
        self.best_model = None
        self.best_model_name = None
        self.results = []
        self.scaler = StandardScaler()
        
    def _print(self, message):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def detect_problem_type(self, X, y=None) -> str:
        """
        Automatically detect the type of ML problem.
        
        Returns:
            'classification', 'regression', or 'clustering'
        """
        self._print("\n" + "="*70)
        self._print("🔍 ANALYZING DATASET...")
        self._print("="*70)
        
        # If no labels provided, it's clustering
        if y is None:
            self._print("✓ No labels detected → CLUSTERING problem")
            return 'clustering'
        
        # Check if y is numeric
        if isinstance(y, pd.Series):
            y = y.values
        
        unique_values = len(np.unique(y))
        n_samples = len(y)
        
        self._print(f"Dataset size: {n_samples} samples")
        self._print(f"Number of unique target values: {unique_values}")
        
        # Check if y is floating point - strong indicator of regression
        is_float = np.issubdtype(y.dtype, np.floating)
        
        # Check if y contains non-integer values
        has_decimals = not np.all(np.equal(np.mod(y, 1), 0))
        
        # If y is floating point or has decimal values, it's regression
        if is_float or has_decimals:
            self._print(f"✓ Continuous values detected → REGRESSION problem")
            return 'regression'
        
        # Heuristic: if unique values < 15 and < 5% of samples, it's classification
        # Otherwise treat as regression (even for integer counts)
        if unique_values < 15 and unique_values < 0.05 * n_samples:
            self._print(f"✓ Few unique values ({unique_values}) → CLASSIFICATION problem")
            return 'classification'
        else:
            self._print(f"✓ Many unique values ({unique_values}) → REGRESSION problem")
            return 'regression'
    
    def get_dataset_characteristics(self, X, y=None) -> Dict:
        """Analyze dataset characteristics to help with model selection."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
            
        n_samples, n_features = X.shape
        
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'is_small_dataset': n_samples < 1000,
            'is_high_dimensional': n_features > 50,
            'has_missing_values': np.isnan(X).any()
        }
        
        if y is not None:
            characteristics['n_classes'] = len(np.unique(y))
            characteristics['is_balanced'] = self._check_balance(y)
        
        self._print("\n📊 Dataset Characteristics:")
        self._print(f"  Samples: {n_samples}")
        self._print(f"  Features: {n_features}")
        self._print(f"  Size: {'Small' if characteristics['is_small_dataset'] else 'Large'}")
        
        if y is not None and self.problem_type == 'classification':
            self._print(f"  Classes: {characteristics['n_classes']}")
            self._print(f"  Balance: {'Balanced' if characteristics['is_balanced'] else 'Imbalanced'}")
        
        return characteristics
    
    def _check_balance(self, y) -> bool:
        """Check if classes are balanced (for classification)."""
        unique, counts = np.unique(y, return_counts=True)
        ratios = counts / len(y)
        # Consider balanced if no class is < 30% or > 70%
        return all(0.3 <= ratio <= 0.7 for ratio in ratios)
    
    def get_classification_models(self, characteristics: Dict) -> Dict:
        """Get appropriate classification models based on dataset characteristics."""
        models = {}
        
        # Always include these baseline models
        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
        
        # Random Forest - good for most cases
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100 if characteristics['n_samples'] < 5000 else 50,
            random_state=42
        )
        
        # For smaller datasets, add KNN and Naive Bayes
        if characteristics['is_small_dataset']:
            models['K-Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=5)
            models['Naive Bayes'] = GaussianNB()
        
        # For larger datasets with more features, add Gradient Boosting
        if not characteristics['is_small_dataset']:
            models['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        
        # SVM for smaller datasets (it's slower)
        if characteristics['n_samples'] < 5000 and not characteristics['is_high_dimensional']:
            models['SVM'] = SVC(random_state=42)
        
        return models
    
    def get_regression_models(self, characteristics: Dict) -> Dict:
        """Get appropriate regression models based on dataset characteristics."""
        models = {}
        
        # Baseline models
        models['Linear Regression'] = LinearRegression()
        models['Ridge Regression'] = Ridge(random_state=42)
        models['Decision Tree'] = DecisionTreeRegressor(random_state=42)
        
        # Random Forest - good for most cases
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=100 if characteristics['n_samples'] < 5000 else 50,
            random_state=42
        )
        
        # For smaller datasets
        if characteristics['is_small_dataset']:
            models['K-Nearest Neighbors'] = KNeighborsRegressor(n_neighbors=5)
        
        # For larger datasets
        if not characteristics['is_small_dataset']:
            models['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        
        # For high-dimensional data, add Lasso
        if characteristics['is_high_dimensional']:
            models['Lasso Regression'] = Lasso(random_state=42)
        
        return models
    
    def get_clustering_models(self, characteristics: Dict) -> Dict:
        """Get appropriate clustering models based on dataset characteristics."""
        models = {}
        
        # K-Means - most common
        n_clusters = min(8, max(2, characteristics['n_samples'] // 100))
        models['K-Means'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # DBSCAN - good for arbitrary shapes
        models['DBSCAN'] = DBSCAN(eps=0.5, min_samples=5)
        
        # Hierarchical - good for smaller datasets
        if characteristics['n_samples'] < 5000:
            models['Hierarchical'] = AgglomerativeClustering(n_clusters=n_clusters)
        
        return models
    
    def train_and_evaluate_classification(self, X, y, models: Dict, test_size=0.2):
        """Train and evaluate classification models."""
        self._print("\n" + "="*70)
        self._print("🚀 TRAINING AND EVALUATING MODELS...")
        self._print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        for name, model in models.items():
            self._print(f"\n🔧 Training {name}...")
            
            start_time = time.time()
            
            # Train
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle binary vs multi-class
            average = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            precision = precision_score(y_test, y_pred, average=average, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Score': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Training Time (s)': training_time,
                'model_object': model
            })
            
            self._print(f"  ✓ Accuracy: {accuracy:.4f}")
            self._print(f"  ✓ F1-Score: {f1:.4f}")
            self._print(f"  ✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return results
    
    def train_and_evaluate_regression(self, X, y, models: Dict, test_size=0.2):
        """Train and evaluate regression models."""
        self._print("\n" + "="*70)
        self._print("🚀 TRAINING AND EVALUATING MODELS...")
        self._print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        for name, model in models.items():
            self._print(f"\n🔧 Training {name}...")
            
            start_time = time.time()
            
            # Train
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            
            results.append({
                'Model': name,
                'R² Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV Score': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Training Time (s)': training_time,
                'model_object': model
            })
            
            self._print(f"  ✓ R² Score: {r2:.4f}")
            self._print(f"  ✓ RMSE: {rmse:.4f}")
            self._print(f"  ✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return results
    
    def train_and_evaluate_clustering(self, X, models: Dict):
        """Train and evaluate clustering models."""
        self._print("\n" + "="*70)
        self._print("🚀 TRAINING AND EVALUATING MODELS...")
        self._print("="*70)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        
        for name, model in models.items():
            self._print(f"\n🔧 Training {name}...")
            
            try:
                start_time = time.time()
                
                # Train/Predict
                labels = model.fit_predict(X_scaled)
                training_time = time.time() - start_time
                
                # Evaluate
                n_clusters = len(np.unique(labels))
                
                # Silhouette score (only if we have valid clusters)
                if n_clusters > 1 and n_clusters < len(X_scaled):
                    silhouette = silhouette_score(X_scaled, labels)
                else:
                    silhouette = -1
                
                # Inertia (for K-Means)
                inertia = model.inertia_ if hasattr(model, 'inertia_') else None
                
                results.append({
                    'Model': name,
                    'N Clusters': n_clusters,
                    'Silhouette Score': silhouette,
                    'Inertia': inertia,
                    'Training Time (s)': training_time,
                    'model_object': model
                })
                
                self._print(f"  ✓ Clusters: {n_clusters}")
                self._print(f"  ✓ Silhouette: {silhouette:.4f}")
                
            except Exception as e:
                self._print(f"  ✗ Failed: {str(e)}")
        
        return results
    
    def select_best_model(self, results: List[Dict]) -> Tuple[Any, str, str]:
        """Select the best model based on results."""
        if self.problem_type == 'classification':
            # Sort by CV Score (most reliable), then Accuracy
            best = max(results, key=lambda x: (x['CV Score'], x['Accuracy']))
            metric_name = 'CV Score'
            metric_value = best['CV Score']
            
        elif self.problem_type == 'regression':
            # Sort by CV Score (R²)
            best = max(results, key=lambda x: x['CV Score'])
            metric_name = 'CV R² Score'
            metric_value = best['CV Score']
            
        else:  # clustering
            # Sort by Silhouette Score
            valid_results = [r for r in results if r['Silhouette Score'] > -1]
            if valid_results:
                best = max(valid_results, key=lambda x: x['Silhouette Score'])
                metric_name = 'Silhouette Score'
                metric_value = best['Silhouette Score']
            else:
                best = results[0]
                metric_name = 'N/A'
                metric_value = 0
        
        reason = self._get_recommendation_reason(best, results)
        
        return best['model_object'], best['Model'], reason
    
    def _get_recommendation_reason(self, best: Dict, all_results: List[Dict]) -> str:
        """Generate explanation for why this model was chosen."""
        reasons = []
        
        if self.problem_type == 'classification':
            reasons.append(f"Highest cross-validation score: {best['CV Score']:.4f}")
            reasons.append(f"Test accuracy: {best['Accuracy']:.4f}")
            reasons.append(f"F1-Score: {best['F1-Score']:.4f}")
            
        elif self.problem_type == 'regression':
            reasons.append(f"Best R² score: {best['R² Score']:.4f}")
            reasons.append(f"Lowest RMSE: {best['RMSE']:.4f}")
            
        else:  # clustering
            reasons.append(f"Identified {best['N Clusters']} clusters")
            if best['Silhouette Score'] > -1:
                reasons.append(f"Silhouette score: {best['Silhouette Score']:.4f}")
        
        # Add training time consideration
        if best['Training Time (s)'] < 1:
            reasons.append("Fast training time")
        
        return " | ".join(reasons)
    
    def fit(self, X, y=None, test_size=0.2):
        """
        Main method: Analyze dataset and automatically select best model.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            y: Target variable (None for clustering)
            test_size: Proportion of test data (for supervised learning)
        
        Returns:
            self
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
        
        # Step 1: Detect problem type
        self.problem_type = self.detect_problem_type(X, y)
        
        # Step 2: Analyze dataset
        characteristics = self.get_dataset_characteristics(X, y)
        
        # Step 3: Get appropriate models
        self._print("\n" + "="*70)
        self._print("🤖 SELECTING CANDIDATE MODELS...")
        self._print("="*70)
        
        if self.problem_type == 'classification':
            models = self.get_classification_models(characteristics)
            self.results = self.train_and_evaluate_classification(X, y, models, test_size)
            
        elif self.problem_type == 'regression':
            models = self.get_regression_models(characteristics)
            self.results = self.train_and_evaluate_regression(X, y, models, test_size)
            
        else:  # clustering
            models = self.get_clustering_models(characteristics)
            self.results = self.train_and_evaluate_clustering(X, models)
        
        self._print(f"\n✓ Evaluated {len(models)} models")
        
        # Step 4: Select best model
        self.best_model, self.best_model_name, reason = self.select_best_model(self.results)
        
        # Step 5: Display results
        self.display_results(reason)
        
        return self
    
    def display_results(self, reason: str):
        """Display comprehensive results."""
        self._print("\n" + "="*70)
        self._print("📊 RESULTS SUMMARY")
        self._print("="*70)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.drop('model_object', axis=1)
        
        if self.problem_type == 'classification':
            results_df = results_df.sort_values('CV Score', ascending=False)
        elif self.problem_type == 'regression':
            results_df = results_df.sort_values('CV Score', ascending=False)
        else:
            results_df = results_df.sort_values('Silhouette Score', ascending=False)
        
        print("\n" + results_df.to_string(index=False))
        
        # Best model recommendation
        self._print("\n" + "="*70)
        self._print("🏆 BEST MODEL RECOMMENDATION")
        self._print("="*70)
        self._print(f"\n✨ Selected Model: {self.best_model_name}")
        self._print(f"\n📝 Reason: {reason}")
        
        # Advice
        self._print("\n💡 Recommendation:")
        if self.problem_type == 'classification':
            self._print("  - Use this model for making predictions on new data")
            self._print("  - Consider tuning hyperparameters for better performance")
            self._print("  - Check confusion matrix to understand error patterns")
        elif self.problem_type == 'regression':
            self._print("  - Use this model for predicting continuous values")
            self._print("  - Consider feature engineering to improve R² score")
            self._print("  - Check residual plots to validate assumptions")
        else:
            self._print("  - Use these clusters for customer segmentation or pattern discovery")
            self._print("  - Analyze cluster centers to understand group characteristics")
            self._print("  - Consider different K values for K-Means if needed")
    
    def predict(self, X):
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_best_model(self):
        """Return the best model."""
        return self.best_model


def main():
    """Example usage of AutoModelSelector."""
    print("="*70)
    print("  INTELLIGENT AUTO-MODEL SELECTOR")
    print("="*70)
    print("\nThis system automatically:")
    print("  1. Detects your problem type (classification/regression/clustering)")
    print("  2. Analyzes dataset characteristics")
    print("  3. Selects appropriate models")
    print("  4. Trains and evaluates all models")
    print("  5. Recommends the best model for your data")
    print("\n" + "="*70)
    
    # Example 1: Classification
    print("\n\n" + "="*70)
    print("EXAMPLE 1: CLASSIFICATION DATASET")
    print("="*70)
    
    from sklearn.datasets import make_classification
    X_class, y_class = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_classes=3, random_state=42
    )
    
    selector = AutoModelSelector(verbose=True)
    selector.fit(X_class, y_class)
    
    # Make predictions
    print("\n📍 Making predictions on new data...")
    predictions = selector.predict(X_class[:5])
    print(f"Predictions: {predictions}")
    
    # Example 2: Regression
    print("\n\n" + "="*70)
    print("EXAMPLE 2: REGRESSION DATASET")
    print("="*70)
    
    from sklearn.datasets import make_regression
    X_reg, y_reg = make_regression(
        n_samples=300, n_features=5, noise=10, random_state=42
    )
    
    selector2 = AutoModelSelector(verbose=True)
    selector2.fit(X_reg, y_reg)
    
    # Example 3: Clustering
    print("\n\n" + "="*70)
    print("EXAMPLE 3: CLUSTERING DATASET")
    print("="*70)
    
    from sklearn.datasets import make_blobs
    X_cluster, _ = make_blobs(
        n_samples=300, n_features=4, centers=4, random_state=42
    )
    
    selector3 = AutoModelSelector(verbose=True)
    selector3.fit(X_cluster)  # No y provided
    
    print("\n\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nNow try with your own dataset!")
    print("Example:")
    print("  selector = AutoModelSelector()")
    print("  selector.fit(your_X, your_y)")
    print("  predictions = selector.predict(new_data)")


if __name__ == "__main__":
    main()
