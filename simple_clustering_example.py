"""
Simple Clustering Example
==========================
Segment customers into groups based on their shopping behavior.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate sample customer data
n_customers = 200

# Create three natural customer segments
segment1 = np.random.normal([25, 30], [3, 5], [60, 2])  # Young, low spending
segment2 = np.random.normal([45, 70], [5, 8], [70, 2])  # Middle-aged, high spending
segment3 = np.random.normal([60, 50], [4, 6], [70, 2])  # Older, medium spending

# Combine all segments
data = np.vstack([segment1, segment2, segment3])

# Create DataFrame
df = pd.DataFrame(data, columns=['Age', 'Annual_Spending_k$'])

print("=" * 50)
print("CUSTOMER SEGMENTATION")
print("=" * 50)
print("\nDataset Overview:")
print(df.head(10))
print(f"\nTotal Customers: {len(df)}")
print("\nDataset Statistics:")
print(df.describe())

# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal number of clusters using Elbow Method
print("\n" + "=" * 50)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 50)

inertias = []
K_range = range(1, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia = {kmeans.inertia_:.2f}")

# Apply K-Means with optimal k=3
print("\n" + "=" * 50)
print("CLUSTERING CUSTOMERS (K=3)")
print("=" * 50)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("✅ Clustering completed!")

# Analyze clusters
print("\n" + "=" * 50)
print("CLUSTER ANALYSIS")
print("=" * 50)

for cluster_id in range(3):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} customers):")
    print(f"  Average Age: {cluster_data['Age'].mean():.1f} years")
    print(f"  Average Spending: ${cluster_data['Annual_Spending_k$'].mean():.1f}k")
    print(f"  Characteristics: ", end='')
    
    avg_age = cluster_data['Age'].mean()
    avg_spending = cluster_data['Annual_Spending_k$'].mean()
    
    if avg_age < 35:
        age_desc = "Young"
    elif avg_age < 50:
        age_desc = "Middle-aged"
    else:
        age_desc = "Mature"
    
    if avg_spending < 40:
        spend_desc = "Budget Shoppers"
    elif avg_spending < 60:
        spend_desc = "Moderate Spenders"
    else:
        spend_desc = "Premium Customers"
    
    print(f"{age_desc}, {spend_desc}")

# Get cluster centers in original scale
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

print("\n" + "=" * 50)
print("MARKETING RECOMMENDATIONS")
print("=" * 50)

recommendations = {
    0: "Target with trendy, affordable products",
    1: "Focus on premium, quality offerings",
    2: "Emphasize reliability and value for money"
}

for i, (age, spending) in enumerate(centers_original):
    print(f"\nCluster {i}:")
    print(f"  Center: Age {age:.1f}, Spending ${spending:.1f}k")
    print(f"  Strategy: {recommendations.get(i, 'Customize approach')}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0].set_title('Elbow Method for Optimal K')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Optimal K=3')
axes[0].legend()

# Plot 2: Customer segments
colors = ['red', 'green', 'blue']
for cluster_id in range(3):
    cluster_data = df[df['Cluster'] == cluster_id]
    axes[1].scatter(cluster_data['Age'], 
                   cluster_data['Annual_Spending_k$'],
                   c=colors[cluster_id],
                   label=f'Cluster {cluster_id}',
                   alpha=0.6,
                   edgecolors='black',
                   s=80)

# Plot cluster centers
axes[1].scatter(centers_original[:, 0], 
               centers_original[:, 1],
               c='yellow',
               marker='*',
               s=500,
               edgecolors='black',
               linewidth=2,
               label='Centroids',
               zorder=10)

axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Annual Spending ($1000s)')
axes[1].set_title('Customer Segments')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'clustering_results.png'")

# Save clustered data to CSV
df.to_csv('customer_segments.csv', index=False)
print("✅ Segmented data saved to 'customer_segments.csv'")

print("\n" + "=" * 50)
print("DONE! 🎉")
print("=" * 50)
print("\nYou can now target each customer segment with tailored marketing!")
