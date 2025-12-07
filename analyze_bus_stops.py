"""
XML Bus Stop Data Analyzer & Predictor
========================================
Analyzes Berlin bus stop data and creates predictive models.

Dataset: Berlin Public Transport Bus Stops (a_busstopp.xml)
Source: GDI Berlin (Geodata Infrastructure)
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from auto_model_selector import AutoModelSelector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("="*80)
print("  XML BUS STOP DATA ANALYZER & PREDICTOR")
print("="*80)

# Parse XML file
print("\n📂 Loading XML data...")
tree = ET.parse('data_busstopp.xml')
root = tree.getroot()

# Define namespaces
ns = {
    'wfs': 'http://www.opengis.net/wfs/2.0',
    'gml': 'http://www.opengis.net/gml/3.2',
    'oepnv': 'oepnv_ungestoert'
}

# Extract data
data = []
for member in root.findall('.//wfs:member', ns):
    busstopp = member.find('.//oepnv:a_busstopp', ns)
    if busstopp is not None:
        # Extract bus stop information
        no = busstopp.find('.//oepnv:no', ns)
        hstname = busstopp.find('.//oepnv:hstname', ns)
        anzlinien = busstopp.find('.//oepnv:anzlinien', ns)
        code = busstopp.find('.//oepnv:code', ns)
        
        # Extract coordinates
        pos = busstopp.find('.//gml:pos', ns)
        
        if all([no is not None, hstname is not None, anzlinien is not None, pos is not None]):
            coords = pos.text.split()
            
            data.append({
                'stop_id': int(no.text),
                'stop_name': hstname.text,
                'num_lines': int(anzlinien.text),
                'stop_code': code.text if code is not None else 'N/A',
                'x_coord': float(coords[0]),
                'y_coord': float(coords[1])
            })

# Create DataFrame
df = pd.DataFrame(data)

print(f"✅ Loaded {len(df)} bus stops")
print(f"\n📊 Dataset Preview:")
print(df.head(10))

print(f"\n📈 Dataset Statistics:")
print(df.describe())

print(f"\n🔍 Dataset Info:")
print(f"  Total Bus Stops: {len(df)}")
print(f"  Number of Lines Range: {df['num_lines'].min()} to {df['num_lines'].max()}")
print(f"  Average Lines per Stop: {df['num_lines'].mean():.2f}")
print(f"  Unique Stop Names: {df['stop_name'].nunique()}")

# Distribution analysis
print(f"\n📊 Lines per Stop Distribution:")
print(df['num_lines'].value_counts().sort_index())

# Save to CSV for easy viewing
df.to_csv('bus_stops_berlin.csv', index=False)
print(f"\n✅ Data saved to 'bus_stops_berlin.csv'")

# =============================================================================
# POSSIBLE PREDICTIONS FROM THIS DATA
# =============================================================================

print("\n" + "="*80)
print("  POSSIBLE PREDICTIONS FROM THIS DATASET")
print("="*80)

print("""
Based on the bus stop data, here are the predictions we can make:

1. 🚍 BUS STOP IMPORTANCE CLASSIFICATION
   Predict: High-traffic vs Low-traffic stops
   Based on: Number of lines serving the stop
   Use Case: Transit planning, resource allocation
   
2. 🗺️ GEOGRAPHICAL CLUSTERING
   Predict: Bus stop clusters/zones in Berlin
   Based on: Coordinates (x, y)
   Use Case: Service area definition, route optimization
   
3. 📍 STOP TYPE PREDICTION
   Predict: Hub vs Regular vs Minor stop
   Based on: Number of lines + location
   Use Case: Infrastructure investment planning
   
4. 🚌 REQUIRED LINES PREDICTION
   Predict: How many lines should serve a location
   Based on: Coordinates and nearby stops
   Use Case: New stop planning
   
5. 🎯 HUB IDENTIFICATION
   Predict: Which stops are major transfer hubs
   Based on: Lines + geographic centrality
   Use Case: Navigation and route suggestions
""")

# =============================================================================
# PREDICTION 1: BUS STOP IMPORTANCE CLASSIFICATION
# =============================================================================

print("\n" + "="*80)
print("  PREDICTION 1: BUS STOP IMPORTANCE CLASSIFICATION")
print("="*80)

# Create importance categories based on number of lines
def categorize_importance(num_lines):
    if num_lines >= 4:
        return 'High'  # Major hub
    elif num_lines >= 2:
        return 'Medium'  # Regular stop
    else:
        return 'Low'  # Minor stop

df['importance'] = df['num_lines'].apply(categorize_importance)

print("\n📊 Stop Importance Distribution:")
print(df['importance'].value_counts())

# Prepare data for classification
from sklearn.preprocessing import LabelEncoder

X_class = df[['x_coord', 'y_coord', 'num_lines']].values
le = LabelEncoder()
y_class = le.fit_transform(df['importance'])

print("\n🤖 Training classifier to predict stop importance...")
print("Features: Location (x, y) + Current lines")
print("Target: Stop importance (High/Medium/Low)")

selector1 = AutoModelSelector(verbose=False)
selector1.fit(X_class, y_class)

print(f"\n✅ Best Model: {selector1.best_model_name}")

# Display results
results_df = pd.DataFrame(selector1.results).drop('model_object', axis=1)
results_df = results_df.sort_values('CV Score', ascending=False)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# =============================================================================
# PREDICTION 2: GEOGRAPHICAL CLUSTERING
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 2: GEOGRAPHICAL CLUSTERING (BUS ZONES)")
print("="*80)

X_cluster = df[['x_coord', 'y_coord']].values

print("\n🤖 Clustering bus stops into geographical zones...")
selector2 = AutoModelSelector(verbose=False)
selector2.fit(X_cluster)

print(f"\n✅ Best Model: {selector2.best_model_name}")

# Get cluster assignments
clusters = selector2.predict(X_cluster)
df['zone'] = clusters

print(f"\n📊 Identified {len(np.unique(clusters))} geographical zones:")
zone_stats = df.groupby('zone').agg({
    'stop_id': 'count',
    'num_lines': 'mean',
    'x_coord': 'mean',
    'y_coord': 'mean'
}).round(2)
zone_stats.columns = ['Stops', 'Avg Lines', 'Center X', 'Center Y']
print(zone_stats)

# =============================================================================
# PREDICTION 3: HUB IDENTIFICATION
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 3: MAJOR HUB IDENTIFICATION")
print("="*80)

# Identify hubs (stops with 4+ lines)
hubs = df[df['num_lines'] >= 4].sort_values('num_lines', ascending=False)

print(f"\n🚍 Identified {len(hubs)} Major Transit Hubs:")
print("\nTop 10 Hubs:")
print(hubs[['stop_name', 'num_lines', 'stop_code']].head(10).to_string(index=False))

# =============================================================================
# PREDICTION 4: PREDICT LINES FOR NEW LOCATION
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 4: PREDICT LINES NEEDED FOR NEW STOP")
print("="*80)

# Train regression model to predict number of lines based on location
X_reg = df[['x_coord', 'y_coord']].values
y_reg = df['num_lines'].values.astype(float)  # Ensure regression

print("\n🤖 Training model to predict required lines for any location...")
selector3 = AutoModelSelector(verbose=False)
# Force regression by making sure y is continuous
selector3.fit(X_reg, y_reg)

print(f"\n✅ Best Model: {selector3.best_model_name}")

# Test predictions for new locations
print("\n📍 Predicting required lines for hypothetical new stops:")

# Use actual coordinate ranges from data
x_min, x_max = df['x_coord'].min(), df['x_coord'].max()
y_min, y_max = df['y_coord'].min(), df['y_coord'].max()

new_locations = [
    [(x_min + x_max) / 2, (y_min + y_max) / 2],  # Center of Berlin
    [x_min + 1000, y_min + 1000],  # Edge area
    [x_max - 1000, y_max - 1000],  # Another edge
]

predictions = selector3.predict(new_locations)

for i, (loc, pred) in enumerate(zip(new_locations, predictions)):
    print(f"\nLocation {i+1}: ({loc[0]:.0f}, {loc[1]:.0f})")
    print(f"  Predicted Lines Needed: {max(1, round(pred))}")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n\n" + "="*80)
print("  CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Geographic distribution with importance
scatter1 = axes[0, 0].scatter(
    df['x_coord'], df['y_coord'],
    c=df['num_lines'],
    s=df['num_lines'] * 20,
    alpha=0.6,
    cmap='YlOrRd',
    edgecolors='black'
)
axes[0, 0].set_xlabel('X Coordinate (meters)', fontsize=11)
axes[0, 0].set_ylabel('Y Coordinate (meters)', fontsize=11)
axes[0, 0].set_title('Berlin Bus Stops - Lines per Stop', fontsize=13, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
cbar1.set_label('Number of Lines', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Geographic zones
scatter2 = axes[0, 1].scatter(
    df['x_coord'], df['y_coord'],
    c=df['zone'],
    s=50,
    alpha=0.6,
    cmap='tab10',
    edgecolors='black'
)
axes[0, 1].set_xlabel('X Coordinate (meters)', fontsize=11)
axes[0, 1].set_ylabel('Y Coordinate (meters)', fontsize=11)
axes[0, 1].set_title('Berlin Bus Stops - Geographical Zones', fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('Zone', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of lines per stop
lines_dist = df['num_lines'].value_counts().sort_index()
axes[1, 0].bar(lines_dist.index, lines_dist.values, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Number of Lines', fontsize=11)
axes[1, 0].set_ylabel('Number of Stops', fontsize=11)
axes[1, 0].set_title('Distribution of Lines per Stop', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Stop importance
importance_dist = df['importance'].value_counts()
colors_imp = ['#FF6B6B', '#FFA500', '#4ECDC4']
axes[1, 1].bar(importance_dist.index, importance_dist.values, 
               color=colors_imp, edgecolor='black')
axes[1, 1].set_xlabel('Stop Importance', fontsize=11)
axes[1, 1].set_ylabel('Number of Stops', fontsize=11)
axes[1, 1].set_title('Stop Importance Distribution', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('bus_stop_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'bus_stop_analysis.png'")

# =============================================================================
# SAVE ENRICHED DATA
# =============================================================================

# Save enriched dataset with predictions
df_enriched = df.copy()
df_enriched['predicted_importance'] = le.inverse_transform(selector1.predict(X_class))
df_enriched.to_csv('bus_stops_enriched.csv', index=False)
print("✅ Enriched data saved to 'bus_stops_enriched.csv'")

# =============================================================================
# SUMMARY & INSIGHTS
# =============================================================================

print("\n\n" + "="*80)
print("  SUMMARY & INSIGHTS")
print("="*80)

print(f"""
📊 DATASET SUMMARY:
  • Total Bus Stops: {len(df)}
  • Geographic Coverage: Berlin area
  • Lines Range: {df['num_lines'].min()} to {df['num_lines'].max()} per stop
  • Average Lines: {df['num_lines'].mean():.2f}

🚍 PREDICTIONS COMPLETED:

1. ✅ Stop Importance Classification
   - Model: {selector1.best_model_name}
   - Accuracy: {max([r['Accuracy'] for r in selector1.results]):.2%}
   - Categories: High (≥4 lines), Medium (2-3), Low (1)

2. ✅ Geographical Clustering
   - Model: {selector2.best_model_name}
   - Zones Identified: {len(np.unique(clusters))}
   - Use: Service area definition

3. ✅ Hub Identification
   - Major Hubs Found: {len(hubs)}
   - Top Hub: {hubs.iloc[0]['stop_name']} ({hubs.iloc[0]['num_lines']} lines)

4. ✅ Lines Prediction for New Locations
   - Model: {selector3.best_model_name}
   - R² Score: {max([r['R² Score'] for r in selector3.results]):.4f}
   - Use: Planning new bus stops

💡 KEY INSIGHTS:
  • {(df['importance'] == 'High').sum()} major hubs (≥4 lines)
  • {(df['importance'] == 'Medium').sum()} regular stops (2-3 lines)
  • {(df['importance'] == 'Low').sum()} minor stops (1 line)
  • Geographic clusters can optimize route planning
  • Location strongly predicts required service level

🎯 PRACTICAL APPLICATIONS:
  1. Transit Planning: Identify areas needing more service
  2. Resource Allocation: Prioritize high-importance stops
  3. New Stop Planning: Predict service requirements
  4. Route Optimization: Use geographic clusters
  5. Infrastructure Investment: Focus on hubs
""")

print("\n" + "="*80)
print("  ANALYSIS COMPLETE!")
print("="*80)

print("""
📁 FILES CREATED:
  • bus_stops_berlin.csv           - Original data in CSV format
  • bus_stops_enriched.csv         - Data with predictions
  • bus_stop_analysis.png          - Visualization

🚀 NEXT STEPS:
  1. Review the visualizations
  2. Check enriched CSV for predictions
  3. Use the trained models for new predictions
  4. Apply insights to transit planning
""")
