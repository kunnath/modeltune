"""
Berlin Traffic & Construction Sites Analysis 2023
==================================================
Analyzes construction site data (dtvw2023baustellen.xml) and predicts traffic patterns.

Dataset: Berlin Traffic Construction Sites 2023 (DTV = Average Daily Traffic)
Source: GDI Berlin - Verkehrsmengen (Traffic Volumes)
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter
from auto_model_selector import AutoModelSelector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

print("="*80)
print("  BERLIN TRAFFIC & CONSTRUCTION SITES ANALYSIS 2023")
print("="*80)

# Parse XML file
print("\n📂 Loading construction site data...")
tree = ET.parse('dtvw2023baustellen.xml')
root = tree.getroot()

# Define namespaces
ns = {
    'wfs': 'http://www.opengis.net/wfs/2.0',
    'gml': 'http://www.opengis.net/gml/3.2',
    'verkehr': 'verkehrsmengen_2023'
}

# Extract data
data = []
for member in root.findall('.//wfs:member', ns):
    baustelle = member.find('.//verkehr:dtvw2023baustellen', ns)
    if baustelle is not None:
        # Extract construction site information
        site_id = baustelle.find('.//verkehr:id', ns)
        strasse = baustelle.find('.//verkehr:strasse', ns)
        ortsteil = baustelle.find('.//verkehr:ortsteil', ns)
        
        # Extract coordinates
        pos = baustelle.find('.//gml:pos', ns)
        
        if all([site_id is not None, strasse is not None, pos is not None]):
            coords = pos.text.split()
            
            # Extract district from street name if available
            street_name = strasse.text
            district = ortsteil.text if ortsteil is not None else 'Unknown'
            
            # Extract street type
            street_type = 'Straße'  # default
            if 'Brücke' in street_name:
                street_type = 'Brücke'
            elif 'Platz' in street_name:
                street_type = 'Platz'
            elif 'Allee' in street_name:
                street_type = 'Allee'
            elif 'Damm' in street_name:
                street_type = 'Damm'
            elif 'Weg' in street_name:
                street_type = 'Weg'
            
            data.append({
                'site_id': int(site_id.text),
                'street_name': street_name,
                'district': district,
                'street_type': street_type,
                'x_coord': float(coords[0]),
                'y_coord': float(coords[1])
            })

# Create DataFrame
df = pd.DataFrame(data)

print(f"✅ Loaded {len(df)} construction sites")
print(f"\n📊 Dataset Preview:")
print(df.head(10))

print(f"\n📈 Dataset Statistics:")
print(df.describe())

print(f"\n🔍 Dataset Info:")
print(f"  Total Construction Sites: {len(df)}")
print(f"  Number of Districts: {df['district'].nunique()}")
print(f"  Number of Unique Streets: {df['street_name'].nunique()}")

# District analysis
print(f"\n📊 Construction Sites by District (Top 10):")
district_counts = df['district'].value_counts().head(10)
print(district_counts)

# Street type analysis
print(f"\n🛣️  Construction Sites by Street Type:")
print(df['street_type'].value_counts())

# Save to CSV
df.to_csv('traffic_construction_sites.csv', index=False)
print(f"\n✅ Data saved to 'traffic_construction_sites.csv'")

# =============================================================================
# FEATURE ENGINEERING FOR TRAFFIC PREDICTION
# =============================================================================

print("\n" + "="*80)
print("  FEATURE ENGINEERING FOR TRAFFIC PREDICTION")
print("="*80)

# Calculate distance from city center (Alexanderplatz approximate: 392000, 5820000)
city_center_x, city_center_y = 392000, 5820000
df['distance_from_center'] = np.sqrt(
    (df['x_coord'] - city_center_x)**2 + 
    (df['y_coord'] - city_center_y)**2
)

# Categorize by distance
def categorize_distance(dist):
    if dist < 2000:
        return 'City Center'
    elif dist < 5000:
        return 'Inner City'
    elif dist < 10000:
        return 'Outer City'
    else:
        return 'Suburbs'

df['location_type'] = df['distance_from_center'].apply(categorize_distance)

print("\n📍 Construction Sites by Location Type:")
print(df['location_type'].value_counts())

# Estimate traffic impact based on street type and location
def estimate_traffic_level(row):
    """Estimate traffic level based on street type, location, and district"""
    base_traffic = {
        'Brücke': 5,    # Bridges typically have high traffic
        'Platz': 4,     # Squares/Plazas
        'Allee': 4,     # Avenues
        'Damm': 4,      # Major roads
        'Straße': 3,    # Regular streets
        'Weg': 2        # Minor roads
    }
    
    location_multiplier = {
        'City Center': 1.5,
        'Inner City': 1.3,
        'Outer City': 1.1,
        'Suburbs': 0.9
    }
    
    # Major districts with higher traffic
    major_districts = ['Mitte', 'Charlottenburg', 'Kreuzberg', 'Prenzlauer Berg']
    district_bonus = 1 if row['district'] in major_districts else 0
    
    traffic = base_traffic.get(row['street_type'], 3) * location_multiplier.get(row['location_type'], 1.0)
    traffic += district_bonus
    
    return min(int(round(traffic)), 10)  # Cap at 10

df['estimated_traffic_level'] = df.apply(estimate_traffic_level, axis=1)

print("\n🚦 Estimated Traffic Level Distribution:")
print(df['estimated_traffic_level'].value_counts().sort_index())

# Categorize traffic impact (adjusted thresholds for balanced distribution)
def categorize_traffic_impact(level):
    if level >= 5:
        return 'High'
    elif level >= 4:
        return 'Medium'
    else:
        return 'Low'

df['traffic_impact'] = df['estimated_traffic_level'].apply(categorize_traffic_impact)

print("\n⚠️  Traffic Impact Distribution:")
print(df['traffic_impact'].value_counts())

# =============================================================================
# POSSIBLE PREDICTIONS FROM THIS DATA
# =============================================================================

print("\n" + "="*80)
print("  POSSIBLE TRAFFIC PREDICTIONS")
print("="*80)

print("""
Based on the construction site data, here are the traffic predictions we can make:

1. 🚦 TRAFFIC IMPACT CLASSIFICATION
   Predict: High/Medium/Low traffic impact at construction sites
   Based on: Location, street type, district
   Use Case: Traffic management, route planning
   
2. 🗺️ CONSTRUCTION ZONE CLUSTERING
   Predict: High-density construction zones in Berlin
   Based on: Geographic coordinates
   Use Case: City planning, resource allocation
   
3. 📍 DISTRICT TRAFFIC PREDICTION
   Predict: Which districts have highest construction impact
   Based on: District, location type, street characteristics
   Use Case: Infrastructure planning, budget allocation
   
4. 🛣️  STREET TYPE TRAFFIC PREDICTION
   Predict: Expected traffic level for different street types
   Based on: Street type + location + district
   Use Case: Construction scheduling, detour planning
   
5. 🎯 HIGH-IMPACT SITE IDENTIFICATION
   Predict: Which construction sites cause most disruption
   Based on: Multiple features (location, type, district)
   Use Case: Priority management, communication planning
""")

# =============================================================================
# PREDICTION 1: TRAFFIC IMPACT CLASSIFICATION
# =============================================================================

print("\n" + "="*80)
print("  PREDICTION 1: TRAFFIC IMPACT CLASSIFICATION")
print("="*80)

# Encode categorical features
le_district = LabelEncoder()
le_street_type = LabelEncoder()
le_location_type = LabelEncoder()

df['district_encoded'] = le_district.fit_transform(df['district'])
df['street_type_encoded'] = le_street_type.fit_transform(df['street_type'])
df['location_type_encoded'] = le_location_type.fit_transform(df['location_type'])

# Prepare features for classification
X_class = df[['x_coord', 'y_coord', 'distance_from_center', 
              'district_encoded', 'street_type_encoded', 'location_type_encoded']].values

le_impact = LabelEncoder()
y_class = le_impact.fit_transform(df['traffic_impact'])

print("\n🤖 Training classifier to predict traffic impact...")
print("Features: Location, distance from center, district, street type")
print("Target: Traffic impact (High/Medium/Low)")

selector1 = AutoModelSelector(verbose=False)
selector1.fit(X_class, y_class)

print(f"\n✅ Best Model: {selector1.best_model_name}")

# Display results
results_df = pd.DataFrame(selector1.results).drop('model_object', axis=1)
results_df = results_df.sort_values('CV Score', ascending=False)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# =============================================================================
# PREDICTION 2: CONSTRUCTION ZONE CLUSTERING
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 2: CONSTRUCTION ZONE CLUSTERING")
print("="*80)

X_cluster = df[['x_coord', 'y_coord']].values

print("\n🤖 Clustering construction sites into geographic zones...")
selector2 = AutoModelSelector(verbose=False)
selector2.fit(X_cluster)

print(f"\n✅ Best Model: {selector2.best_model_name}")

# Get cluster assignments
clusters = selector2.predict(X_cluster)
df['construction_zone'] = clusters

print(f"\n📊 Identified {len(np.unique(clusters))} construction zones:")
zone_stats = df.groupby('construction_zone').agg({
    'site_id': 'count',
    'estimated_traffic_level': 'mean',
    'distance_from_center': 'mean',
    'x_coord': 'mean',
    'y_coord': 'mean'
}).round(2)
zone_stats.columns = ['Sites', 'Avg Traffic', 'Avg Dist Center', 'Center X', 'Center Y']
print(zone_stats)

# =============================================================================
# PREDICTION 3: DISTRICT TRAFFIC ANALYSIS
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 3: DISTRICT TRAFFIC ANALYSIS")
print("="*80)

# Aggregate by district
district_analysis = df.groupby('district').agg({
    'site_id': 'count',
    'estimated_traffic_level': 'mean',
    'distance_from_center': 'mean'
}).round(2)
district_analysis.columns = ['Construction Sites', 'Avg Traffic Level', 'Dist from Center']
district_analysis = district_analysis.sort_values('Construction Sites', ascending=False)

print("\n🏙️  Top 10 Districts by Construction Activity:")
print(district_analysis.head(10))

# Identify high-impact districts
high_impact_districts = df[df['traffic_impact'] == 'High']['district'].value_counts()
print(f"\n⚠️  Districts with Most High-Impact Sites:")
print(high_impact_districts.head(10))

# =============================================================================
# PREDICTION 4: TRAFFIC LEVEL REGRESSION
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 4: TRAFFIC LEVEL PREDICTION")
print("="*80)

# Predict traffic level based on features
X_reg = df[['x_coord', 'y_coord', 'distance_from_center', 
            'district_encoded', 'street_type_encoded', 'location_type_encoded']].values
y_reg = df['estimated_traffic_level'].values.astype(float)

print("\n🤖 Training model to predict traffic level (1-10 scale)...")
selector3 = AutoModelSelector(verbose=False)
selector3.fit(X_reg, y_reg)

print(f"\n✅ Best Model: {selector3.best_model_name}")

# Display results
results_reg = pd.DataFrame(selector3.results).drop('model_object', axis=1)
results_reg = results_reg.sort_values('R² Score', ascending=False)
print("\nModel Comparison:")
print(results_reg.to_string(index=False))

# Test predictions for hypothetical sites
print("\n📍 Predicting traffic levels for hypothetical construction sites:")

# Define test locations (using street types that exist in the data)
test_locations = [
    {'name': 'Central Berlin (Alexanderplatz area)', 'x': 392000, 'y': 5820000, 
     'district': 'Mitte', 'street_type': 'Straße', 'location_type': 'City Center'},
    {'name': 'West Berlin (Charlottenburg)', 'x': 385000, 'y': 5817000,
     'district': 'Charlottenburg', 'street_type': 'Straße', 'location_type': 'Inner City'},
    {'name': 'East Berlin (Lichtenberg)', 'x': 398000, 'y': 5819000,
     'district': 'Lichtenberg', 'street_type': 'Allee', 'location_type': 'Outer City'},
]

for loc in test_locations:
    dist = np.sqrt((loc['x'] - city_center_x)**2 + (loc['y'] - city_center_y)**2)
    
    # Encode features
    dist_enc = le_district.transform([loc['district']])[0] if loc['district'] in le_district.classes_ else 0
    street_enc = le_street_type.transform([loc['street_type']])[0]
    loc_enc = le_location_type.transform([loc['location_type']])[0]
    
    test_X = np.array([[loc['x'], loc['y'], dist, dist_enc, street_enc, loc_enc]])
    prediction = selector3.predict(test_X)[0]
    
    print(f"\n{loc['name']}:")
    print(f"  Predicted Traffic Level: {prediction:.1f}/10")
    print(f"  Impact: {'High' if prediction >= 7 else 'Medium' if prediction >= 4 else 'Low'}")

# =============================================================================
# HIGH-IMPACT SITES IDENTIFICATION
# =============================================================================

print("\n\n" + "="*80)
print("  PREDICTION 5: HIGH-IMPACT SITE IDENTIFICATION")
print("="*80)

# Identify high-impact sites
high_impact = df[df['traffic_impact'] == 'High'].sort_values('estimated_traffic_level', ascending=False)

print(f"\n⚠️  Identified {len(high_impact)} High-Impact Construction Sites:")
print("\nTop 10 High-Impact Sites:")
print(high_impact[['street_name', 'district', 'street_type', 
                   'location_type', 'estimated_traffic_level']].head(10).to_string(index=False))

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n\n" + "="*80)
print("  CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Geographic distribution with traffic impact
impact_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
for impact in ['Low', 'Medium', 'High']:
    mask = df['traffic_impact'] == impact
    axes[0, 0].scatter(
        df[mask]['x_coord'], df[mask]['y_coord'],
        c=impact_colors[impact],
        s=100,
        alpha=0.6,
        label=impact,
        edgecolors='black'
    )
axes[0, 0].set_xlabel('X Coordinate (meters)', fontsize=11)
axes[0, 0].set_ylabel('Y Coordinate (meters)', fontsize=11)
axes[0, 0].set_title('Berlin Construction Sites - Traffic Impact 2023', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Construction zones
scatter2 = axes[0, 1].scatter(
    df['x_coord'], df['y_coord'],
    c=df['construction_zone'],
    s=100,
    alpha=0.6,
    cmap='tab10',
    edgecolors='black'
)
axes[0, 1].set_xlabel('X Coordinate (meters)', fontsize=11)
axes[0, 1].set_ylabel('Y Coordinate (meters)', fontsize=11)
axes[0, 1].set_title('Construction Zones (Clusters)', fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('Zone', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Top districts
district_counts_top = df['district'].value_counts().head(10)
axes[1, 0].barh(range(len(district_counts_top)), district_counts_top.values, color='skyblue', edgecolor='black')
axes[1, 0].set_yticks(range(len(district_counts_top)))
axes[1, 0].set_yticklabels(district_counts_top.index)
axes[1, 0].set_xlabel('Number of Construction Sites', fontsize=11)
axes[1, 0].set_title('Top 10 Districts by Construction Sites', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Traffic impact distribution
impact_dist = df['traffic_impact'].value_counts()
colors_impact = ['#4ECDC4', '#FFA500', '#FF6B6B']
axes[1, 1].bar(impact_dist.index, impact_dist.values, 
               color=colors_impact, edgecolor='black')
axes[1, 1].set_xlabel('Traffic Impact Level', fontsize=11)
axes[1, 1].set_ylabel('Number of Sites', fontsize=11)
axes[1, 1].set_title('Traffic Impact Distribution', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('traffic_construction_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'traffic_construction_analysis.png'")

# =============================================================================
# SAVE ENRICHED DATA
# =============================================================================

# Save enriched dataset with predictions
df_enriched = df.copy()
df_enriched['predicted_impact'] = le_impact.inverse_transform(selector1.predict(X_class))
df_enriched.to_csv('traffic_construction_enriched.csv', index=False)
print("✅ Enriched data saved to 'traffic_construction_enriched.csv'")

# =============================================================================
# SUMMARY & INSIGHTS
# =============================================================================

print("\n\n" + "="*80)
print("  SUMMARY & INSIGHTS")
print("="*80)

print(f"""
📊 DATASET SUMMARY:
  • Total Construction Sites: {len(df)}
  • Geographic Coverage: Berlin area
  • Traffic Level Range: 1-10 scale
  • Average Traffic Level: {df['estimated_traffic_level'].mean():.2f}
  • Districts Covered: {df['district'].nunique()}

🚦 TRAFFIC PREDICTIONS COMPLETED:

1. ✅ Traffic Impact Classification
   - Model: {selector1.best_model_name}
   - Accuracy: {max([r['Accuracy'] for r in selector1.results]):.2%}
   - Categories: High ({(df['traffic_impact']=='High').sum()}), Medium ({(df['traffic_impact']=='Medium').sum()}), Low ({(df['traffic_impact']=='Low').sum()})

2. ✅ Construction Zone Clustering
   - Model: {selector2.best_model_name}
   - Zones Identified: {len(np.unique(clusters))}
   - Use: Coordinated construction management

3. ✅ District Traffic Analysis
   - Top District: {district_analysis.index[0]} ({district_analysis.iloc[0]['Construction Sites']:.0f} sites)
   - High-Impact Districts: {len(high_impact_districts)} districts

4. ✅ Traffic Level Prediction
   - Model: {selector3.best_model_name}
   - R² Score: {max([r['R² Score'] for r in selector3.results]):.4f}
   - Use: Planning new construction projects

💡 KEY INSIGHTS:
  • {(df['traffic_impact'] == 'High').sum()} high-impact construction sites identified
  • {district_analysis.index[0]} has most construction activity ({district_analysis.iloc[0]['Construction Sites']:.0f} sites)
  • {'Bridges' if df[df['street_type']=='Brücke']['estimated_traffic_level'].mean() > df['estimated_traffic_level'].mean() else 'Regular streets'} show higher average traffic impact
  • City center sites have {((df[df['location_type']=='City Center']['estimated_traffic_level'].mean() / df['estimated_traffic_level'].mean() - 1) * 100):.0f}% higher traffic levels
  • {len(np.unique(clusters))} distinct construction zones identified

🎯 PRACTICAL APPLICATIONS:
  1. Traffic Management: Identify high-impact sites for priority attention
  2. Route Planning: Use clusters to plan detours
  3. Construction Scheduling: Coordinate work in same zones
  4. Public Communication: Alert residents in high-impact areas
  5. Resource Allocation: Focus on {district_analysis.index[0]} and other busy districts
""")

print("\n" + "="*80)
print("  ANALYSIS COMPLETE!")
print("="*80)

print("""
📁 FILES CREATED:
  • traffic_construction_sites.csv       - Original data in CSV format
  • traffic_construction_enriched.csv    - Data with predictions
  • traffic_construction_analysis.png    - Visualization

🚀 NEXT STEPS:
  1. Review the visualizations
  2. Check enriched CSV for predictions
  3. Use models to predict impact of new construction sites
  4. Apply insights to traffic management planning
""")
