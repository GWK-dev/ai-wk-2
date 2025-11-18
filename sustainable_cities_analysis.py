# Sustainable Cities Analysis - SDG 11
# Jupyter Notebook for AI-powered Urban Sustainability

# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
import plotly.express as px
import plotly.graph_objects as go

print("🏙️ AI for Sustainable Cities - SDG 11 Analysis")
print("Libraries imported successfully!")

# Cell 2: Generate sample data
def generate_urban_data(n_cities=200):
    np.random.seed(42)
    
    data = {
        'city': [f'City_{i}' for i in range(n_cities)],
        'population_density': np.random.normal(2000, 800, n_cities),
        'public_transport_usage': np.random.normal(35, 15, n_cities),
        'green_spaces_percent': np.random.normal(25, 10, n_cities),
        'renewable_energy_usage': np.random.normal(20, 12, n_cities),
        'air_quality_index': np.random.normal(65, 20, n_cities),
        'waste_recycling_rate': np.random.normal(45, 18, n_cities),
        'gdp_per_capita': np.random.normal(30000, 15000, n_cities)
    }
    
    # Calculate sustainability score
    df = pd.DataFrame(data)
    df['sustainability_score'] = (
        df['public_transport_usage'] * 0.2 +
        df['green_spaces_percent'] * 0.2 +
        df['renewable_energy_usage'] * 0.25 +
        (100 - df['air_quality_index']) * 0.15 +
        df['waste_recycling_rate'] * 0.2
    )
    
    return df

urban_df = generate_urban_data()
print(f"Generated data for {len(urban_df)} cities")
urban_df.head()

# Cell 3: Exploratory Data Analysis
print("📊 Dataset Overview:")
print(f"Shape: {urban_df.shape}")
print(f"Columns: {urban_df.columns.tolist()}")
print("\nBasic Statistics:")
urban_df.describe()

# Cell 4: Sustainability Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(urban_df['sustainability_score'], bins=20, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of City Sustainability Scores', fontsize=14, fontweight='bold')
plt.xlabel('Sustainability Score')
plt.ylabel('Number of Cities')
plt.grid(alpha=0.3)
plt.show()

# Cell 5: Correlation Analysis
corr_matrix = urban_df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Urban Sustainability Factors', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Cell 6: Train ML Model for Sustainability Prediction
feature_columns = [
    'population_density', 'public_transport_usage', 'green_spaces_percent',
    'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
    'gdp_per_capita'
]

X = urban_df[feature_columns]
y = urban_df['sustainability_score']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)

# Evaluate model
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("🤖 Machine Learning Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Cell 7: Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance in Sustainability Prediction', 
          fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Cell 8: City Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
urban_df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(urban_df['gdp_per_capita'], urban_df['sustainability_score'],
                     c=urban_df['cluster'], cmap='viridis', alpha=0.7, s=60)
plt.colorbar(scatter, label='Cluster')
plt.title('City Clusters: GDP vs Sustainability', fontsize=14, fontweight='bold')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Sustainability Score')
plt.grid(alpha=0.3)
plt.show()

# Cell 9: Cluster Analysis
cluster_summary = urban_df.groupby('cluster').agg({
    'sustainability_score': 'mean',
    'public_transport_usage': 'mean',
    'renewable_energy_usage': 'mean',
    'gdp_per_capita': 'mean',
    'city': 'count'
}).rename(columns={'city': 'city_count'})

print("🏷️ Cluster Characteristics:")
cluster_summary

# Cell 10: Interactive Visualization with Plotly
fig = px.scatter(urban_df, x='gdp_per_capita', y='sustainability_score',
                color='cluster', hover_data=['city'],
                title='Interactive City Sustainability Analysis',
                labels={'gdp_per_capita': 'GDP per Capita (USD)',
                       'sustainability_score': 'Sustainability Score'})
fig.show()

# Cell 11: Sustainability Recommendations System
def get_sustainability_recommendations(city_data, target_score=70):
    current_score = model.predict(scaler.transform([city_data]))[0]
    
    if current_score >= target_score:
        return "City already meets sustainability target!"
    
    recommendations = []
    gap = target_score - current_score
    
    # Simple recommendation logic
    if city_data[1] < 30:  # public_transport_usage
        recommendations.append("🚍 Improve public transportation infrastructure")
    
    if city_data[2] < 20:  # green_spaces_percent
        recommendations.append("🌳 Increase urban green spaces and parks")
    
    if city_data[3] < 15:  # renewable_energy_usage
        recommendations.append("☀️ Invest in renewable energy sources")
    
    return {
        'current_score': current_score,
        'target_score': target_score,
        'score_gap': gap,
        'recommendations': recommendations
    }

# Test with a sample city
sample_city = urban_df.iloc[0]
sample_features = [sample_city[col] for col in feature_columns]
recommendations = get_sustainability_recommendations(sample_features)

print("💡 AI-Powered Recommendations:")
print(f"Current Score: {recommendations['current_score']:.1f}")
print(f"Target Score: {recommendations['target_score']}")
print(f"Score Gap: {recommendations['score_gap']:.1f}")
print("\nRecommended Actions:")
for rec in recommendations['recommendations']:
    print(f"- {rec}")

# Cell 12: Ethical Considerations
print("⚖️ Ethical Considerations for AI in Urban Sustainability:")
ethical_points = [
    "1. Ensure equitable distribution of sustainability benefits across all neighborhoods",
    "2. Consider economic accessibility of recommended improvements",
    "3. Avoid bias in data collection from different city regions",
    "4. Involve community stakeholders in sustainability planning",
    "5. Monitor unintended consequences of urban changes"
]

for point in ethical_points:
    print(f"   {point}")

print("\n✅ Analysis completed! This AI system supports SDG 11:")
print("   - Sustainable cities and communities")
print("   - Data-driven urban planning")
print("   - Environmental impact reduction")