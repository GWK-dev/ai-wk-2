# sdg_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class SustainableCitiesML:
    """
    Machine Learning System for Urban Sustainability Analysis (SDG 11)
    """
    
    def __init__(self):
        self.prediction_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
        
    def generate_realistic_data(self, n_cities=300):
        """
        Generate realistic urban sustainability dataset
        Simulates data from World Bank, UN Habitat, and urban databases
        """
        np.random.seed(42)
        
        # Realistic city names from different regions
        city_prefixes = ['New', 'Port', 'San', 'Santa', 'Los', 'Las', 'Great', 'Mount', 'Lake']
        city_suffixes = ['City', 'Town', 'Ville', 'Berg', 'Field', 'Wood', 'Bay', 'Springs']
        regions = ['North', 'South', 'East', 'West', 'Central', 'Coastal', 'Mountain']
        
        cities = []
        for i in range(n_cities):
            prefix = np.random.choice(city_prefixes)
            suffix = np.random.choice(city_suffixes)
            region = np.random.choice(regions)
            cities.append(f"{prefix} {suffix} {region}")
        
        data = {
            'city': cities,
            'region': np.random.choice(['Global North', 'Global South'], n_cities, p=[0.4, 0.6]),
            'population_density': np.clip(np.random.normal(2500, 1000, n_cities), 500, 6000),
            'public_transport_usage': np.clip(np.random.normal(40, 20, n_cities), 5, 85),
            'green_spaces_percent': np.clip(np.random.normal(30, 15, n_cities), 5, 70),
            'renewable_energy_usage': np.clip(np.random.normal(25, 18, n_cities), 2, 80),
            'air_quality_index': np.clip(np.random.normal(60, 25, n_cities), 20, 95),
            'waste_recycling_rate': np.clip(np.random.normal(50, 22, n_cities), 10, 90),
            'bike_friendliness_score': np.clip(np.random.normal(65, 20, n_cities), 20, 95),
            'affordable_housing_ratio': np.clip(np.random.normal(70, 18, n_cities), 30, 95),
            'gdp_per_capita': np.clip(np.random.normal(35000, 20000, n_cities), 5000, 80000),
            'co2_emissions_per_capita': np.clip(np.random.normal(8, 4, n_cities), 1, 20)
        }
        
        # Create realistic correlations
        df = pd.DataFrame(data)
        
        # Wealthier cities tend to have better sustainability infrastructure
        df['public_transport_usage'] += df['gdp_per_capita'] * 0.0001
        df['renewable_energy_usage'] += df['gdp_per_capita'] * 0.00015
        df['waste_recycling_rate'] += df['gdp_per_capita'] * 0.0001
        
        # Denser cities have different patterns
        df['public_transport_usage'] += df['population_density'] * 0.002
        df['air_quality_index'] -= df['population_density'] * 0.001  # Worse air quality
        
        # Calculate comprehensive sustainability score
        df['sustainability_score'] = self._calculate_sustainability_score(df)
        
        return df
    
    def _calculate_sustainability_score(self, df):
        """
        Calculate comprehensive sustainability score based on SDG 11 indicators
        """
        # Normalize and weight different factors
        transport_score = df['public_transport_usage'] * 0.15
        green_score = df['green_spaces_percent'] * 0.15
        energy_score = df['renewable_energy_usage'] * 0.20
        air_score = (100 - df['air_quality_index']) * 0.10  # Lower AQI is better
        waste_score = df['waste_recycling_rate'] * 0.15
        bike_score = df['bike_friendliness_score'] * 0.10
        housing_score = df['affordable_housing_ratio'] * 0.10
        emissions_score = (20 - df['co2_emissions_per_capita']) * 0.05  # Lower emissions better
        
        total_score = (transport_score + green_score + energy_score + 
                      air_score + waste_score + bike_score + 
                      housing_score + emissions_score)
        
        return np.clip(total_score, 0, 100)
    
    def train_prediction_model(self, df):
        """
        Train machine learning model to predict sustainability scores
        """
        feature_columns = [
            'population_density', 'public_transport_usage', 'green_spaces_percent',
            'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
            'bike_friendliness_score', 'affordable_housing_ratio', 'gdp_per_capita',
            'co2_emissions_per_capita'
        ]
        
        X = df[feature_columns]
        y = df['sustainability_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_score = -1
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                self.prediction_model = model
        
        # Store feature importance
        if hasattr(self.prediction_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.prediction_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate final model
        y_pred_final = self.prediction_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred_final)
        r2 = r2_score(y_test, y_pred_final)
        
        print(f"✅ Model trained successfully:")
        print(f"   - R² Score: {r2:.3f}")
        print(f"   - MAE: {mae:.2f}")
        print(f"   - Best algorithm: {type(best_model).__name__}")
        
        return mae, r2
    
    def perform_clustering(self, df, n_clusters=4):
        """
        Perform K-means clustering to identify city sustainability patterns
        """
        feature_columns = [
            'public_transport_usage', 'green_spaces_percent', 'renewable_energy_usage',
            'waste_recycling_rate', 'bike_friendliness_score', 'co2_emissions_per_capita'
        ]
        
        X = df[feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.clustering_model.fit_predict(X_scaled)
        
        # Calculate clustering quality
        silhouette_avg = silhouette_score(X_scaled, clusters)
        print(f"✅ Clustering completed:")
        print(f"   - Silhouette Score: {silhouette_avg:.3f}")
        print(f"   - Number of clusters: {n_clusters}")
        
        df['sustainability_cluster'] = clusters
        
        # Analyze cluster characteristics
        cluster_analysis = df.groupby('sustainability_cluster')[feature_columns + ['sustainability_score']].mean()
        
        return df, cluster_analysis, silhouette_avg
    
    def predict_city_sustainability(self, city_features):
        """
        Predict sustainability score for a new city
        """
        if self.prediction_model is None:
            raise ValueError("Prediction model not trained")
        
        features_scaled = self.scaler.transform([city_features])
        prediction = self.prediction_model.predict(features_scaled)[0]
        return max(0, min(100, prediction))  # Ensure within bounds
    
    def generate_improvement_recommendations(self, city_data, target_score):
        """
        Generate AI-powered recommendations to improve city sustainability
        """
        current_score = self.predict_city_sustainability(city_data)
        
        if current_score >= target_score:
            return {"status": "target_achieved", "message": "City already meets target score"}
        
        recommendations = []
        score_gap = target_score - current_score
        
        # Feature indices mapping
        feature_mapping = {
            1: ('public_transport_usage', 'Public Transportation'),
            2: ('green_spaces_percent', 'Green Spaces'),
            3: ('renewable_energy_usage', 'Renewable Energy'),
            4: ('air_quality_index', 'Air Quality'),
            5: ('waste_recycling_rate', 'Waste Management'),
            6: ('bike_friendliness_score', 'Bike Infrastructure'),
            7: ('affordable_housing_ratio', 'Affordable Housing'),
            8: ('co2_emissions_per_capita', 'Carbon Emissions')
        }
        
        # Generate recommendations based on feature importance and current values
        for idx, (feature_col, feature_name) in feature_mapping.items():
            if idx < len(city_data):
                current_value = city_data[idx]
                
                # Recommendation logic based on current performance
                if feature_col == 'public_transport_usage' and current_value < 40:
                    recommendations.append({
                        'area': feature_name,
                        'action': 'Expand public transport network and increase frequency',
                        'potential_impact': min(8, score_gap * 0.3),
                        'cost': 'Medium',
                        'timeline': '1-3 years',
                        'sdg_alignment': 'SDG 11.2: Affordable and sustainable transport systems'
                    })
                
                elif feature_col == 'green_spaces_percent' and current_value < 25:
                    recommendations.append({
                        'area': feature_name,
                        'action': 'Create urban parks and protect natural areas',
                        'potential_impact': min(6, score_gap * 0.25),
                        'cost': 'Low-Medium',
                        'timeline': '2-4 years',
                        'sdg_alignment': 'SDG 11.7: Green and public spaces'
                    })
                
                elif feature_col == 'renewable_energy_usage' and current_value < 20:
                    recommendations.append({
                        'area': feature_name,
                        'action': 'Invest in solar panels and wind energy projects',
                        'potential_impact': min(10, score_gap * 0.4),
                        'cost': 'High',
                        'timeline': '3-5 years',
                        'sdg_alignment': 'SDG 7.2: Renewable energy sources'
                    })
        
        return {
            "status": "recommendations_generated",
            "current_score": current_score,
            "target_score": target_score,
            "score_gap": score_gap,
            "recommendations": recommendations
        }
    
    def create_visualizations(self, df):
        """
        Create comprehensive visualizations for sustainability analysis
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sustainability score distribution
        axes[0, 0].hist(df['sustainability_score'], bins=20, alpha=0.7, color='#2E8B57', edgecolor='black')
        axes[0, 0].set_title('Distribution of City Sustainability Scores', fontweight='bold')
        axes[0, 0].set_xlabel('Sustainability Score')
        axes[0, 0].set_ylabel('Number of Cities')
        
        # Plot 2: Feature importance
        if self.feature_importance is not None:
            axes[0, 1].barh(self.feature_importance['feature'], self.feature_importance['importance'])
            axes[0, 1].set_title('Feature Importance in Sustainability Prediction', fontweight='bold')
            axes[0, 1].set_xlabel('Importance')
        
        # Plot 3: Clustering results
        if 'sustainability_cluster' in df.columns:
            scatter = axes[1, 0].scatter(df['gdp_per_capita'], df['sustainability_score'], 
                                       c=df['sustainability_cluster'], cmap='viridis', alpha=0.7)
            axes[1, 0].set_title('City Clusters: GDP vs Sustainability', fontweight='bold')
            axes[1, 0].set_xlabel('GDP per Capita (USD)')
            axes[1, 0].set_ylabel('Sustainability Score')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Plot 4: Regional comparison
        if 'region' in df.columns:
            region_means = df.groupby('region')['sustainability_score'].mean()
            axes[1, 1].bar(region_means.index, region_means.values, color=['#FF6B6B', '#4ECDC4'])
            axes[1, 1].set_title('Average Sustainability Score by Region', fontweight='bold')
            axes[1, 1].set_ylabel('Average Sustainability Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('sustainability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'sustainability_analysis.png'")
    
    def save_model(self, filepath='sustainable_cities_model.pkl'):
        """Save trained model to file"""
        model_data = {
            'prediction_model': self.prediction_model,
            'clustering_model': self.clustering_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='sustainable_cities_model.pkl'):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.prediction_model = model_data['prediction_model']
        self.clustering_model = model_data['clustering_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")

def main():
    """Main function to demonstrate the ML system"""
    print("🏙️ AI for Sustainable Cities - SDG 11 ML System")
    print("=" * 50)
    
    # Initialize ML system
    ml_system = SustainableCitiesML()
    
    # Generate data
    print("\n📊 Generating urban sustainability dataset...")
    df = ml_system.generate_realistic_data(300)
    print(f"   Generated data for {len(df)} cities")
    print(f"   Average sustainability score: {df['sustainability_score'].mean():.1f}")
    
    # Train prediction model
    print("\n🤖 Training sustainability prediction model...")
    mae, r2 = ml_system.train_prediction_model(df)
    
    # Perform clustering
    print("\n🔍 Performing city clustering analysis...")
    df_with_clusters, cluster_analysis, silhouette_score = ml_system.perform_clustering(df)
    
    # Generate recommendations for a sample city
    print("\n💡 Generating AI-powered recommendations...")
    sample_city = df_with_clusters.iloc[0]
    feature_columns = [
        'population_density', 'public_transport_usage', 'green_spaces_percent',
        'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
        'bike_friendliness_score', 'affordable_housing_ratio', 'gdp_per_capita',
        'co2_emissions_per_capita'
    ]
    sample_features = [sample_city[col] for col in feature_columns]
    
    recommendations = ml_system.generate_improvement_recommendations(sample_features, 75)
    
    if recommendations['status'] == 'recommendations_generated':
        print(f"   Current score: {recommendations['current_score']:.1f}")
        print(f"   Target score: {recommendations['target_score']}")
        print(f"   Score gap: {recommendations['score_gap']:.1f}")
        print(f"   Generated {len(recommendations['recommendations'])} recommendations")
    
    # Create visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    ml_system.create_visualizations(df_with_clusters)
    
    # Save model
    ml_system.save_model()
    
    print("\n✅ AI System for Sustainable Cities completed successfully!")
    print("   Ready for deployment and real-world application")

if __name__ == "__main__":
    main()