# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

# Set page configuration
st.set_page_config(
    page_title="AI for Sustainable Cities - SDG 11",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SustainableCitiesAI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.is_trained = False
        
    def generate_sample_data(self):
        """Generate realistic urban sustainability data"""
        np.random.seed(42)
        n_cities = 200
        
        data = {
            'city': [f'City_{i}' for i in range(n_cities)],
            'population_density': np.random.normal(2000, 800, n_cities),
            'public_transport_usage': np.random.normal(35, 15, n_cities),
            'green_spaces_percent': np.random.normal(25, 10, n_cities),
            'renewable_energy_usage': np.random.normal(20, 12, n_cities),
            'air_quality_index': np.random.normal(65, 20, n_cities),
            'waste_recycling_rate': np.random.normal(45, 18, n_cities),
            'bike_friendliness_score': np.random.normal(60, 15, n_cities),
            'affordable_housing_ratio': np.random.normal(65, 12, n_cities),
            'gdp_per_capita': np.random.normal(30000, 15000, n_cities)
        }
        
        # Calculate sustainability score (target variable)
        data['sustainability_score'] = (
            data['public_transport_usage'] * 0.15 +
            data['green_spaces_percent'] * 0.15 +
            data['renewable_energy_usage'] * 0.20 +
            (100 - data['air_quality_index']) * 0.10 +  # Lower AQI is better
            data['waste_recycling_rate'] * 0.15 +
            data['bike_friendliness_score'] * 0.10 +
            data['affordable_housing_ratio'] * 0.15
        )
        
        # Add some realistic correlations
        data['public_transport_usage'] += data['population_density'] * 0.002
        data['air_quality_index'] -= data['green_spaces_percent'] * 0.3
        data['sustainability_score'] += data['gdp_per_capita'] * 0.0001
        
        return pd.DataFrame(data)
    
    def train_models(self, df):
        """Train ML models for sustainability prediction and clustering"""
        # Features for prediction
        feature_columns = [
            'population_density', 'public_transport_usage', 'green_spaces_percent',
            'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
            'bike_friendliness_score', 'affordable_housing_ratio', 'gdp_per_capita'
        ]
        
        X = df[feature_columns]
        y = df['sustainability_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train prediction model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Train clustering model
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        df['sustainability_cluster'] = self.kmeans.fit_predict(X_scaled)
        
        self.is_trained = True
        return self.model, self.kmeans
    
    def predict_sustainability(self, city_features):
        """Predict sustainability score for new city data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        features_scaled = self.scaler.transform([city_features])
        prediction = self.model.predict(features_scaled)[0]
        return prediction
    
    def get_city_recommendations(self, city_data, target_score):
        """Generate AI-powered recommendations to improve sustainability"""
        recommendations = []
        current_score = self.predict_sustainability(city_data)
        
        if current_score < target_score:
            gap = target_score - current_score
            
            # AI-generated recommendations based on feature importance
            if city_data[1] < 40:  # public_transport_usage
                recommendations.append({
                    'area': 'Public Transportation',
                    'action': 'Increase public transport coverage and frequency',
                    'impact': f'Potential score increase: +{min(8, gap*0.3):.1f}',
                    'cost': 'Medium',
                    'timeline': '1-2 years'
                })
            
            if city_data[2] < 20:  # green_spaces_percent
                recommendations.append({
                    'area': 'Green Spaces',
                    'action': 'Develop urban parks and green corridors',
                    'impact': f'Potential score increase: +{min(6, gap*0.25):.1f}',
                    'cost': 'Low-Medium',
                    'timeline': '2-3 years'
                })
            
            if city_data[3] < 15:  # renewable_energy_usage
                recommendations.append({
                    'area': 'Renewable Energy',
                    'action': 'Invest in solar and wind energy infrastructure',
                    'impact': f'Potential score increase: +{min(10, gap*0.4):.1f}',
                    'cost': 'High',
                    'timeline': '3-5 years'
                })
        
        return recommendations

def main():
    st.title("🏙️ AI for Sustainable Cities - SDG 11")
    st.markdown("### Machine Learning Solutions for Urban Sustainability")
    
    # Initialize AI system
    ai_system = SustainableCitiesAI()
    
    # Generate or load data
    with st.spinner('Loading urban sustainability data...'):
        df = ai_system.generate_sample_data()
        ai_system.train_models(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Analysis", 
                                   ["Dashboard", "City Analysis", "ML Insights", 
                                    "Recommendations", "Ethical Considerations"])
    
    if app_mode == "Dashboard":
        show_dashboard(df, ai_system)
    elif app_mode == "City Analysis":
        show_city_analysis(df, ai_system)
    elif app_mode == "ML Insights":
        show_ml_insights(df, ai_system)
    elif app_mode == "Recommendations":
        show_recommendations(df, ai_system)
    elif app_mode == "Ethical Considerations":
        show_ethical_considerations()

def show_dashboard(df, ai_system):
    st.header("🌍 Sustainable Cities Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['sustainability_score'].mean()
        st.metric("Average Sustainability Score", f"{avg_score:.1f}/100")
    
    with col2:
        best_city = df.loc[df['sustainability_score'].idxmax(), 'city']
        best_score = df['sustainability_score'].max()
        st.metric("Most Sustainable City", f"{best_city} ({best_score:.1f})")
    
    with col3:
        renewable_avg = df['renewable_energy_usage'].mean()
        st.metric("Avg Renewable Energy", f"{renewable_avg:.1f}%")
    
    with col4:
        transport_avg = df['public_transport_usage'].mean()
        st.metric("Avg Public Transport", f"{transport_avg:.1f}%")
    
    # Sustainability distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sustainability Score Distribution")
        fig = px.histogram(df, x='sustainability_score', 
                          title='Distribution of City Sustainability Scores',
                          color_discrete_sequence=['#2E8B57'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("City Clusters by Sustainability")
        fig = px.scatter(df, x='gdp_per_capita', y='sustainability_score',
                        color='sustainability_cluster',
                        hover_data=['city'],
                        title='City Clusters: GDP vs Sustainability',
                        labels={'gdp_per_capita': 'GDP per Capita (USD)',
                               'sustainability_score': 'Sustainability Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Sustainability Factors Correlation")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   title='Correlation Between Sustainability Factors',
                   color_continuous_scale='RdBu_r',
                   aspect='auto')
    st.plotly_chart(fig, use_container_width=True)

def show_city_analysis(df, ai_system):
    st.header("🏙️ City Sustainability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_city = st.selectbox("Select a City", df['city'].unique()[:20])
        city_data = df[df['city'] == selected_city].iloc[0]
        
        st.subheader(f"Current Sustainability Score: {city_data['sustainability_score']:.1f}")
        
        # Display key metrics
        metrics_data = {
            'Metric': ['Public Transport Usage', 'Green Spaces', 'Renewable Energy', 
                      'Air Quality', 'Recycling Rate', 'Bike Friendliness'],
            'Value': [
                f"{city_data['public_transport_usage']:.1f}%",
                f"{city_data['green_spaces_percent']:.1f}%",
                f"{city_data['renewable_energy_usage']:.1f}%",
                f"{city_data['air_quality_index']:.1f}",
                f"{city_data['waste_recycling_rate']:.1f}%",
                f"{city_data['bike_friendliness_score']:.1f}/100"
            ]
        }
        st.table(pd.DataFrame(metrics_data))
    
    with col2:
        st.subheader("Sustainability Radar Chart")
        
        categories = ['Public Transport', 'Green Spaces', 'Renewable Energy', 
                     'Air Quality', 'Recycling', 'Bike Infrastructure']
        
        values = [
            city_data['public_transport_usage'],
            city_data['green_spaces_percent'],
            city_data['renewable_energy_usage'],
            100 - city_data['air_quality_index'],  # Invert for better visualization
            city_data['waste_recycling_rate'],
            city_data['bike_friendliness_score']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_city,
            line_color='#2E8B57'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=f"Sustainability Profile - {selected_city}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_ml_insights(df, ai_system):
    st.header("🤖 Machine Learning Insights")
    
    st.subheader("Feature Importance for Sustainability Prediction")
    
    # Get feature importance
    feature_columns = [
        'population_density', 'public_transport_usage', 'green_spaces_percent',
        'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
        'bike_friendliness_score', 'affordable_housing_ratio', 'gdp_per_capita'
    ]
    
    importance = ai_system.model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_importance_df, 
                 x='Importance', y='Feature',
                 title='Feature Importance in Sustainability Prediction',
                 color='Importance',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    st.subheader("Model Performance")
    
    X = df[feature_columns]
    X_scaled = ai_system.scaler.transform(X)
    y_pred = ai_system.model.predict(X_scaled)
    y_true = df['sustainability_score']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mae = mean_absolute_error(y_true, y_pred)
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    
    with col2:
        r2 = r2_score(y_true, y_pred)
        st.metric("R² Score", f"{r2:.3f}")
    
    with col3:
        accuracy = max(0, 1 - mae / y_true.std())  # Pseudo-accuracy
        st.metric("Prediction Accuracy", f"{accuracy*100:.1f}%")
    
    # Prediction vs Actual
    fig = px.scatter(x=y_true, y=y_pred, 
                    labels={'x': 'Actual Sustainability Score', 
                           'y': 'Predicted Sustainability Score'},
                    title='Predicted vs Actual Sustainability Scores')
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                           y=[y_true.min(), y_true.max()],
                           mode='lines',
                           name='Perfect Prediction',
                           line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations(df, ai_system):
    st.header("💡 AI-Powered Sustainability Recommendations")
    
    st.info("""
    **About this feature**: Our AI system analyzes your city's current sustainability metrics 
    and provides data-driven recommendations to improve your UN SDG 11 performance.
    """)
    
    # City selector for recommendations
    selected_city = st.selectbox("Select City for Recommendations", df['city'].unique()[:15])
    city_data = df[df['city'] == selected_city].iloc[0]
    
    # Target score input
    target_score = st.slider("Target Sustainability Score", 
                           min_value=50, max_value=95, 
                           value=75, step=5)
    
    current_score = city_data['sustainability_score']
    st.subheader(f"Current Score: {current_score:.1f} → Target: {target_score}")
    
    if current_score >= target_score:
        st.success("🎉 This city already meets or exceeds the target sustainability score!")
    else:
        # Get AI recommendations
        feature_columns = [
            'population_density', 'public_transport_usage', 'green_spaces_percent',
            'renewable_energy_usage', 'air_quality_index', 'waste_recycling_rate',
            'bike_friendliness_score', 'affordable_housing_ratio', 'gdp_per_capita'
        ]
        city_features = [city_data[col] for col in feature_columns]
        
        recommendations = ai_system.get_city_recommendations(city_features, target_score)
        
        if recommendations:
            st.subheader("Recommended Actions:")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['area']} - Impact: {rec['impact']}"):
                    st.write(f"**Action:** {rec['action']}")
                    st.write(f"**Cost:** {rec['cost']}")
                    st.write(f"**Timeline:** {rec['timeline']}")
                    
                    # Progress visualization
                    current_val = get_current_city_value(city_data, rec['area'])
                    potential_improvement = float(rec['impact'].split('+')[1].split(')')[0])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Status", f"{current_val}")
                    with col2:
                        st.metric("Potential Improvement", f"+{potential_improvement}")
        else:
            st.warning("No specific recommendations available. The city is already performing well across key metrics.")

def get_current_city_value(city_data, area):
    """Get current value for a recommendation area"""
    area_mapping = {
        'Public Transportation': f"{city_data['public_transport_usage']:.1f}%",
        'Green Spaces': f"{city_data['green_spaces_percent']:.1f}%",
        'Renewable Energy': f"{city_data['renewable_energy_usage']:.1f}%"
    }
    return area_mapping.get(area, "N/A")

def show_ethical_considerations():
    st.header("⚖️ Ethical Considerations & Social Impact")
    
    st.subheader("Potential Biases and Mitigation Strategies")
    
    ethical_points = [
        {
            "issue": "🌍 Data Representation Bias",
            "description": "Urban data may over-represent developed regions",
            "mitigation": "Include diverse city types and economic contexts",
            "impact": "Ensures global applicability of recommendations"
        },
        {
            "issue": "💸 Economic Accessibility",
            "description": "High-cost recommendations may exclude poorer cities",
            "mitigation": "Provide tiered solutions with varying cost levels",
            "impact": "Makes sustainability achievable for all economic levels"
        },
        {
            "issue": "🔍 Algorithmic Fairness",
            "description": "ML models might favor certain urban patterns",
            "mitigation": "Regular fairness audits and diverse training data",
            "impact": "Prevents systematic disadvantage to certain city types"
        },
        {
            "issue": "🌱 Environmental Justice",
            "description": "Sustainability benefits must be distributed equitably",
            "mitigation": "Include affordable housing and social equity metrics",
            "impact": "Ensures inclusive and just sustainable development"
        }
    ]
    
    for point in ethical_points:
        with st.expander(f"{point['issue']}: {point['description']}"):
            st.write(f"**Mitigation:** {point['mitigation']}")
            st.write(f"**Impact:** {point['impact']}")
    
    st.subheader("Alignment with UN SDG 11 Principles")
    
    sdg_principles = [
        "✅ **Inclusive Urbanization**: Ensuring all residents benefit from sustainable development",
        "✅ **Participatory Planning**: Involving communities in sustainability decisions", 
        "✅ **Resource Efficiency**: Optimizing use of materials, energy, and space",
        "✅ **Climate Resilience**: Building cities that can withstand environmental challenges",
        "✅ **Cultural Heritage**: Preserving cultural identity while pursuing modernization"
    ]
    
    for principle in sdg_principles:
        st.write(principle)
    
    st.subheader("Long-term Sustainability Impact")
    
    st.write("""
    This AI system contributes to SDG 11 by:
    
    - **Data-Driven Decision Making**: Helping city planners make evidence-based choices
    - **Resource Optimization**: Identifying most effective sustainability investments  
    - **Progress Tracking**: Monitoring improvements toward sustainability goals
    - **Knowledge Sharing**: Creating transferable insights across cities globally
    - **Innovation Catalyst**: Encouraging adoption of smart city technologies
    """)

if __name__ == "__main__":
    main()