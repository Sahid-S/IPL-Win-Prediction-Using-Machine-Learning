import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from openai import AzureOpenAI
from datetime import datetime, timedelta
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration from environment variables
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# Set page configuration
st.set_page_config(
    page_title="CricWin Pro - AI-Powered IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #1e3a8a;
        --secondary-blue: #3b82f6;
        --accent-orange: #f59e0b;
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --error-red: #ef4444;
        --background-light: #f8fafc;
        --background-dark: #1e293b;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-light: #e5e7eb;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Enhanced card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-orange));
    }

    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .prediction-card .probability {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    /* AI Insights styling */
    .ai-insights {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: var(--shadow);
    }
    
    .ai-insights h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Chat interface */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin: 2rem 0;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid var(--border-light);
    }
    
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .chat-message.user {
        background: var(--secondary-blue);
        color: white;
        margin-left: auto;
    }
    
    .chat-message.assistant {
        background: var(--background-light);
        color: var(--text-primary);
        border: 1px solid var(--border-light);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .status-easy { background: #dcfce7; color: #166534; }
    .status-moderate { background: #fef3c7; color: #92400e; }
    .status-difficult { background: #fed7aa; color: #c2410c; }
    .status-very-difficult { background: #fee2e2; color: #dc2626; }

    /* Loading animations */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--secondary-blue);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .prediction-card { padding: 1rem; }
        .ai-insights { padding: 1rem; }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--secondary-blue);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced data loading with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_match_data():
    """Load and process historical IPL match data"""
    try:
        matches = pd.read_csv('matches.csv', parse_dates=['date'])
        
        # Comprehensive team name normalization
        team_mappings = {
            'Delhi Daredevils': 'Delhi Capitals',
            'Deccan Chargers': 'Sunrisers Hyderabad',
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Gujarat Lions': 'Gujarat Titans',
            'Kochi Tuskers Kerala': 'Kochi Tuskers',
            'Pune Warriors India': 'Pune Warriors'
        }
        
        for col in ['team1', 'team2', 'toss_winner', 'winner']:
            if col in matches.columns:
                for old_name, new_name in team_mappings.items():
                    matches[col] = matches[col].str.replace(old_name, new_name)
        
        return matches.sort_values('date', ascending=False)
    
    except Exception as e:
        st.error(f"Error loading match data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_deliveries_data():
    """Load ball-by-ball delivery data"""
    try:
        deliveries = pd.read_csv('deliveries.csv')
        return deliveries
    except Exception as e:
        st.error(f"Error loading deliveries data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load the trained ML models"""
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            lr_data = pickle.load(f)
            lr_model = lr_data['model'] if isinstance(lr_data, dict) else lr_data
            lr_metrics = lr_data.get('metrics', {}) if isinstance(lr_data, dict) else {}
        
        with open('random_forest_model.pkl', 'rb') as f:
            rf_data = pickle.load(f)
            rf_model = rf_data['model'] if isinstance(rf_data, dict) else rf_data
            rf_metrics = rf_data.get('metrics', {}) if isinstance(rf_data, dict) else {}
        
        return lr_model, rf_model, lr_metrics, rf_metrics
    
    except FileNotFoundError:
        st.error("🚨 Model files not found! Please train the models first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Enhanced team and venue configurations
TEAMS = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
]

CITIES = [
    'Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Hyderabad',
    'Jaipur', 'Mohali', 'Ahmedabad', 'Lucknow', 'Pune', 'Indore', 
    'Dharamshala', 'Guwahati', 'Ranchi'
]

# Azure OpenAI Integration
class AIInsightsGenerator:
    def __init__(self):
        self.client = client
    
    async def generate_match_commentary(self, match_data: Dict) -> str:
        """Generate AI-powered match commentary"""
        prompt = f"""
        Generate engaging cricket commentary for the current IPL match situation:
        
        Match Details:
        - {match_data['batting_team']} vs {match_data['bowling_team']}
        - Venue: {match_data['city']}
        - Target: {match_data['target']} runs
        - Current Score: {match_data['current_score']}/{match_data['wickets_lost']}
        - Overs: {match_data['overs_completed']}/20
        - Required Run Rate: {match_data['rrr']:.2f}
        
        Provide a brief, exciting commentary (2-3 sentences) that captures the current match situation and tension.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert cricket commentator providing engaging IPL match analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate commentary at this time. Error: {str(e)}"
    
    async def generate_prediction_analysis(self, prediction_data: Dict) -> str:
        """Generate AI analysis of prediction results"""
        prompt = f"""
        Analyze the following IPL match prediction data and provide expert insights:
        
        Match: {prediction_data['batting_team']} vs {prediction_data['bowling_team']}
        Win Probability: {prediction_data['win_probability']:.1f}%
        
        Logistic Regression: {prediction_data['lr_prob']:.1f}%
        Random Forest: {prediction_data['rf_prob']:.1f}%
        
        Match Situation:
        - Runs needed: {prediction_data['runs_left']}
        - Balls left: {prediction_data['balls_left']}
        - Wickets in hand: {prediction_data['wickets']}
        - Required RR: {prediction_data['rrr']:.2f}
        
        Provide a brief analysis (2-3 sentences) explaining the prediction and key factors.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert cricket analyst providing match prediction insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate analysis at this time. Error: {str(e)}"

class IPLChatbot:
    def __init__(self, match_data: pd.DataFrame, deliveries_data: pd.DataFrame):
        self.match_data = match_data
        self.deliveries_data = deliveries_data
        self.client = client
    
    def get_context_data(self, query: str) -> str:
        """Extract relevant context from datasets based on query"""
        context = []
        
        # Add recent match results
        if self.match_data is not None:
            recent_matches = self.match_data.head(10)
            context.append(f"Recent IPL matches: {recent_matches[['date', 'team1', 'team2', 'winner']].to_string()}")
        
        # Add team performance stats
        if "team" in query.lower() and self.match_data is not None:
            team_stats = self.match_data['winner'].value_counts().head(10)
            context.append(f"Team win counts: {team_stats.to_string()}")
        
        return "\n".join(context[:3])  # Limit context to avoid token limits
    
    async def get_response(self, query: str) -> str:
        """Get AI response for IPL-related queries"""
        context = self.get_context_data(query)
        
        prompt = f"""
        You are an expert IPL cricket analyst with access to historical match data. 
        Answer the user's question about IPL cricket using the provided context and your knowledge.
        
        Context from recent data:
        {context}
        
        User Question: {query}
        
        Provide a helpful, accurate response about IPL cricket. If you don't have specific data, 
        mention that and provide general knowledge instead.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert IPL cricket analyst providing accurate information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I apologize, but I'm unable to process your query at the moment. Please try again later. Error: {str(e)}"

# Enhanced utility functions
def get_comprehensive_h2h_stats(team1: str, team2: str, match_data: pd.DataFrame) -> Dict:
    """Get comprehensive head-to-head statistics"""
    if match_data is None:
        return None
    
    # Filter matches between these teams
    h2h_matches = match_data[
        ((match_data['team1'] == team1) & (match_data['team2'] == team2)) |
        ((match_data['team1'] == team2) & (match_data['team2'] == team1))
    ].copy()
    
    if h2h_matches.empty:
        return None
    
    # Handle missing winners
    h2h_matches['winner'] = h2h_matches['winner'].fillna('No Result')
    
    # Calculate basic stats
    total_matches = len(h2h_matches)
    team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
    team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
    no_results = len(h2h_matches[h2h_matches['winner'] == 'No Result'])
    
    # Calculate win percentages
    team1_win_pct = (team1_wins / total_matches) * 100 if total_matches > 0 else 0
    team2_win_pct = (team2_wins / total_matches) * 100 if total_matches > 0 else 0
    
    # Get recent form (last 10 matches)
    recent_matches = h2h_matches.head(10)
    recent_results = recent_matches['winner'].tolist()
    
    # Calculate average scores if available
    avg_score_team1 = None
    avg_score_team2 = None
    if 'target_runs' in h2h_matches.columns:
        avg_score_team1 = h2h_matches[h2h_matches['team1'] == team1]['target_runs'].mean()
        avg_score_team2 = h2h_matches[h2h_matches['team1'] == team2]['target_runs'].mean()
    
    return {
        'total_matches': total_matches,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'no_results': no_results,
        'team1_win_pct': team1_win_pct,
        'team2_win_pct': team2_win_pct,
        'recent_results': recent_results,
        'avg_score_team1': avg_score_team1,
        'avg_score_team2': avg_score_team2
    }

def predict_outcome(lr_model, rf_model, input_df: pd.DataFrame) -> Dict:
    """Get enhanced predictions from both models"""
    try:
        lr_proba = lr_model.predict_proba(input_df)[0]
        rf_proba = rf_model.predict_proba(input_df)[0]
        
        # Calculate weighted average (RF typically performs better)
        avg_proba = [(lr_proba[0] * 0.4 + rf_proba[0] * 0.6), 
                     (lr_proba[1] * 0.4 + rf_proba[1] * 0.6)]
        
        # Calculate confidence intervals
        confidence = abs(lr_proba[1] - rf_proba[1])
        confidence_level = "High" if confidence < 0.1 else "Medium" if confidence < 0.2 else "Low"
        
        return {
            'lr_proba': lr_proba,
            'rf_proba': rf_proba,
            'avg_proba': avg_proba,
            'confidence': confidence,
            'confidence_level': confidence_level
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def create_enhanced_match_situation(runs_left: int, balls_left: int, wickets: int) -> tuple:
    """Create enhanced match situation analysis"""
    if balls_left <= 0:
        return "Match Over", "error-red", "⏰"
    
    if runs_left <= 0:
        return "Target Achieved!", "success-green", "🎉"
    
    if wickets <= 0:
        return "All Out!", "error-red", "😞"
    
    required_rr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    if required_rr > 18:
        return "Nearly Impossible", "error-red", "🚨"
    elif required_rr > 15:
        return "Very Difficult", "error-red", "⚠️"
    elif required_rr > 12:
        return "Difficult", "warning-yellow", "😰"
    elif required_rr > 9:
        return "Challenging", "warning-yellow", "😓"
    elif required_rr > 6:
        return "Moderate", "warning-yellow", "🤔"
    elif required_rr > 3:
        return "Comfortable", "success-green", "😊"
    else:
        return "Very Easy", "success-green", "😎"

# Enhanced visualization functions
def create_probability_gauge(probability: float, title: str) -> go.Figure:
    """Create a gauge chart for win probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "blue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig

def create_h2h_visualization(h2h_stats: Dict, team1: str, team2: str) -> go.Figure:
    """Create enhanced head-to-head visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Overall H2H Record", "Win Percentage"),
        horizontal_spacing=0.1
    )
    
    # Pie chart for overall record
    fig.add_trace(
        go.Pie(
            labels=[team1, team2, 'No Result'],
            values=[h2h_stats['team1_wins'], h2h_stats['team2_wins'], h2h_stats['no_results']],
            hole=0.3,
            marker_colors=['#3b82f6', '#ef4444', '#6b7280']
        ),
        row=1, col=1
    )
    
    # Bar chart for win percentages
    fig.add_trace(
        go.Bar(
            x=[team1, team2],
            y=[h2h_stats['team1_win_pct'], h2h_stats['team2_win_pct']],
            marker_color=['#3b82f6', '#ef4444'],
            text=[f"{h2h_stats['team1_win_pct']:.1f}%", f"{h2h_stats['team2_win_pct']:.1f}%"],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text=f"{team1} vs {team2} - Historical Analysis",
        font={'family': "Arial", 'size': 12}
    )
    
    return fig

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 CricWin Pro</h1>
        <p>AI-Powered IPL Match Prediction & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        match_data = load_match_data()
        deliveries_data = load_deliveries_data()
        lr_model, rf_model, lr_metrics, rf_metrics = load_models()
    
    # Initialize AI components
    ai_insights = AIInsightsGenerator()
    chatbot = IPLChatbot(match_data, deliveries_data)
    
    # Sidebar for match inputs
    with st.sidebar:
        st.header("⚙️ Match Configuration")
        
        # Team selection
        batting_team = st.selectbox("🏏 Batting Team", TEAMS, index=0)
        bowling_team = st.selectbox("🎳 Bowling Team", 
                                   [team for team in TEAMS if team != batting_team], 
                                   index=1)
        
        # Venue selection
        city = st.selectbox("🏟️ Venue", CITIES, index=2)
        
        # Match parameters
        st.subheader("📊 Match Parameters")
        target = st.number_input("🎯 Target Score", min_value=100, max_value=300, value=180, step=5)
        current_score = st.number_input("📈 Current Score", min_value=0, max_value=target, value=120, step=1)
        overs_completed = st.number_input("⏱️ Overs Completed", min_value=0.0, max_value=20.0, value=15.0, step=0.1)
        wickets_lost = st.number_input("❌ Wickets Lost", min_value=0, max_value=10, value=3, step=1)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            pitch_condition = st.selectbox("Pitch Condition", ["Batting Friendly", "Bowling Friendly", "Neutral"])
            weather_condition = st.selectbox("Weather", ["Clear", "Overcast", "Humid"])
            pressure_situation = st.selectbox("Pressure Level", ["Low", "Medium", "High"])
    
    # Calculate derived metrics
    runs_left = target - current_score
    balls_left = 120 - int(overs_completed * 6)
    wickets_remaining = 10 - wickets_lost
    crr = (current_score * 6) / (120 - balls_left) if (120 - balls_left) > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    # Main dashboard
    st.markdown("## 📊 Live Match Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Runs Needed</h3>
            <h2 style="color: var(--primary-blue); margin: 0;">{runs_left}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Balls Left</h3>
            <h2 style="color: var(--accent-orange); margin: 0;">{balls_left}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏏 Wickets</h3>
            <h2 style="color: var(--success-green); margin: 0;">{wickets_remaining}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 CRR</h3>
            <h2 style="color: var(--secondary-blue); margin: 0;">{crr:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎪 RRR</h3>
            <h2 style="color: var(--error-red); margin: 0;">{rrr:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Match situation analysis
    situation, color_class, emoji = create_enhanced_match_situation(runs_left, balls_left, wickets_remaining)
    
    st.markdown(f"""
    <div class="metric-card">
        <h2>{emoji} Match Situation: <span class="status-indicator status-{color_class.replace('-', '-')}">{situation}</span></h2>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
            {batting_team} needs {runs_left} runs from {balls_left} balls at {rrr:.2f} RPO
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Head-to-Head Analysis
    st.markdown("## 🤝 Head-to-Head Analysis")
    
    h2h_stats = get_comprehensive_h2h_stats(batting_team, bowling_team, match_data)
    
    if h2h_stats and h2h_stats['total_matches'] > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Enhanced H2H visualization
            h2h_fig = create_h2h_visualization(h2h_stats, batting_team, bowling_team)
            st.plotly_chart(h2h_fig, use_container_width=True)
        
        with col2:
            # Recent form
            st.markdown("### 📊 Recent Form")
            recent_results = h2h_stats['recent_results'][:5]
            
            for i, result in enumerate(recent_results):
                if result == batting_team:
                    st.markdown(f"**Match {i+1}:** ✅ {batting_team} won")
                elif result == bowling_team:
                    st.markdown(f"**Match {i+1}:** ❌ {bowling_team} won")
                else:
                    st.markdown(f"**Match {i+1}:** ⚪ No Result")
            
            # Head-to-head summary
            st.markdown("### 📈 H2H Summary")
            st.markdown(f"**Total Matches:** {h2h_stats['total_matches']}")
            st.markdown(f"**{batting_team}:** {h2h_stats['team1_wins']} wins ({h2h_stats['team1_win_pct']:.1f}%)")
            st.markdown(f"**{bowling_team}:** {h2h_stats['team2_wins']} wins ({h2h_stats['team2_win_pct']:.1f}%)")
    else:
        st.warning("📊 No historical data available for these teams")
    
    # Prediction Section
    st.markdown("## 🔮 AI-Powered Predictions")
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        # Prepare input data
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })
        
        with st.spinner("🤖 AI is analyzing the match situation..."):
            # Get ML predictions
            prediction_results = predict_outcome(lr_model, rf_model, input_df)
            
            if prediction_results:
                lr_proba = prediction_results['lr_proba']
                rf_proba = prediction_results['rf_proba']
                avg_proba = prediction_results['avg_proba']
                confidence_level = prediction_results['confidence_level']
                
                # Display predictions with gauges
                st.markdown("### 📊 Model Predictions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    lr_gauge = create_probability_gauge(lr_proba[1], "Logistic Regression")
                    st.plotly_chart(lr_gauge, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>📈 Logistic Regression</h3>
                        <div class="probability">{lr_proba[1]*100:.1f}%</div>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rf_gauge = create_probability_gauge(rf_proba[1], "Random Forest")
                    st.plotly_chart(rf_gauge, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🌲 Random Forest</h3>
                        <div class="probability">{rf_proba[1]*100:.1f}%</div>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_gauge = create_probability_gauge(avg_proba[1], "Ensemble Average")
                    st.plotly_chart(avg_gauge, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Final Prediction</h3>
                        <div class="probability">{avg_proba[1]*100:.1f}%</div>
                        <p>Confidence: {confidence_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI-Generated Match Commentary
                st.markdown("### 🎤 AI Match Commentary")
                
                match_data_for_ai = {
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'city': city,
                    'target': target,
                    'current_score': current_score,
                    'wickets_lost': wickets_lost,
                    'overs_completed': overs_completed,
                    'rrr': rrr
                }
                
                try:
                    # Generate commentary (synchronous call for simplicity)
                    commentary_prompt = f"""
                    Generate engaging cricket commentary for the current IPL match situation:
                    
                    Match Details:
                    - {batting_team} vs {bowling_team}
                    - Venue: {city}
                    - Target: {target} runs
                    - Current Score: {current_score}/{wickets_lost}
                    - Overs: {overs_completed}/20
                    - Required Run Rate: {rrr:.2f}
                    - Win Probability: {avg_proba[1]*100:.1f}%
                    
                    Provide brief, exciting commentary (2-3 sentences) that captures the current match situation and tension.
                    """
                    
                    commentary_response = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": "You are an expert cricket commentator providing engaging IPL match analysis."},
                            {"role": "user", "content": commentary_prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    commentary = commentary_response.choices[0].message.content.strip()
                    
                    st.markdown(f"""
                    <div class="ai-insights">
                        <h3>🎤 Live Commentary</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">{commentary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="ai-insights">
                        <h3>🎤 Live Commentary</h3>
                        <p>What an intense situation! {batting_team} needs {runs_left} runs from {balls_left} balls. 
                        With {wickets_remaining} wickets in hand and a required run rate of {rrr:.2f}, 
                        this match could go either way!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI Prediction Analysis
                st.markdown("### 🧠 AI Prediction Analysis")
                
                prediction_data_for_ai = {
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'win_probability': avg_proba[1] * 100,
                    'lr_prob': lr_proba[1] * 100,
                    'rf_prob': rf_proba[1] * 100,
                    'runs_left': runs_left,
                    'balls_left': balls_left,
                    'wickets': wickets_remaining,
                    'rrr': rrr
                }
                
                try:
                    analysis_prompt = f"""
                    Analyze the following IPL match prediction data and provide expert insights:
                    
                    Match: {batting_team} vs {bowling_team}
                    Win Probability: {avg_proba[1]*100:.1f}%
                    
                    Logistic Regression: {lr_proba[1]*100:.1f}%
                    Random Forest: {rf_proba[1]*100:.1f}%
                    
                    Match Situation:
                    - Runs needed: {runs_left}
                    - Balls left: {balls_left}
                    - Wickets in hand: {wickets_remaining}
                    - Required RR: {rrr:.2f}
                    
                    Provide a brief analysis (2-3 sentences) explaining the prediction and key factors.
                    """
                    
                    analysis_response = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": "You are an expert cricket analyst providing match prediction insights."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=150,
                        temperature=0.6
                    )
                    
                    analysis = analysis_response.choices[0].message.content.strip()
                    
                    st.markdown(f"""
                    <div class="ai-insights">
                        <h3>🧠 Expert Analysis</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">{analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="ai-insights">
                        <h3>🧠 Expert Analysis</h3>
                        <p>Based on the current match situation and historical data, our AI models predict a 
                        {avg_proba[1]*100:.1f}% chance of {batting_team} winning. The required run rate of {rrr:.2f} 
                        is {'challenging' if rrr > 10 else 'manageable'} with {wickets_remaining} wickets remaining.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # IPL Chatbot Section
    st.markdown("## 🤖 IPL AI Assistant")
    
    st.markdown("""
    <div class="metric-card">
        <h3>💬 Ask me anything about IPL!</h3>
        <p>I can help you with IPL statistics, team performance, player records, match history, and more.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_query = st.text_input("Ask your IPL question here:", placeholder="e.g., Which team has won the most IPL titles?")
    
    if st.button("🚀 Ask AI", type="secondary") and user_query:
        with st.spinner("🤖 AI is thinking..."):
            try:
                # Get AI response
                chat_prompt = f"""
                You are an expert IPL cricket analyst with access to historical match data. 
                Answer the user's question about IPL cricket using your knowledge and any available context.
                
                User Question: {user_query}
                
                Provide a helpful, accurate response about IPL cricket. Keep it conversational and informative.
                """
                
                chat_response = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert IPL cricket analyst providing accurate information."},
                        {"role": "user", "content": chat_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                ai_response = chat_response.choices[0].message.content.strip()

                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"Error getting AI response: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <strong>🤖 AI Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Model Performance Analytics
    st.markdown("## 📈 Model Performance Analytics")
    
    if lr_metrics and rf_metrics:
        # Create accuracy comparison
        metrics_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [
                lr_metrics.get('accuracy', 0),
                rf_metrics.get('accuracy', 0)
            ]
        })
        
        # Display accuracy values
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Logistic Regression Accuracy", f"{lr_metrics.get('accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("Random Forest Accuracy", f"{rf_metrics.get('accuracy', 0)*100:.2f}%")
        
        # Accuracy metrics visualization
        accuracy_fig = px.bar(
            metrics_df,
            x='Model',
            y='Accuracy',
            title='Model Accuracy Comparison',
            color='Model',
            color_discrete_map={
                'Logistic Regression': '#3b82f6',
                'Random Forest': '#10b981'
            }
        )
        
        accuracy_fig.update_layout(
            height=400,
            font={'family': "Arial"},
            title_font_size=16,
            yaxis_title='Accuracy Score',
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(accuracy_fig, use_container_width=True)

    # How It Works Section
    st.markdown("## 🔬 How CricWin Pro Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Machine Learning Models</h3>
            <ul>
                <li><strong>Logistic Regression:</strong> Statistical model for probability estimation</li>
                <li><strong>Random Forest:</strong> Ensemble method handling complex patterns</li>
                <li><strong>Ensemble Prediction:</strong> Weighted average of both models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Key Features</h3>
            <ul>
                <li>Real-time match situation analysis</li>
                <li>Historical head-to-head statistics</li>
                <li>AI-powered commentary and insights</li>
                <li>Interactive IPL knowledge chatbot</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Technology Section
    st.markdown("""
    <div class="ai-insights">
        <h3>🧠 Powered by Azure OpenAI GPT-4o</h3>
        <p>Our AI assistant uses advanced language models to provide:</p>
        <ul>
            <li>Real-time match commentary and analysis</li>
            <li>Intelligent prediction explanations</li>
            <li>Comprehensive IPL knowledge base</li>
            <li>Interactive Q&A capabilities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer with disclaimers
    st.markdown("---")
    st.markdown("""
<div class="metric-card">
    <h3>⚠️ Important Notes</h3>
    <ul>
        <li>🤖 <strong>AI-Powered:</strong> Enhanced with Azure OpenAI GPT-4o for intelligent insights</li>
        <li>📊 <strong>Data-Driven:</strong> Predictions based on historical IPL match data and advanced ML models</li>
        <li>🎯 <strong>Entertainment Only:</strong> Use for entertainment purposes only, not for betting</li>
        <li>⚡ <strong>Real-time Factors:</strong> Player form, weather, and pitch conditions may affect outcomes</li>
        <li>📅 <strong>Dataset Limitations:</strong>
            <ul style="margin-top: 0.5rem; margin-bottom: 0;">
                <li>Current data covers <strong>2008-2017 IPL seasons</strong></li>
                <li>Team strategies and player rosters may have changed</li>
                <li>Dataset will be updated to <strong>include 2018-2025 matches</strong> for 2026 predictions</li>
                <li>Treat predictions as <strong>historical trends</strong> rather than absolute forecasts</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)
    
    # Performance info
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
        <p>Developed by [SAHID/TACTGEN] | Built with ❤️ using Streamlit, Azure OpenAI, and Advanced ML | Version 2.0 Pro</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()