
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
print("Loading and preprocessing data...")
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Calculate total scores for first innings
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge match data with total scores
match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')

# Define current IPL teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

# Normalize team names
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team1'] = match_df['team1'].str.replace('Gujarat Lions','Gujarat Titans')
match_df['team2'] = match_df['team2'].str.replace('Gujarat Lions','Gujarat Titans')

# Normalize winner column for old team names
match_df['winner'] = match_df['winner'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['winner'] = match_df['winner'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['winner'] = match_df['winner'].str.replace('Gujarat Lions','Gujarat Titans')

# Filter matches with current teams only
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df = match_df[['match_id','city','winner','total_runs']]

# Create delivery dataframe for second innings
delivery_df = match_df.merge(delivery,on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]
delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')

# Normalize batting_team and bowling_team for old team names
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Gujarat Lions','Gujarat Titans')
delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Gujarat Lions','Gujarat Titans')

# Calculate match state features
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])

# Calculate wickets remaining
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].notnull().astype(int)
delivery_df['wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()

# Calculate rates
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']

# Create target variable
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result,axis=1)

# Prepare final dataset
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
final_df = final_df.sample(final_df.shape[0], random_state=42)
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

print(f"Dataset shape: {final_df.shape}")
print(f"Target distribution:\n{final_df['result'].value_counts()}")

# Split features and target
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipeline
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team','bowling_team','city'])
], remainder='passthrough')

# Create and train Logistic Regression model
print("\nTraining Logistic Regression model...")
lr_pipe = Pipeline(steps=[
    ('preprocessing', trf),
    ('classifier', LogisticRegression(solver='lbfgs', C=0.0006951927961775605, max_iter=1000, random_state=42))
])

lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)

# Create and train Random Forest model
print("Training Random Forest model...")
rf_pipe = Pipeline(steps=[
    ('preprocessing', trf),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
])

rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)

# Calculate metrics for both models
def calculate_metrics(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

# Calculate metrics
lr_metrics = calculate_metrics(y_test, lr_pred, 'Logistic Regression')
rf_metrics = calculate_metrics(y_test, rf_pred, 'Random Forest')

# Create comparison dataframe
comparison_df = pd.DataFrame([lr_metrics, rf_metrics])

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print(comparison_df.round(4))

# Test case for prediction
input_df = pd.DataFrame({
    'batting_team': ['Sunrisers Hyderabad'],
    'bowling_team': ['Royal Challengers Bangalore'],
    'city': ['Bangalore'],
    'runs_left': [50],
    'balls_left': [30],
    'wickets': [3],
    'total_runs_x': [210],
    'crr': [6.0],
    'rrr': [10.0]
})

# Get predictions from both models
lr_proba = lr_pipe.predict_proba(input_df)
rf_proba = rf_pipe.predict_proba(input_df)

# Calculate probabilities
lr_loss_prob = lr_proba[0][0]
lr_win_prob = lr_proba[0][1]

rf_loss_prob = rf_proba[0][0]
rf_win_prob = rf_proba[0][1]

# Calculate average probabilities
avg_win_prob = (lr_win_prob + rf_win_prob) / 2
avg_loss_prob = (lr_loss_prob + rf_loss_prob) / 2

print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
print("Scenario: Sunrisers Hyderabad vs Royal Challengers Bangalore")
print("Location: Bangalore")
print("Situation: 50 runs needed from 30 balls, 3 wickets in hand")
print("Target: 210 runs")
print("-" * 60)

print(f"Logistic Regression:")
print(f"  Win Probability:  {lr_win_prob*100:.1f}%")
print(f"  Loss Probability: {lr_loss_prob*100:.1f}%")

print(f"\nRandom Forest:")
print(f"  Win Probability:  {rf_win_prob*100:.1f}%")
print(f"  Loss Probability: {rf_loss_prob*100:.1f}%")

print(f"\nAverage of Both Models:")
print(f"  Win Probability:  {avg_win_prob*100:.1f}%")
print(f"  Loss Probability: {avg_loss_prob*100:.1f}%")

# Feature importance for Random Forest
feature_names = (rf_pipe.named_steps['preprocessing']
                .named_transformers_['trf']
                .get_feature_names_out(['batting_team','bowling_team','city']).tolist() +
                ['runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'])

rf_importance = rf_pipe.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
print("="*60)
print(importance_df.head(10).to_string(index=False))

# In your run.py, replace the model saving section with this:

# Save models with their metrics
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump({
        'model': lr_pipe,
        'metrics': lr_metrics,
        'feature_names': feature_names,
        'test_case_prediction': {
            'input': input_df.iloc[0].to_dict(),
            'output': {
                'win_prob': lr_win_prob,
                'loss_prob': lr_loss_prob
            }
        }
    }, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump({
        'model': rf_pipe,
        'metrics': rf_metrics,
        'feature_importance': importance_df.to_dict('records'),
        'test_case_prediction': {
            'input': input_df.iloc[0].to_dict(),
            'output': {
                'win_prob': rf_win_prob,
                'loss_prob': rf_loss_prob
            }
        }
    }, f)

# Also save the metrics separately for easy access
with open('model_metrics.pkl', 'wb') as f:
    pickle.dump({
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics,
        'last_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, f)

print(f"\nModels and metrics saved successfully!")

# Function to make predictions with both models
def predict_match_outcome_both_models(lr_model, rf_model, input_df):
    """
    Make predictions using both models and return average probabilities
    """
    lr_proba = lr_model.predict_proba(input_df)
    rf_proba = rf_model.predict_proba(input_df)
    
    lr_win_prob = lr_proba[0][1]
    lr_loss_prob = lr_proba[0][0]
    
    rf_win_prob = rf_proba[0][1]
    rf_loss_prob = rf_proba[0][0]
    
    avg_win_prob = (lr_win_prob + rf_win_prob) / 2
    avg_loss_prob = (lr_loss_prob + rf_loss_prob) / 2
    
    return {
        'logistic_regression': {'win': lr_win_prob, 'loss': lr_loss_prob},
        'random_forest': {'win': rf_win_prob, 'loss': rf_loss_prob},
        'average': {'win': avg_win_prob, 'loss': avg_loss_prob}
    }

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print("The code implements an IPL match outcome prediction system using:")
print("1. Data preprocessing and feature engineering")
print("2. Two machine learning models: Logistic Regression and Random Forest")
print("3. Comprehensive model comparison and evaluation")
print("4. Ensemble prediction using average probabilities")
print("\nKey features used for prediction:")
print("- Team matchups (batting and bowling teams)")
print("- Match venue (city)")
print("- Match situation (runs left, balls left, wickets remaining)")
print("- Performance metrics (current run rate, required run rate)")