import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


# Load the updated dataset from the EDA to a dataframe
df = pd.read_csv('0_Updated_f1_master_database.csv')

# Check the dataset
print(df.head())


# # STEP 1: PREDICT THE QUALIFYING SESSION

# ## 1.1: Feature engineering

# ### 1.1.1: Get Drivers and Teams historical qualifying performance

# Driver's overall average quali time (-> driver's skill level)
driver_quali_stats = df.groupby('Driver').agg({
    'Best_Quali_Time': ['mean', 'median', 'std', 'min']
}).round(4)

driver_quali_stats.columns = ['Driver_Avg_Quali', 'Driver_Median_Quali', 'Driver_Std_Quali', 'Driver_Best_Quali']

print("DRIVER QUALIFYING STATISTICS")
driver_quali_stats.head(10)

# Merge to main dataframe
df = df.merge(driver_quali_stats, left_on='Driver', right_index=True, how='left')

# Team's overall average quali time (-> team's performance)
team_quali_stats = df.groupby('Team').agg({
    'Best_Quali_Time': ['mean', 'median', 'std']
}).round(4)

team_quali_stats.columns = ['Team_Avg_Quali', 'Team_Median_Quali', 'Team_Std_Quali']

print("TEAM QUALIFYING STATISTICS")
team_quali_stats

# Merge to main dataframe
df = df.merge(team_quali_stats, left_on='Team', right_index=True, how='left')

# df.head()

# %%
# ### 1.1.2: Events historical features

# Events historical qualifying times

event_quali_stats = df.groupby('Event').agg({
    'Best_Quali_Time': ['mean', 'std', 'min', 'max']
}).round(4)

event_quali_stats.columns = ['Event_Avg_Quali', 'Event_Std_Quali', 
                             'Event_Min_Quali', 'Event_Max_Quali']

print("EVENT QUALIFYING STATISTICS")
event_quali_stats

# merge to main dataframe

df = df.merge(event_quali_stats, left_on='Event', right_index=True, how='left')

#Check the dataframe (uncomment next line)
# print(df.head())

# %% 
# ### 1.1.3: Driver avg quali performance by event

# Historical performance at specific events
driver_event_quali = df.groupby(['Driver', 'Event']).agg({
    'Best_Quali_Time': ['mean', 'count']
}).round(4)

driver_event_quali.columns = ['Driver_Event_Avg_Quali', 'Driver_Event_Count']

print("DRIVER-EVENT STATISTICS")
driver_event_quali.head(20)

# %% 
# ## 1.2: Data preparation

# %%
# ### 1.2.1: Create training dataset

# Select features for modeling
feature_columns = [
    'Driver',
    'Team',
    'Year',
    'Event',
    'FP1_AvgLapTime',
    'FP1_BestLapTime',
    'FP1_AvgSpeedI1',
    'FP1_AvgSpeedI2',
    'FP1_AvgSpeedFL',
    'FP1_AvgSpeedST',
    'FP2_AvgLapTime',
    'FP2_BestLapTime',
    'FP2_AvgSpeedI1',
    'FP2_AvgSpeedI2',
    'FP2_AvgSpeedFL',
    'FP2_AvgSpeedST',
    'FP3_AvgLapTime',
    'FP3_BestLapTime',
    'FP3_AvgSpeedI1',
    'FP3_AvgSpeedI2',
    'FP3_AvgSpeedFL',
    'FP3_AvgSpeedST',
    # Historical features
    'Driver_Avg_Quali',
    'Driver_Median_Quali',
    'Driver_Std_Quali',
    'Driver_Best_Quali',
    'Team_Avg_Quali',
    'Team_Median_Quali',
    'Team_Std_Quali',
    'Event_Avg_Quali',
    'Event_Std_Quali',
    'Event_Min_Quali',
    'Event_Max_Quali'
]

target_column = 'Best_Quali_Time'


# Create clean dataset (remove rows with missing target)
df_model = df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()

print(f"Dataset for modeling: {df_model.shape}")
print(f"Rows with complete data: {df_model.dropna().shape[0]}")

# print(df_model.head())

# Check missing values in features
print("Missing values in features:")
missing_summary = df_model.isnull().sum()
missing_summary[missing_summary > 0].sort_values(ascending=False)

# %%
# ### 1.2.2: Handle missing values

# %%
# Fill missing free practice data with mean by event

fp_columns = [col for col in feature_columns if col.startswith('FP')]

for col in fp_columns:
    df_model[col] = df_model.groupby('Event')[col].transform(
        lambda x: x.fillna(x.mean())
    )

# If still missing, fill with overall mean
df_model[fp_columns] = df_model[fp_columns].fillna(df_model[fp_columns].mean())

print("Missing values after handling:")
print(df_model.isnull().sum().sum())

# Remove any remaining rows with NaN
df_model = df_model.dropna()
print(f"Final dataset size: {df_model.shape}")

# %%
# ### 1.2.3: Categorical features encoding

# %%
# Encode driver, team, and event
driver_encoder = LabelEncoder()
team_encoder = LabelEncoder()
event_encoder = LabelEncoder()

df_model['Driver_Encoded'] = driver_encoder.fit_transform(df_model['Driver'])
df_model['Team_Encoded'] = team_encoder.fit_transform(df_model['Team'])
df_model['Event_Encoded'] = event_encoder.fit_transform(df_model['Event'])

print(f"Unique Drivers: {len(driver_encoder.classes_)}")
print(f"Unique Teams: {len(team_encoder.classes_)}")
print(f"Unique Events: {len(event_encoder.classes_)}")

# Save encoders for later use (for predicting new races)
pickle.dump(driver_encoder, open('driver_encoder.pkl', 'wb'))
pickle.dump(team_encoder, open('team_encoder.pkl', 'wb'))
pickle.dump(event_encoder, open('event_encoder.pkl', 'wb'))

# %%
# ## 1.3: Model training

# %%
# ### 1.3.1: Training and test sets

# %%
# Final features (excluding original categorical columns)
final_features = [col for col in feature_columns if col not in ['Driver', 'Team', 'Event']] + \
                 ['Driver_Encoded', 'Team_Encoded', 'Event_Encoded']

X = df_model[final_features]
y = df_model[target_column]

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")

# %%
# Split: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set: {X_train.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

# %%
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for predictions
pickle.dump(scaler, open('scaler_quali.pkl', 'wb'))

# %%
# ### 1.3.2: Train multiple models

# %%
# Model 1: Random Forest --> is overfitting
print("Random Forest")
rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

print(f"Training R²: {r2_score(y_train, rf_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, rf_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, rf_pred_test):.4f} seconds")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred_test)):.4f} seconds")

# %%
# Model 2: Gradient Boosting with regularization
# Gradient Boosting was best_model but Overfitting, so we performed regualrization

print("\n" + "="*60)
print("SIMPLIFIED REGULARIZATION") 
print("="*60 + "\n")

# Conservative regularization settings
gb_regularized = GradientBoostingRegressor(
    n_estimators=100,           # ← Fewer boosting rounds
    max_depth=3,                # ← VERY shallow trees
    learning_rate=0.05,         # ← Slower learning
    min_samples_split=20,       # ← Require more samples to split
    min_samples_leaf=10,        # ← Require more samples in leaf nodes
    subsample=0.8,              # ← Use 80% of training data per tree
    max_features='sqrt',        # ← Use sqrt of features (sklearn equivalent to colsample)
    random_state=42
)

print("Training regularized Gradient Boosting...")
gb_regularized.fit(X_train, y_train)

y_train_pred_reg = gb_regularized.predict(X_train)
y_test_pred_reg = gb_regularized.predict(X_test)

print(f"\n--- Performance with Regularization ---")
train_r2_reg = r2_score(y_train, y_train_pred_reg)
test_r2_reg = r2_score(y_test, y_test_pred_reg)
gap_reg = train_r2_reg - test_r2_reg

print(f"Training R²: {train_r2_reg:.4f}")
print(f"Test R²: {test_r2_reg:.4f}")

print(f"\nTraining MAE: {mean_absolute_error(y_train, y_train_pred_reg):.4f}s")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred_reg):.4f}s")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_reg)):.4f}s")

# %%
"""
# Model 2: Gradient Boosting with GridSearchCV
# GB was still overfitting even after regularization, so we made adjustments in the hyperparameters with GridSearchCV
# This takes more than 15min to run...

print("Tuning GB to avoid overfitting")

# Define hyperparameter grid (focus on regularization)
param_grid = {
    'n_estimators': [50, 100, 150],          # Fewer trees
    'max_depth': [3, 4, 5, 6],               # Shallower trees
    'learning_rate': [0.01, 0.05, 0.1],      # Slower learning
    'min_samples_split': [10, 15, 20],       # Require more samples to split
    'min_samples_leaf': [5, 10, 15],         # Require more samples in leaves
    'subsample': [0.7, 0.8, 0.9, 1.0],       # Use subset of training data
    'max_features': ['sqrt', 'log2', None]   # Use subset of features
}

# Create base model
gb_base = GradientBoostingRegressor(random_state=42)

# Grid search with cross-validation
print("Performing Grid Search (this may take a few minutes)...\n")

grid_search = GridSearchCV(
    gb_base,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Best parameters found:")
best_params = grid_search.best_params_
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nBest CV R² Score: {grid_search.best_score_:.4f}")

# Use the best model
best_model_tuned = grid_search.best_estimator_

# Evaluate
y_train_pred_tuned = best_model_tuned.predict(X_train)
y_test_pred_tuned = best_model_tuned.predict(X_test)

print(f"\n--- Performance After Tuning ---")
print(f"Training R²: {r2_score(y_train, y_train_pred_tuned):.4f}")
print(f"Test R²: {r2_score(y_test, y_test_pred_tuned):.4f}")
print(f"Overfitting Gap: {r2_score(y_train, y_train_pred_tuned) - r2_score(y_test, y_test_pred_tuned):.4f}")

print(f"\nTraining MAE: {mean_absolute_error(y_train, y_train_pred_tuned):.4f}s")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred_tuned):.4f}s")

print("Gradient Boosting")
gb_model = GradientBoostingRegressor(
    n_estimators=50, # was 100
    max_depth=3, # was 5
    learning_rate=0.05, # was 0.1
    min_samples_split=15, # was 5
    min_samples_leaf=7, # was 2
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)

print(f"Training R²: {r2_score(y_train, gb_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, gb_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, gb_pred_test):.4f} seconds")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, gb_pred_test)):.4f} seconds")
"""

# %%
# Model 2: Gradient Boosting with agressive regularization
# GB was still overfitting after GridSearchCV, so we made used very conservative hyperparameters

print("\n" + "="*60)
print("AGGRESSIVE REGULARIZATION (ULTRA-CONSERVATIVE)")
print("="*60 + "\n")

# Very conservative settings
gb_aggressive = GradientBoostingRegressor(
    n_estimators=50,            # ← Very few trees
    max_depth=2,                # ← Stumps (almost shallow as possible)
    learning_rate=0.01,         # ← Very slow learning
    min_samples_split=30,       # ← Require many samples to split
    min_samples_leaf=20,        # ← Require many samples in leaves
    subsample=0.7,              # ← Use only 70% of training data
    max_features='log2',        # ← Use log2 of features
    random_state=42,
    tol=1e-4
)

print("Training ULTRA-regularized Gradient Boosting...")
gb_aggressive.fit(X_train, y_train)

y_train_pred_agg = gb_aggressive.predict(X_train)
y_test_pred_agg = gb_aggressive.predict(X_test)

# Check overfitting directly

train_r2_agg = r2_score(y_train, y_train_pred_agg)
test_r2_agg = r2_score(y_test, y_test_pred_agg)
gap_agg = train_r2_agg - test_r2_agg

print(f"\n--- Performance with Aggressive Regularization ---")
print(f"Training R²: {train_r2_agg:.4f}")
print(f"Test R²: {test_r2_agg:.4f}")

print(f"\nTraining MAE: {mean_absolute_error(y_train, y_train_pred_agg):.4f}s")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred_agg):.4f}s")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_agg)):.4f}s")

# %%
# Model 3: Ensemble of models

print("\n" + "="*60)
print("ENSEMBLE APPROACH - COMBINING MULTIPLE MODELS")
print("="*60 + "\n")

from sklearn.ensemble import VotingRegressor

# Create ensemble of different model types
ensemble_model = VotingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_split=20, 
            min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10, subsample=0.8,
            max_features='sqrt', random_state=42
        ))
    ],
    weights=[0.5, 0.5]
)

print("Training Voting Ensemble...")
ensemble_model.fit(X_train, y_train)

y_train_pred_ens = ensemble_model.predict(X_train)
y_test_pred_ens = ensemble_model.predict(X_test)

train_r2_ens = r2_score(y_train, y_train_pred_ens)
test_r2_ens = r2_score(y_test, y_test_pred_ens)
gap_ens = train_r2_ens - test_r2_ens

print(f"\n--- Ensemble Performance ---")
print(f"Training R²: {train_r2_ens:.4f}")
print(f"Test R²: {test_r2_ens:.4f}")

print(f"\nTraining MAE: {mean_absolute_error(y_train, y_train_pred_ens):.4f}s")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred_ens):.4f}s")
# %%
# Save the best model
"""
best_model = rf_model if r2_score(y_test, rf_pred_test) > r2_score(y_test, y_test_pred_agg) else gb_aggressive
model_name = "Random Forest" if best_model == rf_model else "Gradient Boosting - Aggressive Regularisation"
print(f"Best model: {model_name}")

pickle.dump(best_model, open('quali_predictor_model.pkl', 'wb'))
"""

best_model = gb_aggressive # because rf is overfitting

# %% 
# ### 1.3.3: Feature importance analysis

# %%
# Check which features are most important
feature_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
print(feature_importance.head(15))

# %%
# Visualize
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Features for Qualifying Time Prediction')
plt.tight_layout()
plt.show()

# %%
# ## 1.4: Cross-validation
# %%
# ### 1.4.1: K-Fold Cross-Validation

print("CROSS-VALIDATION: MODEL STABILITY")

# Prepare data
X = df_model[final_features]
y = df_model[target_column]

# Define K-Fold (5 folds is standard)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores (R² score)
cv_r2_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='r2', n_jobs=-1)
cv_mae_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
cv_rmse_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)

# Convert RMSE to positive
cv_rmse_scores = np.sqrt(-cv_rmse_scores)
cv_mae_scores = -cv_mae_scores

print("R² Scores (5 folds):")
for i, score in enumerate(cv_r2_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std():.4f})")

print("\nMAE Scores (5 folds) in seconds:")
for i, score in enumerate(cv_mae_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_mae_scores.mean():.4f} (+/- {cv_mae_scores.std():.4f})")

print("\nRMSE Scores (5 folds) in seconds:")
for i, score in enumerate(cv_rmse_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")

# Stability assessment
stability_threshold = 0.1  # 10% variation considered stable
cv_variation = cv_r2_scores.std()

print(f"\nModel Stability Assessment")
print(f"Standard Deviation of R² scores: {cv_variation:.4f}")
if cv_variation < stability_threshold:
    print(f"✓ Model is stable (low variance across folds)")
else:
    print(f"⚠️ Model shows some variance (consider regularization)")

# %%
# ### 1.4.2: Learning-curves (to see if model overfitted)

print("LEARNING CURVES: OVERFITTING")

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, 
    cv=5, 
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training R²', linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')

plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation R²', linewidth=2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves: Training vs Validation Performance')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Interpret learning curves
gap = train_mean[-1] - val_mean[-1]
print(f"Final Training R²: {train_mean[-1]:.4f}")
print(f"Final Validation R²: {val_mean[-1]:.4f}")
print(f"Gap (Train - Val): {gap:.4f}")

print("Overfitting Assessment")
if gap < 0.05:
    print("✓ Model is NOT overfitting (small train-val gap)")
elif gap < 0.15:
    print("⚠️ Model shows slight overfitting (moderate train-val gap)")
else:
    print("✗ Model is OVERFITTING (large train-val gap)")

# %% 
# ## 1.5: Predict Las Vegas 2025 Qualifying Times with gb_aggressive

# %% 
# ### 1.5.1: Prepare the data

# Reminder: best_model = gb_aggressive

# Get list of drivers competing in 2025
drivers_2025 = df[df['Year'] == 2025]['Driver'].unique()
teams_2025 = df[df['Year'] == 2025][['Driver', 'Team']].drop_duplicates()

print(f"Drivers competing in 2025: {len(drivers_2025)}")
print(f"Teams: {len(teams_2025['Team'].unique())}\n")

# Check if 2025 Las Vegas race data exists
vegas_2025_drivers = df[(df['Event'] == 'Las Vegas Grand Prix') & (df['Year'] == 2025)].copy()
vegas_2024 = df[(df['Event'] == 'Las Vegas Grand Prix') & (df['Year'] == 2024)].copy()

print(f"2024 Las Vegas races: {len(vegas_2024)}")
print(f"2025 Las Vegas races (if any completed): {len(vegas_2025_drivers)}\n")

if len(vegas_2025_drivers) > 0:
    print("⚠️ 2025 Las Vegas already has race data. Using actual data as comparison.")
else:
    print("✓ 2025 Las Vegas not yet completed. Ready for prediction.\n")

# Create prediction dataset
prediction_data = []

for driver in drivers_2025:
    # Get driver info for 2025
    driver_2025_data = df[df['Driver'] == driver].sort_values('Year', ascending=False).iloc[0]
    
    # Get team
    team = driver_2025_data['Team']
    
    # Get most recent FP data (from 2025 races or 2024 if not available)
    driver_2025_recent = df[(df['Driver'] == driver) & (df['Year'] == 2025)].sort_values('Event', ascending=False)
    
    if len(driver_2025_recent) > 0:
        # Use most recent 2025 race FP data
        fp_data = driver_2025_recent.iloc[0]
    else:
        # Fallback to 2024 Las Vegas FP data
        driver_2024_vegas = df[(df['Driver'] == driver) & 
                               (df['Event'] == 'Las Vegas Grand Prix') & 
                               (df['Year'] == 2024)]
        if len(driver_2024_vegas) > 0:
            fp_data = driver_2024_vegas.iloc[0]
        else:
            # Fallback to average FP data for all 2024 races
            driver_2024_all = df[(df['Driver'] == driver) & (df['Year'] == 2024)]
            if len(driver_2024_all) > 0:
                fp_data = driver_2024_all.mean()
            else:
                # New driver - use 2025 average
                fp_data = df[(df['Driver'] == driver) & (df['Year'] == 2025)].mean()
    
    # Get historical averages
    driver_avg_quali = driver_quali_stats.loc[driver, 'Driver_Avg_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_median_quali = driver_quali_stats.loc[driver, 'Driver_Median_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_std_quali = driver_quali_stats.loc[driver, 'Driver_Std_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_best_quali = driver_quali_stats.loc[driver, 'Driver_Best_Quali'] if driver in driver_quali_stats.index else np.nan
    
    team_avg_quali = team_quali_stats.loc[team, 'Team_Avg_Quali'] if team in team_quali_stats.index else np.nan
    team_median_quali = team_quali_stats.loc[team, 'Team_Median_Quali'] if team in team_quali_stats.index else np.nan
    team_std_quali = team_quali_stats.loc[team, 'Team_Std_Quali'] if team in team_quali_stats.index else np.nan
    
    event_avg_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Avg_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    event_std_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Std_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    event_min_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Min_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    event_max_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Max_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    
    # Build row
    row = {
        'Driver': driver,
        'Team': team,
        'Year': 2025,
        'Event': 'Las Vegas Grand Prix',
        'FP1_AvgLapTime': fp_data.get('FP1_AvgLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_AvgLapTime'] if 'FP1_AvgLapTime' in fp_data.index else np.nan,
        'FP1_BestLapTime': fp_data.get('FP1_BestLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_BestLapTime'] if 'FP1_BestLapTime' in fp_data.index else np.nan,
        'FP1_AvgSpeedI1': fp_data.get('FP1_AvgSpeedI1', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_AvgSpeedI1'] if 'FP1_AvgSpeedI1' in fp_data.index else np.nan,
        'FP1_AvgSpeedI2': fp_data.get('FP1_AvgSpeedI2', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_AvgSpeedI2'] if 'FP1_AvgSpeedI2' in fp_data.index else np.nan,
        'FP1_AvgSpeedFL': fp_data.get('FP1_AvgSpeedFL', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_AvgSpeedFL'] if 'FP1_AvgSpeedFL' in fp_data.index else np.nan,
        'FP1_AvgSpeedST': fp_data.get('FP1_AvgSpeedST', np.nan) if isinstance(fp_data, dict) else fp_data['FP1_AvgSpeedST'] if 'FP1_AvgSpeedST' in fp_data.index else np.nan,
        'FP2_AvgLapTime': fp_data.get('FP2_AvgLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_AvgLapTime'] if 'FP2_AvgLapTime' in fp_data.index else np.nan,
        'FP2_BestLapTime': fp_data.get('FP2_BestLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_BestLapTime'] if 'FP2_BestLapTime' in fp_data.index else np.nan,
        'FP2_AvgSpeedI1': fp_data.get('FP2_AvgSpeedI1', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_AvgSpeedI1'] if 'FP2_AvgSpeedI1' in fp_data.index else np.nan,
        'FP2_AvgSpeedI2': fp_data.get('FP2_AvgSpeedI2', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_AvgSpeedI2'] if 'FP2_AvgSpeedI2' in fp_data.index else np.nan,
        'FP2_AvgSpeedFL': fp_data.get('FP2_AvgSpeedFL', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_AvgSpeedFL'] if 'FP2_AvgSpeedFL' in fp_data.index else np.nan,
        'FP2_AvgSpeedST': fp_data.get('FP2_AvgSpeedST', np.nan) if isinstance(fp_data, dict) else fp_data['FP2_AvgSpeedST'] if 'FP2_AvgSpeedST' in fp_data.index else np.nan,
        'FP3_AvgLapTime': fp_data.get('FP3_AvgLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_AvgLapTime'] if 'FP3_AvgLapTime' in fp_data.index else np.nan,
        'FP3_BestLapTime': fp_data.get('FP3_BestLapTime', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_BestLapTime'] if 'FP3_BestLapTime' in fp_data.index else np.nan,
        'FP3_AvgSpeedI1': fp_data.get('FP3_AvgSpeedI1', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_AvgSpeedI1'] if 'FP3_AvgSpeedI1' in fp_data.index else np.nan,
        'FP3_AvgSpeedI2': fp_data.get('FP3_AvgSpeedI2', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_AvgSpeedI2'] if 'FP3_AvgSpeedI2' in fp_data.index else np.nan,
        'FP3_AvgSpeedFL': fp_data.get('FP3_AvgSpeedFL', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_AvgSpeedFL'] if 'FP3_AvgSpeedFL' in fp_data.index else np.nan,
        'FP3_AvgSpeedST': fp_data.get('FP3_AvgSpeedST', np.nan) if isinstance(fp_data, dict) else fp_data['FP3_AvgSpeedST'] if 'FP3_AvgSpeedST' in fp_data.index else np.nan,
        'Driver_Avg_Quali': driver_avg_quali,
        'Driver_Median_Quali': driver_median_quali,
        'Driver_Std_Quali': driver_std_quali,
        'Driver_Best_Quali': driver_best_quali,
        'Team_Avg_Quali': team_avg_quali,
        'Team_Median_Quali': team_median_quali,
        'Team_Std_Quali': team_std_quali,
        'Event_Avg_Quali': event_avg_quali,
        'Event_Std_Quali': event_std_quali,
        'Event_Min_Quali': event_min_quali,
        'Event_Max_Quali': event_max_quali
    }
    
    prediction_data.append(row)

# Create DataFrame
df_predict = pd.DataFrame(prediction_data)

print("Las Vegas 2025 Prediction Dataset Created:")
print(f"  Drivers: {len(df_predict)}")
print(f"  Features: {len(df_predict.columns)}")
print(f"\nSample data:")
print(df_predict[['Driver', 'Team', 'Driver_Avg_Quali', 'Team_Avg_Quali', 'Event_Avg_Quali']].head(10))

# %% 
# ### 1.5.2: Handle missing data

# Check for missing values
missing_in_predict = df_predict.isnull().sum()
print(f"\nMissing values in prediction dataset:")
missing_summary = missing_in_predict[missing_in_predict > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("  None detected yet (will be filled in next step)")

# We will use 2025 Azerbaijan Grand Prix (most similar to Las Vegas circuit, with this year's cars & drivers performance)

# Fill missing FP data with 2025 Azerbaijan Grand Prix event averages
fp_columns = [col for col in df_predict.columns if col.startswith('FP')]

for col in fp_columns:
    # Get average from 2025 Azerbaijan
    event_avg = df[(df['Event'] == 'Azerbaijan Grand Prix') & (df['Year'] == 2025)][col].mean()
    missing_before = df_predict[col].isnull().sum()
    df_predict[col] = df_predict[col].fillna(event_avg)
    missing_after = df_predict[col].isnull().sum()

print(f"Values still missing: {missing_after}")
# Check that missing values are handled
df_predict[['Driver', 'FP2_AvgLapTime', 'FP3_AvgLapTime']].head()

# %%
# ### 1.5.3: Qualifying prediction

# Encode categorical variables in prediction data

# Load the encoders we created during training
driver_encoder = pickle.load(open('driver_encoder.pkl', 'rb'))
team_encoder = pickle.load(open('team_encoder.pkl', 'rb'))
event_encoder = pickle.load(open('event_encoder.pkl', 'rb'))

# Encode the prediction data
df_predict['Driver_Encoded'] = driver_encoder.transform(df_predict['Driver'])
df_predict['Team_Encoded'] = team_encoder.transform(df_predict['Team'])
df_predict['Event_Encoded'] = event_encoder.transform(df_predict['Event'])

print(f"Driver encoded: {df_predict['Driver_Encoded'].nunique()} unique values")
print(f"Team encoded: {df_predict['Team_Encoded'].nunique()} unique values")
print(f"Event encoded: {df_predict['Event_Encoded'].nunique()} unique values")

print(f"\nFeatures for prediction: {len(final_features)}")

# Verify all features exist in prediction data
print(f"\nVerifying feature availability:")
missing_features = [col for col in final_features if col not in df_predict.columns]

if missing_features:
    print(f"⚠️ Missing features: {missing_features}")
    raise KeyError(f"Missing features: {missing_features}")
else:
    print(f"✓ All {len(final_features)} features available in prediction data")

# Extract features for prediction
X_predict = df_predict[final_features]

print(f"Prediction dataset shape: {X_predict.shape}")
X_predict.head()

# Scale features using the fitted scaler
X_predict_scaled = scaler.transform(X_predict)

# Make predictions
predicted_quali_times = best_model.predict(X_predict)

# Get prediction uncertainty (using residual std from training)
residuals_train = y_train - best_model.predict(X_train)
pred_std = np.ones_like(predicted_quali_times) * residuals_train.std()

# Create results DataFrame
results_df = df_predict[['Driver', 'Team']].copy()
results_df['Predicted_Quali_Time'] = predicted_quali_times
results_df['Prediction_Std'] = pred_std
results_df['Lower_95%_CI'] = predicted_quali_times - 1.96 * pred_std
results_df['Upper_95%_CI'] = predicted_quali_times + 1.96 * pred_std

# Sort by predicted qualifying time
results_df = results_df.sort_values('Predicted_Quali_Time').reset_index(drop=True)
results_df['Predicted_Grid_Position'] = range(1, len(results_df) + 1)

print(f"\n✓ Predictions generated for {len(results_df)} drivers")
print(f"\nPREDICTED QUALIFYING TIMES FOR 2025 LAS VEGAS GRAND PRIX:\n")
print(results_df[['Predicted_Grid_Position', 'Driver', 'Team', 'Predicted_Quali_Time', 'Prediction_Std']].to_string(index=False))

# Prediction statistics
print(f"\n--- Prediction Statistics ---")
print(f"Fastest predicted time: {results_df['Predicted_Quali_Time'].min():.4f}s")
print(f"Slowest predicted time: {results_df['Predicted_Quali_Time'].max():.4f}s")
print(f"Time spread: {results_df['Predicted_Quali_Time'].max() - results_df['Predicted_Quali_Time'].min():.4f}s")
print(f"Average prediction uncertainty: ±{results_df['Prediction_Std'].mean():.4f}s")

# %%
# ### 1.5.4: Qualifying prediction visualization

# Sort for better visualization
results_sorted = results_df.sort_values('Predicted_Quali_Time')

# Define team colors
team_color_map = {
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Red Bull Racing': '#0600EF',
    'Aston Martin': '#006F62',
    'Alpine': '#0093D0',
    'Williams': '#005AFF',
    'Alfa Romeo': '#900000',
    'Haas': '#FFFFFF',
    'RB': '#6692FF'
}

colors = [team_color_map.get(team, '#1e3050') for team in results_sorted['Team'].values]

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Predicted Qualifying Times with Confidence Intervals
ax = axes[0]
drivers = results_sorted['Driver'].values
times = results_sorted['Predicted_Quali_Time'].values
errors = results_sorted['Prediction_Std'].values

bars = ax.barh(range(len(drivers)), times, xerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(drivers)))
ax.set_yticklabels([f"{pos}. {driver}" for pos, driver in zip(results_sorted['Predicted_Grid_Position'], drivers)], fontsize=10)
ax.set_xlabel('Predicted Qualifying Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('2025 Las Vegas Grand Prix - Predicted Qualifying Times (with 95% Confidence Intervals)', fontsize=14, fontweight='bold')
ax.invert_yaxis()  # Best (fastest) at top
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Grid Position with Team Colors and Names
ax = axes[1]
grid_pos = results_df.sort_values('Predicted_Grid_Position')['Predicted_Grid_Position'].values
drivers_pos = results_df.sort_values('Predicted_Grid_Position')['Driver'].values
teams_pos = results_df.sort_values('Predicted_Grid_Position')['Team'].values
times_pos = results_df.sort_values('Predicted_Grid_Position')['Predicted_Quali_Time'].values

colors_pos = [team_color_map.get(team, '#1e3050') for team in teams_pos]

y_pos = range(len(drivers_pos))
ax.barh(y_pos, times_pos, color=colors_pos, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"P{pos}: {driver} ({team})" for pos, driver, team in zip(grid_pos, drivers_pos, teams_pos)], fontsize=10)
ax.set_xlabel('Predicted Qualifying Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('2025 Las Vegas Grand Prix - Predicted Grid Positions (Qualifying Order)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("✓ Visualizations created successfully")
# %%
# ### 1.5.5: Saving predictions

# Create comprehensive output
final_output = results_df[[
    'Predicted_Grid_Position',
    'Driver',
    'Team',
    'Predicted_Quali_Time',
    'Prediction_Std',
    'Lower_95%_CI',
    'Upper_95%_CI'
]].copy()

final_output.columns = [
    'Grid_Position',
    'Driver',
    'Team',
    'Qualifying_Time',
    'Uncertainty_±',
    'Confidence_Lower',
    'Confidence_Upper'
]

# Export to csv
csv_filename = 'LAS_VEGAS_2025_QUALIFYING_PREDICTIONS.csv'
final_output.to_csv(csv_filename, index=False)

# Also save raw predictions for next step (RaceTime prediction)
results_df.to_csv('LAS_VEGAS_2025_QUALIFYING_RAW.csv', index=False)
print(f"  - LAS_VEGAS_2025_QUALIFYING_RAW.csv (for RaceTime prediction)")
#%%
#################################################
#################################################
#################################################
# # STEP 2: PREDICT THE RACE

# %% 
# ## 2.1: Prepare training data

# Load qualifying predictions we just made
results_quali_2025_vegas = results_df.copy()

# Filter to only races that were completed (Status = 'Finished')
df_race_completed = df[df['Status'] == 'Finished'].copy()

print(f"Total races in dataset: {len(df)}")
print(f"Completed races (Status='Finished'): {len(df_race_completed)}")

# Create race features based on qualifying performance

# Historical statistics by driver (race performance)
driver_race_stats = df_race_completed.groupby('Driver').agg({
    'RaceTime': ['mean', 'median', 'std', 'min'],
    'RacePosition': ['mean', 'median', 'count'],
}).round(4)

driver_race_stats.columns = ['Driver_Avg_RaceTime', 'Driver_Median_RaceTime', 
                              'Driver_Std_RaceTime', 'Driver_Best_RaceTime',
                              'Driver_Avg_RacePos', 'Driver_Median_RacePos', 
                              'Driver_Race_Count']

print("Top 10 Drivers by Average Race Time:")
print(driver_race_stats['Driver_Avg_RaceTime'].nsmallest(10))

# Historical statistics by team (race performance)
team_race_stats = df_race_completed.groupby('Team').agg({
    'RaceTime': ['mean', 'median', 'std'],
    'RacePosition': ['mean', 'median']
}).round(4)

team_race_stats.columns = ['Team_Avg_RaceTime', 'Team_Median_RaceTime', 
                           'Team_Std_RaceTime', 'Team_Avg_RacePos', 'Team_Median_RacePos']

print("\nTop 5 Teams by Average Race Time:")
print(team_race_stats['Team_Avg_RaceTime'].nsmallest(5))

# Historical statistics by event (circuit characteristics)
event_race_stats = df_race_completed.groupby('Event').agg({
    'RaceTime': ['mean', 'std', 'min', 'max'],
    'RacePosition': 'mean'
}).round(4)

event_race_stats.columns = ['Event_Avg_RaceTime', 'Event_Std_RaceTime', 
                            'Event_Min_RaceTime', 'Event_Max_RaceTime', 'Event_Avg_RacePos']

# Check if Las Vegas has historical data
if 'Las Vegas Grand Prix' in event_race_stats.index:
    print(f"\nLas Vegas Grand Prix stats:")
    print(f"  Average RaceTime: {event_race_stats.loc['Las Vegas Grand Prix', 'Event_Avg_RaceTime']:.2f}s")
else:
    print(f"\nLas Vegas Grand Prix not in completed races yet")

print(f"\n✓ Historical race statistics calculated")

# %% 
# ## 2.2: Create training dataset

# Merge race statistics with main dataset
df_race_model = df_race_completed.copy()

# Merge driver race stats
df_race_model = df_race_model.merge(driver_race_stats, left_on='Driver', right_index=True, how='left')

# Merge team race stats
df_race_model = df_race_model.merge(team_race_stats, left_on='Team', right_index=True, how='left')

# Merge event race stats
df_race_model = df_race_model.merge(event_race_stats, left_on='Event', right_index=True, how='left')

# Also include qualifying statistics (which we already have)
df_race_model = df_race_model.merge(driver_quali_stats, left_on='Driver', right_index=True, how='left')
df_race_model = df_race_model.merge(team_quali_stats, left_on='Team', right_index=True, how='left')
df_race_model = df_race_model.merge(event_quali_stats, left_on='Event', right_index=True, how='left')

# Check dataset (uncomment next line)
# print(df_race_model.head())

# Remove duplicated columns
df_race_model = df_race_model.drop(columns=['Driver_Avg_Quali_y', 'Driver_Median_Quali_y', 'Driver_Std_Quali_y',
       'Driver_Best_Quali_y', 'Team_Avg_Quali_y', 'Team_Median_Quali_y',
       'Team_Std_Quali_y', 'Event_Avg_Quali_y', 'Event_Std_Quali_y',
       'Event_Min_Quali_y', 'Event_Max_Quali_y'])

# Check the columns (uncomment next line)
# print(df_race_model.columns)

# Create qualifying vs race performance indicator
df_race_model['Quali_to_Race_Performance'] = df_race_model['Best_Quali_Time'] / df_race_model['RaceTime']

print(f"Training dataset created:")
print(f"  Rows: {len(df_race_model)}")
print(f"  Columns: {len(df_race_model.columns)}")

# Check missing values in target
print(f"\nTarget variable (RaceTime) statistics:")
print(df_race_model['RaceTime'].describe())

# Check for NaN in target
nan_in_target = df_race_model['RaceTime'].isnull().sum()
print(f"\nMissing RaceTime values: {nan_in_target}")

if nan_in_target > 0:
    print(f"Removing rows with missing RaceTime...")
    df_race_model = df_race_model.dropna(subset=['RaceTime'])
    print(f"  Rows remaining: {len(df_race_model)}")

# %% 
# ## 2.3: Features selection

# Define race prediction features
race_feature_columns = [
    # Basic identifiers (will be encoded)
    'Driver',
    'Team',
    'Year',
    'Event',
    
    # Qualifying features (strong predictor of race performance)
    'Best_Quali_Time',
    'Q1_time', 'Q2_time', 'Q3_time',
    
    # Free Practice features (setup, car performance)
    'FP1_AvgLapTime', 'FP1_BestLapTime', 'FP1_AvgSpeedI1', 'FP1_AvgSpeedI2', 'FP1_AvgSpeedFL', 'FP1_AvgSpeedST',
    'FP2_AvgLapTime', 'FP2_BestLapTime', 'FP2_AvgSpeedI1', 'FP2_AvgSpeedI2', 'FP2_AvgSpeedFL', 'FP2_AvgSpeedST',
    'FP3_AvgLapTime', 'FP3_BestLapTime', 'FP3_AvgSpeedI1', 'FP3_AvgSpeedI2', 'FP3_AvgSpeedFL', 'FP3_AvgSpeedST',
    
    # Driver historical race performance
    'Driver_Avg_RaceTime',
    'Driver_Median_RaceTime',
    'Driver_Std_RaceTime',
    'Driver_Best_RaceTime',
    'Driver_Avg_RacePos',
    'Driver_Median_RacePos',
    'Driver_Race_Count',
    
    # Driver historical qualifying performance
    'Driver_Avg_Quali_x',
    'Driver_Median_Quali_x',
    'Driver_Std_Quali_x',
    'Driver_Best_Quali_x',
    
    # Team historical race performance
    'Team_Avg_RaceTime',
    'Team_Median_RaceTime',
    'Team_Std_RaceTime',
    'Team_Avg_RacePos',
    
    # Team historical qualifying performance
    'Team_Avg_Quali_x',
    'Team_Median_Quali_x',
    'Team_Std_Quali_x',
    
    # Event/Circuit characteristics
    'Event_Avg_RaceTime',
    'Event_Std_RaceTime',
    'Event_Avg_Quali_x',
    'Event_Std_Quali_x',
]

# Target
race_target = 'RaceTime'

print(f"Race prediction features: {len(race_feature_columns)}")
print(f"Target variable: {race_target}\n")

# Create clean dataset
df_race_clean = df_race_model[race_feature_columns + [race_target]].copy()

# Remove rows with missing target
df_race_clean = df_race_clean.dropna(subset=[race_target])

print(f"Rows with valid target: {len(df_race_clean)}")

# Check for missing values in features
missing_summary = df_race_clean.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(missing_summary) > 0:
    print(f"\nMissing values in features (top 10):")
    print(missing_summary)
else:
    print(f"✓ No missing values detected")

# Handle missing values

# Fill missing qualifying times (Q1, Q2, Q3) with best quali time
quali_cols = ['Q1_time', 'Q2_time', 'Q3_time']
for col in quali_cols:
    missing_count = df_race_clean[col].isnull().sum()
    if missing_count > 0:
        df_race_clean[col] = df_race_clean[col].fillna(df_race_clean['Best_Quali_Time'])
        print(f"Filled {missing_count} missing {col} values with Best_Quali_Time")

# Fill missing FP data with event averages
fp_cols = [col for col in race_feature_columns if col.startswith('FP')]
print(f"\nFilling missing Free Practice data ({len(fp_cols)} columns):")

for col in fp_cols:
    missing_count = df_race_clean[col].isnull().sum()
    if missing_count > 0:
        # Fill with event average
        event_avg = df_race_model.groupby('Event')[col].mean()
        df_race_clean[col] = df_race_clean.apply(
            lambda row: row[col] if pd.notna(row[col]) else event_avg.get(row['Event'], df_race_clean[col].mean()),
            axis=1
        )
        print(f"  {col}: filled {missing_count} values")

# Fill remaining historical stats with means
historical_cols = [col for col in race_feature_columns if 'Driver_' in col or 'Team_' in col or 'Event_' in col]
print(f"\nFilling remaining historical data ({len(historical_cols)} columns):")

for col in historical_cols:
    missing_count = df_race_clean[col].isnull().sum()
    if missing_count > 0:
        df_race_clean[col] = df_race_clean[col].fillna(df_race_clean[col].mean())
        print(f"  {col}: filled {missing_count} values")

# Final check
remaining_missing = df_race_clean.isnull().sum().sum()
print(f"\nFinal missing values: {remaining_missing}")

if remaining_missing == 0:
    print("✓ All missing values handled!")

# %%
# ### 2.4: Categorical values encoding

# %%
driver_encoder_race = LabelEncoder()
team_encoder_race = LabelEncoder()
event_encoder_race = LabelEncoder()
year_encoder_race = LabelEncoder()

df_race_clean['Driver_Encoded'] = driver_encoder_race.fit_transform(df_race_clean['Driver'])
df_race_clean['Team_Encoded'] = team_encoder_race.fit_transform(df_race_clean['Team'])
df_race_clean['Event_Encoded'] = event_encoder_race.fit_transform(df_race_clean['Event'])
df_race_clean['Year_Encoded'] = year_encoder_race.fit_transform(df_race_clean['Year'])

print(f"Driver encoding: {len(driver_encoder_race.classes_)} unique drivers")
print(f"Team encoding: {len(team_encoder_race.classes_)} unique teams")
print(f"Event encoding: {len(event_encoder_race.classes_)} unique events")
print(f"Year encoding: {len(year_encoder_race.classes_)} unique years")

# Save encoders
pickle.dump(driver_encoder_race, open('driver_encoder_race.pkl', 'wb'))
pickle.dump(team_encoder_race, open('team_encoder_race.pkl', 'wb'))
pickle.dump(event_encoder_race, open('event_encoder_race.pkl', 'wb'))
pickle.dump(year_encoder_race, open('year_encoder_race.pkl', 'wb'))

print("\n✓ Encoders saved")

# %%
# ## 2.5: Prepare features and split data
# %% 
# ### 2.5.1: prepare features

# Final feature set (replacing original categorical columns with encoded versions)
race_final_features = [col for col in race_feature_columns if col not in ['Driver', 'Team', 'Event', 'Year']] + \
                      ['Driver_Encoded', 'Team_Encoded', 'Event_Encoded', 'Year_Encoded']

X_race = df_race_clean[race_final_features]
y_race = df_race_clean[race_target]

print(f"Feature matrix shape: {X_race.shape}")
print(f"Target shape: {y_race.shape}")

# Check for any remaining NaN
nan_count = X_race.isnull().sum().sum()
print(f"Missing values in features: {nan_count}")

if nan_count > 0:
    print("⚠️ Found NaN, removing rows...")
    valid_idx = ~(X_race.isnull().any(axis=1))
    X_race = X_race[valid_idx]
    y_race = y_race[valid_idx]
    print(f"After removal: {X_race.shape}")

# %% 
# ### 2.5.2: Split data

# Split data: 80% train, 20% test
X_race_train, X_race_test, y_race_train, y_race_test = train_test_split(
    X_race, y_race, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_race_train.shape[0]}")
print(f"Test set size: {X_race_test.shape[0]}")

# Scale features
scaler_race = StandardScaler()
X_race_train_scaled = scaler_race.fit_transform(X_race_train)
X_race_test_scaled = scaler_race.transform(X_race_test)

pickle.dump(scaler_race, open('scaler_race.pkl', 'wb'))

print("\n✓ Data prepared and scaled")

# %% 
# ## 7.6: Train RaceTime models

# Model 1: Gradient Boosting (using aggressive regularization)
print("Training Gradient Boosting (Aggressive Regularization)...")

gb_race = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.01,
    min_samples_split=30,
    min_samples_leaf=20,
    subsample=0.7,
    max_features='log2',
    random_state=42,
    tol=1e-4
)

gb_race.fit(X_race_train, y_race_train)

y_race_train_pred_gb = gb_race.predict(X_race_train)
y_race_test_pred_gb = gb_race.predict(X_race_test)

gb_train_r2 = r2_score(y_race_train, y_race_train_pred_gb)
gb_test_r2 = r2_score(y_race_test, y_race_test_pred_gb)
gb_gap = gb_train_r2 - gb_test_r2
gb_test_mae = mean_absolute_error(y_race_test, y_race_test_pred_gb)
gb_test_rmse = np.sqrt(mean_squared_error(y_race_test, y_race_test_pred_gb))

print(f"  Training R²: {gb_train_r2:.4f}")
print(f"  Test R²: {gb_test_r2:.4f}")
print(f"  Overfitting Gap: {gb_gap:.4f}")
print(f"  Test MAE: {gb_test_mae:.4f}s")
print(f"  Test RMSE: {gb_test_rmse:.4f}s")

# %%
# Model 2: Random Forest (with regularization)
print("\nTraining Random Forest (Regularized)...")

rf_race = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_race.fit(X_race_train, y_race_train)

y_race_train_pred_rf = rf_race.predict(X_race_train)
y_race_test_pred_rf = rf_race.predict(X_race_test)

rf_train_r2 = r2_score(y_race_train, y_race_train_pred_rf)
rf_test_r2 = r2_score(y_race_test, y_race_test_pred_rf)
rf_gap = rf_train_r2 - rf_test_r2
rf_test_mae = mean_absolute_error(y_race_test, y_race_test_pred_rf)
rf_test_rmse = np.sqrt(mean_squared_error(y_race_test, y_race_test_pred_rf))

print(f"  Training R²: {rf_train_r2:.4f}")
print(f"  Test R²: {rf_test_r2:.4f}")
print(f"  Overfitting Gap: {rf_gap:.4f}")
print(f"  Test MAE: {rf_test_mae:.4f}s")
print(f"  Test RMSE: {rf_test_rmse:.4f}s")

# Select best model
print("\n--- Model Selection ---")
if gb_test_r2 >= rf_test_r2:
    best_race_model = gb_race
    model_name_race = "Gradient Boosting"
    print(f"✓ Selected: {model_name_race} (Test R²: {gb_test_r2:.4f})")
else:
    best_race_model = rf_race
    model_name_race = "Random Forest"
    print(f"✓ Selected: {model_name_race} (Test R²: {rf_test_r2:.4f})")


# Save model
pickle.dump(best_race_model, open('raceTime_predictor_model.pkl', 'wb'))
print("\n✓ Best model saved as 'raceTime_predictor_model.pkl'")

# %% 
# ##2.7: Cross-validation for model stability

# 5-fold cross-validation
kfold_race = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_race = cross_val_score(best_race_model, X_race, y_race, cv=kfold_race, scoring='r2', n_jobs=-1)
cv_mae_race = -cross_val_score(best_race_model, X_race, y_race, cv=kfold_race, scoring='neg_mean_absolute_error', n_jobs=-1)
cv_rmse_race = np.sqrt(-cross_val_score(best_race_model, X_race, y_race, cv=kfold_race, scoring='neg_mean_squared_error', n_jobs=-1))

print("R² Scores (5 folds):")
for i, score in enumerate(cv_r2_race, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_r2_race.mean():.4f} (+/- {cv_r2_race.std():.4f})")

print("\nMAE Scores (5 folds) in seconds:")
for i, score in enumerate(cv_mae_race, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_mae_race.mean():.4f} (+/- {cv_mae_race.std():.4f})")

print("\nRMSE Scores (5 folds) in seconds:")
for i, score in enumerate(cv_rmse_race, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean: {cv_rmse_race.mean():.4f} (+/- {cv_rmse_race.std():.4f})")

# Stability assessment
print(f"\n--- Model Stability Assessment ---")
if cv_r2_race.std() < 0.1:
    print(f"✓ Model is STABLE (CV std: {cv_r2_race.std():.4f})")
else:
    print(f"⚠️ Model shows some variance (CV std: {cv_r2_race.std():.4f})")

# %% 
# ## 7.8: Feature importance

if hasattr(best_race_model, 'feature_importances_'):
    feature_importance_race = pd.DataFrame({
        'Feature': race_final_features,
        'Importance': best_race_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 20 Most Important Features for RaceTime Prediction:\n")
    print(feature_importance_race.head(20).to_string(index=False))
    
    # Visualize top features
    plt.figure(figsize=(10, 8))
    top_15 = feature_importance_race.head(15)
    plt.barh(range(len(top_15)), top_15['Importance'], color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features for RaceTime Prediction')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# %% 
# ## 7.9: Residual analysis

# Get residuals
y_race_pred = best_race_model.predict(X_race)
residuals_race = y_race - y_race_pred

print(f"Residual Statistics:")
print(f"  Mean: {residuals_race.mean():.6f}")
print(f"  Std Dev: {residuals_race.std():.4f}")
print(f"  Min: {residuals_race.min():.4f}s")
print(f"  Max: {residuals_race.max():.4f}s")

# Visualize residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals histogram
axes[0, 0].hist(residuals_race, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Residual (seconds)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Residuals')

# 2. Residuals vs Predicted
axes[0, 1].scatter(y_race_pred, residuals_race, alpha=0.5, color='steelblue')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted RaceTime (seconds)')
axes[0, 1].set_ylabel('Residual (seconds)')
axes[0, 1].set_title('Residuals vs Predicted Values')

# 3. Actual vs Predicted
axes[1, 0].scatter(y_race, y_race_pred, alpha=0.5, color='steelblue')
perfect_line = np.array([y_race.min(), y_race.max()])
axes[1, 0].plot(perfect_line, perfect_line, 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual RaceTime (seconds)')
axes[1, 0].set_ylabel('Predicted RaceTime (seconds)')
axes[1, 0].set_title('Actual vs Predicted Values')
axes[1, 0].legend()

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals_race, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)')

plt.tight_layout()
plt.show()

print("\n✓ Residual analysis complete")

# %%
# ## 2.10: Summary of model performance

print(f"Model Type: {model_name_race}")
print(f"Training Set Size: {len(X_race_train)}")
print(f"Test Set Size: {len(X_race_test)}")
print(f"Number of Features: {len(race_final_features)}")

train_r2_final = gb_train_r2 if model_name_race == 'Gradient Boosting' else rf_train_r2
test_r2_final = gb_test_r2 if model_name_race == 'Gradient Boosting' else rf_test_r2
test_mae_final = gb_test_mae if model_name_race == 'Gradient Boosting' else rf_test_mae
test_rmse_final = gb_test_rmse if model_name_race == 'Gradient Boosting' else rf_test_rmse

print(f"\n--- Training Performance ---")
print(f"R² Score: {train_r2_final:.4f}")
print(f"MAE: {mean_absolute_error(y_race_train, best_race_model.predict(X_race_train)):.4f}s")
print(f"RMSE: {np.sqrt(mean_squared_error(y_race_train, best_race_model.predict(X_race_train))):.4f}s")

print(f"\n--- Test Performance ---")
print(f"R² Score: {test_r2_final:.4f}")
print(f"MAE: {test_mae_final:.4f}s")
print(f"RMSE: {test_rmse_final:.4f}s")

print(f"\n--- Cross-Validation Performance ---")
print(f"R² Mean: {cv_r2_race.mean():.4f} (+/- {cv_r2_race.std():.4f})")
print(f"MAE Mean: {cv_mae_race.mean():.4f} (+/- {cv_mae_race.std():.4f})s")
print(f"RMSE Mean: {cv_rmse_race.mean():.4f} (+/- {cv_rmse_race.std():.4f})s")

print(f"\n--- Overfitting Assessment ---")
overfitting_gap_final = train_r2_final - test_r2_final
print(f"Gap: {overfitting_gap_final:.4f}")
if overfitting_gap_final < 0.1:
    print("✓ Minimal overfitting")
elif overfitting_gap_final < 0.15:
    print("⚠️ Moderate overfitting (acceptable)")
else:
    print("✗ Significant overfitting")

print(f"\n✓ RaceTime predictor is ready for 2025 Las Vegas predictions!")

# %%
# ## 2.11: Prepare data for prediction

# Load the predicted qualifying times from Phase 6
results_quali_2025_vegas = results_df.copy()

print(f"Loaded predicted qualifying times for {len(results_quali_2025_vegas)} drivers\n")

# Create prediction dataset for race time prediction
prediction_data_race = []

for driver in results_quali_2025_vegas['Driver'].values:
    # Get driver info
    driver_data = df[df['Driver'] == driver].sort_values('Year', ascending=False).iloc[0]
    team = driver_data['Team']
    
    # Get most recent FP data (from 2025 races or 2024 if not available)
    driver_2025_recent = df[(df['Driver'] == driver) & (df['Year'] == 2025)].sort_values('Event', ascending=False)
    
    if len(driver_2025_recent) > 0:
        fp_data = driver_2025_recent.iloc[0]
    else:
        # Fallback to 2024 Las Vegas FP data
        driver_2024_vegas = df[(df['Driver'] == driver) & 
                               (df['Event'] == 'Las Vegas Grand Prix') & 
                               (df['Year'] == 2024)]
        if len(driver_2024_vegas) > 0:
            fp_data = driver_2024_vegas.iloc[0]
        else:
            # Fallback to average FP data for all 2024 races
            driver_2024_all = df[(df['Driver'] == driver) & (df['Year'] == 2024)]
            if len(driver_2024_all) > 0:
                fp_data = driver_2024_all.mean()
            else:
                fp_data = df[(df['Driver'] == driver) & (df['Year'] == 2025)].mean()
    
    # Get qualifying times for this driver
    predicted_quali_info = results_quali_2025_vegas[results_quali_2025_vegas['Driver'] == driver].iloc[0]
    predicted_quali_time = predicted_quali_info['Predicted_Quali_Time']
    
    # Get historical race stats
    driver_avg_race_time = driver_race_stats.loc[driver, 'Driver_Avg_RaceTime'] if driver in driver_race_stats.index else np.nan
    driver_median_race_time = driver_race_stats.loc[driver, 'Driver_Median_RaceTime'] if driver in driver_race_stats.index else np.nan
    driver_std_race_time = driver_race_stats.loc[driver, 'Driver_Std_RaceTime'] if driver in driver_race_stats.index else np.nan
    driver_best_race_time = driver_race_stats.loc[driver, 'Driver_Best_RaceTime'] if driver in driver_race_stats.index else np.nan
    driver_avg_race_pos = driver_race_stats.loc[driver, 'Driver_Avg_RacePos'] if driver in driver_race_stats.index else np.nan
    driver_median_race_pos = driver_race_stats.loc[driver, 'Driver_Median_RacePos'] if driver in driver_race_stats.index else np.nan
    driver_race_count = driver_race_stats.loc[driver, 'Driver_Race_Count'] if driver in driver_race_stats.index else np.nan
    
    # Get historical qualifying stats
    driver_avg_quali = driver_quali_stats.loc[driver, 'Driver_Avg_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_median_quali = driver_quali_stats.loc[driver, 'Driver_Median_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_std_quali = driver_quali_stats.loc[driver, 'Driver_Std_Quali'] if driver in driver_quali_stats.index else np.nan
    driver_best_quali = driver_quali_stats.loc[driver, 'Driver_Best_Quali'] if driver in driver_quali_stats.index else np.nan
    
    # Get team stats
    team_avg_race_time = team_race_stats.loc[team, 'Team_Avg_RaceTime'] if team in team_race_stats.index else np.nan
    team_median_race_time = team_race_stats.loc[team, 'Team_Median_RaceTime'] if team in team_race_stats.index else np.nan
    team_std_race_time = team_race_stats.loc[team, 'Team_Std_RaceTime'] if team in team_race_stats.index else np.nan
    team_avg_race_pos = team_race_stats.loc[team, 'Team_Avg_RacePos'] if team in team_race_stats.index else np.nan
    
    team_avg_quali = team_quali_stats.loc[team, 'Team_Avg_Quali'] if team in team_quali_stats.index else np.nan
    team_median_quali = team_quali_stats.loc[team, 'Team_Median_Quali'] if team in team_quali_stats.index else np.nan
    team_std_quali = team_quali_stats.loc[team, 'Team_Std_Quali'] if team in team_quali_stats.index else np.nan
    
    # Get event stats
    event_avg_race_time = event_race_stats.loc['Las Vegas Grand Prix', 'Event_Avg_RaceTime'] if 'Las Vegas Grand Prix' in event_race_stats.index else np.nan
    event_std_race_time = event_race_stats.loc['Las Vegas Grand Prix', 'Event_Std_RaceTime'] if 'Las Vegas Grand Prix' in event_race_stats.index else np.nan
    event_avg_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Avg_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    event_std_quali = event_quali_stats.loc['Las Vegas Grand Prix', 'Event_Std_Quali'] if 'Las Vegas Grand Prix' in event_quali_stats.index else np.nan
    
    # Build row
    row = {
        'Driver': driver,
        'Team': team,
        'Year': 2025,
        'Event': 'Las Vegas Grand Prix',
        'Best_Quali_Time': predicted_quali_time,
        'Q1_time': predicted_quali_time,  # Using predicted qualifying as proxy
        'Q2_time': predicted_quali_time,
        'Q3_time': predicted_quali_time,
        'FP1_AvgLapTime': fp_data['FP1_AvgLapTime'] if 'FP1_AvgLapTime' in fp_data.index else fp_data.get('FP1_AvgLapTime', np.nan),
        'FP1_BestLapTime': fp_data['FP1_BestLapTime'] if 'FP1_BestLapTime' in fp_data.index else fp_data.get('FP1_BestLapTime', np.nan),
        'FP1_AvgSpeedI1': fp_data['FP1_AvgSpeedI1'] if 'FP1_AvgSpeedI1' in fp_data.index else fp_data.get('FP1_AvgSpeedI1', np.nan),
        'FP1_AvgSpeedI2': fp_data['FP1_AvgSpeedI2'] if 'FP1_AvgSpeedI2' in fp_data.index else fp_data.get('FP1_AvgSpeedI2', np.nan),
        'FP1_AvgSpeedFL': fp_data['FP1_AvgSpeedFL'] if 'FP1_AvgSpeedFL' in fp_data.index else fp_data.get('FP1_AvgSpeedFL', np.nan),
        'FP1_AvgSpeedST': fp_data['FP1_AvgSpeedST'] if 'FP1_AvgSpeedST' in fp_data.index else fp_data.get('FP1_AvgSpeedST', np.nan),
        'FP2_AvgLapTime': fp_data['FP2_AvgLapTime'] if 'FP2_AvgLapTime' in fp_data.index else fp_data.get('FP2_AvgLapTime', np.nan),
        'FP2_BestLapTime': fp_data['FP2_BestLapTime'] if 'FP2_BestLapTime' in fp_data.index else fp_data.get('FP2_BestLapTime', np.nan),
        'FP2_AvgSpeedI1': fp_data['FP2_AvgSpeedI1'] if 'FP2_AvgSpeedI1' in fp_data.index else fp_data.get('FP2_AvgSpeedI1', np.nan),
        'FP2_AvgSpeedI2': fp_data['FP2_AvgSpeedI2'] if 'FP2_AvgSpeedI2' in fp_data.index else fp_data.get('FP2_AvgSpeedI2', np.nan),
        'FP2_AvgSpeedFL': fp_data['FP2_AvgSpeedFL'] if 'FP2_AvgSpeedFL' in fp_data.index else fp_data.get('FP2_AvgSpeedFL', np.nan),
        'FP2_AvgSpeedST': fp_data['FP2_AvgSpeedST'] if 'FP2_AvgSpeedST' in fp_data.index else fp_data.get('FP2_AvgSpeedST', np.nan),
        'FP3_AvgLapTime': fp_data['FP3_AvgLapTime'] if 'FP3_AvgLapTime' in fp_data.index else fp_data.get('FP3_AvgLapTime', np.nan),
        'FP3_BestLapTime': fp_data['FP3_BestLapTime'] if 'FP3_BestLapTime' in fp_data.index else fp_data.get('FP3_BestLapTime', np.nan),
        'FP3_AvgSpeedI1': fp_data['FP3_AvgSpeedI1'] if 'FP3_AvgSpeedI1' in fp_data.index else fp_data.get('FP3_AvgSpeedI1', np.nan),
        'FP3_AvgSpeedI2': fp_data['FP3_AvgSpeedI2'] if 'FP3_AvgSpeedI2' in fp_data.index else fp_data.get('FP3_AvgSpeedI2', np.nan),
        'FP3_AvgSpeedFL': fp_data['FP3_AvgSpeedFL'] if 'FP3_AvgSpeedFL' in fp_data.index else fp_data.get('FP3_AvgSpeedFL', np.nan),
        'FP3_AvgSpeedST': fp_data['FP3_AvgSpeedST'] if 'FP3_AvgSpeedST' in fp_data.index else fp_data.get('FP3_AvgSpeedST', np.nan),
        'Driver_Avg_RaceTime': driver_avg_race_time,
        'Driver_Median_RaceTime': driver_median_race_time,
        'Driver_Std_RaceTime': driver_std_race_time,
        'Driver_Best_RaceTime': driver_best_race_time,
        'Driver_Avg_RacePos': driver_avg_race_pos,
        'Driver_Median_RacePos': driver_median_race_pos,
        'Driver_Race_Count': driver_race_count,
        'Driver_Avg_Quali_x': driver_avg_quali,
        'Driver_Median_Quali_x': driver_median_quali,
        'Driver_Std_Quali_x': driver_std_quali,
        'Driver_Best_Quali_x': driver_best_quali,
        'Team_Avg_RaceTime': team_avg_race_time,
        'Team_Median_RaceTime': team_median_race_time,
        'Team_Std_RaceTime': team_std_race_time,
        'Team_Avg_RacePos': team_avg_race_pos,
        'Team_Avg_Quali_x': team_avg_quali,
        'Team_Median_Quali_x': team_median_quali,
        'Team_Std_Quali_x': team_std_quali,
        'Event_Avg_RaceTime': event_avg_race_time,
        'Event_Std_RaceTime': event_std_race_time,
        'Event_Avg_Quali_x': event_avg_quali,
        'Event_Std_Quali_x': event_std_quali
    }
    
    prediction_data_race.append(row)

# Create DataFrame
df_predict_race = pd.DataFrame(prediction_data_race)

print("Las Vegas 2025 Race Prediction Dataset Created:")
print(f"  Drivers: {len(df_predict_race)}")
print(f"  Features: {len(df_predict_race.columns)}")
print(f"\nSample data:")
df_predict_race[['Driver', 'Team', 'Best_Quali_Time', 'Driver_Avg_RaceTime', 'Team_Avg_RaceTime']].head(10)

# Check for missing values
missing_in_predict = df_predict_race.isnull().sum()
print(f"\nMissing values in prediction dataset:")
missing_summary = missing_in_predict[missing_in_predict > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    print(missing_summary.head(10))
else:
    print("  None detected yet")

# Load the encoders we created during training
driver_encoder_race = pickle.load(open('driver_encoder_race.pkl', 'rb'))
team_encoder_race = pickle.load(open('team_encoder_race.pkl', 'rb'))
event_encoder_race = pickle.load(open('event_encoder_race.pkl', 'rb'))
year_encoder_race = pickle.load(open('year_encoder_race.pkl', 'rb'))

# Encode the prediction data
df_predict_race['Driver_Encoded'] = driver_encoder_race.transform(df_predict_race['Driver'])
df_predict_race['Team_Encoded'] = team_encoder_race.transform(df_predict_race['Team'])
df_predict_race['Event_Encoded'] = event_encoder_race.transform(df_predict_race['Event'])
df_predict_race['Year_Encoded'] = year_encoder_race.transform(df_predict_race['Year'])

print(f"✓ Encoding complete")
print(f"  Driver encoded")
print(f"  Team encoded")
print(f"  Event encoded")
print(f"  Year encoded")

# Build feature list for prediction (WITHOUT GridPosition, identifiers will be replaced with encoded versions)
race_prediction_features = [col for col in race_final_features if col not in ['Driver', 'Team', 'Event', 'Year']]

print(f"\nFeatures for prediction: {len(race_prediction_features)}")
print(f"  (Identifiers: Driver, Team, Event, Year → Encoded versions)")

# Extract features for prediction
X_predict_race = df_predict_race[race_prediction_features]

print(f"\n✓ Prediction feature matrix created: {X_predict_race.shape}")
print(f"  Rows: {X_predict_race.shape[0]} drivers")
print(f"  Columns: {X_predict_race.shape[1]} features")

# Check for any remaining NaN
nan_count = X_predict_race.isnull().sum().sum()
if nan_count > 0:
    print(f"\n⚠️ Warning: {nan_count} NaN values found")
    print("Filling with column means...")
    X_predict_race = X_predict_race.fillna(X_predict_race.mean())
else:
    print(f"\n✓ No missing values in features")

# %%
# ## 2.12: Handling missing values

# Fill missing FP data with 2024 Las Vegas event averages
fp_columns = [col for col in df_predict_race.columns if col.startswith('FP')]

print("Filling missing Free Practice data:")
for col in fp_columns:
    # Get average from 2024 Las Vegas completed races
    event_avg = df[(df['Event'] == 'Las Vegas Grand Prix') & (df['Year'] == 2024) & (df['Status'] == 'Finished')][col].mean()
    missing_before = df_predict_race[col].isnull().sum()
    df_predict_race[col] = df_predict_race[col].fillna(event_avg)
    missing_after = df_predict_race[col].isnull().sum()
    if missing_before > 0:
        print(f"  {col}: filled {missing_before} values with 2024 LV avg ({event_avg:.4f})")

# Fill missing historical stats with global means
historical_cols = [col for col in df_predict_race.columns if 'Driver_' in col or 'Team_' in col or 'Event_' in col]

print(f"\nFilling missing historical stats ({len(historical_cols)} columns):")
for col in historical_cols:
    missing_count = df_predict_race[col].isnull().sum()
    if missing_count > 0:
        fill_value = df[col].mean()
        df_predict_race[col] = df_predict_race[col].fillna(fill_value)
        print(f"  {col}: filled {missing_count} values with global mean ({fill_value:.4f})")

# Final check
print(f"\n--- Final Missing Value Check ---")
remaining_missing = df_predict_race.isnull().sum().sum()
print(f"Total missing values: {remaining_missing}")

if remaining_missing == 0:
    print("✓ All missing values handled!")
else:
    print(f"⚠️ Warning: {remaining_missing} missing values remain")
    print("Filling with column means...")
    df_predict_race = df_predict_race.fillna(df_predict_race.mean())

# %% 
# ## 2.13: Encoding categorical variables for race prediction

# Load the encoders we created during training
driver__encoder_race = pickle.load(open('driver_encoder_race.pkl', 'rb'))
team_encoder_race = pickle.load(open('team_encoder_race.pkl', 'rb'))
event_encoder_race = pickle.load(open('event_encoder_race.pkl', 'rb'))
year_encoder_race = pickle.load(open('year_encoder_race.pkl', 'rb'))

# Encode the prediction data
df_predict_race['Driver_Encoded'] = driver__encoder_race.transform(df_predict_race['Driver'])
df_predict_race['Team_Encoded'] = team_encoder_race.transform(df_predict_race['Team'])
df_predict_race['Event_Encoded'] = event_encoder_race.transform(df_predict_race['Event'])
df_predict_race['Year_Encoded'] = year_encoder_race.transform(df_predict_race['Year'])

print(f"✓ Encoding complete")
print(f"  Driver encoded")
print(f"  Team encoded")
print(f"  Event encoded")
print(f"  Year encoded")

# Build feature list for prediction
race_prediction_features = [col for col in race_final_features if col not in ['Driver', 'Team', 'Event', 'Year']]

print(f"\nFeatures for prediction: {len(race_prediction_features)}")
print(f"  (Identifiers: Driver, Team, Event, Year → Encoded versions)")

# Verify all features exist in prediction data
print(f"\nVerifying feature availability:")
missing_features = [col for col in race_prediction_features if col not in df_predict_race.columns]

if missing_features:
    print(f"⚠️ Missing features: {missing_features}")
    raise KeyError(f"Missing features: {missing_features}")
else:
    print(f"✓ All {len(race_prediction_features)} features available in prediction data")

# Extract features for prediction
X_predict_race = df_predict_race[race_prediction_features]

print(f"\n✓ Prediction feature matrix created: {X_predict_race.shape}")
print(f"  Rows: {X_predict_race.shape[0]} drivers")
print(f"  Columns: {X_predict_race.shape[1]} features")

# Check for any remaining NaN
nan_count = X_predict_race.isnull().sum().sum()
if nan_count > 0:
    print(f"\n⚠️ Warning: {nan_count} NaN values found")
    print("Filling with column means...")
    X_predict_race = X_predict_race.fillna(X_predict_race.mean())
else:
    print(f"\n✓ No missing values in features")

# %% 
# ## 2.14: Scale features

# Load the fitted scaler from training
scaler_race = pickle.load(open('scaler_race.pkl', 'rb'))

# Scale features using the fitted scaler
X_predict_race_scaled = scaler_race.transform(X_predict_race)

print(f"✓ Features scaled using training scaler")
print(f"  Scaled shape: {X_predict_race_scaled.shape}")

# %% 
# ## 2.15: RACE PREDICTION

# Load the best race time model
best_race_model = pickle.load(open('raceTime_predictor_model.pkl', 'rb'))

print(f"\nLoaded RaceTime model")
print(f"Making predictions...\n")

# Make predictions (use unscaled data for tree-based models)
predicted_race_times = best_race_model.predict(X_predict_race)

# Get prediction uncertainty (using residual std from training)
residuals_train_race = y_race_train - best_race_model.predict(X_race_train)
pred_std_race = np.ones_like(predicted_race_times) * residuals_train_race.std()

# Create results DataFrame
race_results_df = df_predict_race[['Driver', 'Team']].copy()
race_results_df['Predicted_Race_Time'] = predicted_race_times
race_results_df['Prediction_Std'] = pred_std_race
race_results_df['Lower_95%_CI'] = predicted_race_times - 1.96 * pred_std_race
race_results_df['Upper_95%_CI'] = predicted_race_times + 1.96 * pred_std_race

# Sort by predicted race time (fastest = winner)
race_results_df = race_results_df.sort_values('Predicted_Race_Time').reset_index(drop=True)
race_results_df['Predicted_Race_Position'] = range(1, len(race_results_df) + 1)

print(f"✓ Predictions generated for {len(race_results_df)} drivers")
print(f"\nPREDICTED RACE RESULTS FOR 2025 LAS VEGAS GRAND PRIX:\n")
print(race_results_df[['Predicted_Race_Position', 'Driver', 'Team', 'Predicted_Race_Time', 'Prediction_Std']].to_string(index=False))

# Race statistics
print(f"\n--- Prediction Statistics ---")
print(f"Predicted winner: {race_results_df.iloc[0]['Driver']} ({race_results_df.iloc[0]['Team']})")
print(f"Winner predicted time: {race_results_df.iloc[0]['Predicted_Race_Time']:.2f}s")
print(f"Last place predicted time: {race_results_df.iloc[-1]['Predicted_Race_Time']:.2f}s")
print(f"Time spread: {race_results_df.iloc[-1]['Predicted_Race_Time'] - race_results_df.iloc[0]['Predicted_Race_Time']:.2f}s")
print(f"Average prediction uncertainty: ±{race_results_df['Prediction_Std'].mean():.2f}s")

# %% 
# ## 2.16: Compare quali prediction to race prediction

print("\n" + "="*60)
print("GRID VS FINISH POSITION ANALYSIS")
print("="*60 + "\n")

# Merge qualifying predictions with race predictions
comparison_grid_finish = results_quali_2025_vegas[['Driver', 'Team', 'Predicted_Quali_Time', 'Predicted_Grid_Position']].copy()
comparison_grid_finish = comparison_grid_finish.merge(
    race_results_df[['Driver', 'Predicted_Race_Time', 'Predicted_Race_Position']],
    on='Driver',
    how='left'
)

# Calculate position change
comparison_grid_finish['Position_Change'] = comparison_grid_finish['Predicted_Race_Position'] - comparison_grid_finish['Predicted_Grid_Position']

# Sort by predicted finishing position
comparison_grid_finish = comparison_grid_finish.sort_values('Predicted_Race_Position')

print("Predicted Grid to Finish Analysis:\n")
print(comparison_grid_finish[[
    'Predicted_Grid_Position', 'Driver', 'Team', 
    'Predicted_Race_Position', 'Position_Change'
]].to_string(index=False))

# Analyze position changes
print(f"\n--- Position Change Analysis ---")
improved = comparison_grid_finish[comparison_grid_finish['Position_Change'] < -1]
declined = comparison_grid_finish[comparison_grid_finish['Position_Change'] > 1]
maintained = comparison_grid_finish[(comparison_grid_finish['Position_Change'] >= -1) & (comparison_grid_finish['Position_Change'] <= 1)]

print(f"Drivers predicted to IMPROVE: {len(improved)}")
if len(improved) > 0:
    for _, row in improved.head(5).iterrows():
        print(f"  {row['Driver']:3s}: P{int(row['Predicted_Grid_Position'])} → P{int(row['Predicted_Race_Position'])} ({row['Position_Change']:.0f})")

print(f"\nDrivers predicted to DECLINE: {len(declined)}")
if len(declined) > 0:
    for _, row in declined.head(5).iterrows():
        print(f"  {row['Driver']:3s}: P{int(row['Predicted_Grid_Position'])} → P{int(row['Predicted_Race_Position'])} ({row['Position_Change']:+.0f})")

print(f"\nDrivers predicted to MAINTAIN position: {len(maintained)}")

# Average position change
avg_change = comparison_grid_finish['Position_Change'].mean()
print(f"\nAverage position change: {avg_change:+.2f} positions")

# %% 
# ## 2.17: Visualize race prediction

print("\n" + "="*60)
print("VISUALIZING RACE PREDICTIONS")
print("="*60 + "\n")

# Define team colors
team_color_map = {
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Red Bull Racing': '#0600EF',
    'Aston Martin': '#006F62',
    'Alpine': '#0093D0',
    'Williams': '#005AFF',
    'Alfa Romeo': '#900000',
    'Haas': '#FFFFFF',
    'RB': '#6692FF'
}

colors_race = [team_color_map.get(team, '#1e3050') for team in race_results_df['Team'].values]

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Predicted Race Times with Confidence Intervals
ax = axes[0]
drivers_race = race_results_df['Driver'].values
times_race = race_results_df['Predicted_Race_Time'].values
errors_race = race_results_df['Prediction_Std'].values

bars = ax.barh(range(len(drivers_race)), times_race, xerr=errors_race, capsize=5, color=colors_race, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(drivers_race)))
ax.set_yticklabels([f"P{pos}. {driver}" for pos, driver in zip(race_results_df['Predicted_Race_Position'], drivers_race)], fontsize=10)
ax.set_xlabel('Predicted Race Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('2025 Las Vegas Grand Prix - Predicted Race Times (with 95% Confidence Intervals)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Final Race Classification with Team Colors
ax = axes[1]
positions = race_results_df['Predicted_Race_Position'].values
drivers_final = race_results_df['Driver'].values
teams_final = race_results_df['Team'].values
times_final = race_results_df['Predicted_Race_Time'].values

colors_final = [team_color_map.get(team, '#1e3050') for team in teams_final]

y_pos = range(len(drivers_final))
ax.barh(y_pos, times_final, color=colors_final, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"P{pos}: {driver} ({team})" for pos, driver, team in zip(positions, drivers_final, teams_final)], fontsize=10)
ax.set_xlabel('Predicted Race Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('2025 Las Vegas Grand Prix - Final Predicted Classification', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("✓ Visualizations created successfully")

# %%
# ## 2.18: Export final quali and race prediction

# Create comprehensive output combining qualifying and race predictions
final_race_output = comparison_grid_finish[[
    'Predicted_Grid_Position',
    'Driver',
    'Team',
    'Predicted_Quali_Time',
    'Predicted_Race_Position',
    'Predicted_Race_Time',
    'Position_Change'
]].copy()

final_race_output.columns = [
    'Grid_Position',
    'Driver',
    'Team',
    'Qualifying_Time',
    'Race_Position',
    'Race_Time',
    'Position_Change'
]

print("PREDICTED RESULTS FOR 2025 LAS VEGAS GRAND PRIX:\n")
print(final_race_output.to_string(index=False))

# Export to multiple formats
csv_filename_race = 'LAS_VEGAS_2025_RACE_PREDICTIONS.csv'
final_race_output.to_csv(csv_filename_race, index=False)

print(f"\n✓ Predictions exported:")
print(f"  - {csv_filename_race}")

# Save raw predictions
race_results_df.to_csv('LAS_VEGAS_2025_RACE_RAW.csv', index=False)
print(f"  - LAS_VEGAS_2025_RACE_RAW.csv (raw predictions)")