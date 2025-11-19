# f1_las_vegas_2025_prediction
This is a machine learning project to predict the outcomes of the qualifying session and the race of the Las Vegas Grand prix 2025

Ariane BOULET, Sereine TAWAMBA
# 🏎️ Predicting the 2025 Las Vegas Grand Prix with a Machine Learning model

## Project Overview

This project utilizes advanced machine learning techniques to forecast the outcomes of both the Qualifying session and the Race of the Formula 1 Las Vegas Grand Prix in 2025. The goal is to provide data-driven predictions for key F1 stakeholders, teams, and fans.

### 1. Business Use Case

Predicting F1 Qualifying and Race outcomes provides valuable, actionable insights for various stakeholders within the motorsport ecosystem:

F1 Teams: Predictions can aid in race strategy planning, competitor performance assessment, and optimal car setup configuration before track sessions begin.

Betting & Financial Markets: Accurate forecasting provides an edge for predicting driver and constructor performance for betting and performance-related financial derivatives.

Fans & Media: The model offers data-backed discussion points and enhances the viewing experience by setting data-driven expectations for the weekend's events.

### 2. Dataset Description

The core of this predictive model is built upon a comprehensive historical dataset of Formula 1 races, aggregated from various sources, primarily leveraging the FastF1 library for detailed session data.

- Primary Data Source: 0_f1_master_database.csv (and its updated version), collected from the fastf1 API.

- Data Scope: The dataset contains detailed lap, driver, constructor, and event metrics for historical Grand Prix events, spanning multiple seasons.

- Key Variables:

  - Features: Include historical race results, lap times, weather conditions, driver and constructor metadata, and track-specific characteristics.

  - Target Variables: The model focuses on predicting two key time metrics:

    - QualiTime: The fastest lap time achieved by a driver in the qualifying session.

    - RaceTime: The total elapsed time for a driver to complete the race.

### 3. Initial Model Attempt: Random Forest (first for Qualifying Time, then for Race Time)

The first model established was a Random Forest Regressor, intended as the rapid baseline for predicting Best_Quali_Time.

- Model Type: RandomForestRegressor (Initial Baseline)

- Target Variable: Qualifying Time (Best_Quali_Time) and RaceTime

- Key Features Used: historical Free Practice (FP1, FP2, FP3) metrics, Driver/Team/Event historical Quali stats.

- Pre-processing:
  - Missing Free Practice data filled with event-specific means.
  - Categorical features encoded with LabelEncoder.
  - Numerical features scaled using StandardScaler.

- Metrics Obtained (Test Set): The model showed a large gap between training R² (0.9632) and testing R² (0.8409), indicating significant overfitting.
  - QualiTime Metric (MAE): 0.9996 seconds
  - QualiTime Metric (RMSE): 1.3653 seconds

### 4. Iteration: Stabilized Gradient Boosting & expansion to Race Prediction

This iteration focused on addressing the overfitting issue in the Qualifying model and expanding the scope to the full race prediction.

#### What was changed & Why

- Model Switch & Regularization: Switched to GradientBoostingRegressor. A highly aggressive regularization strategy (max_depth=2, learning_rate=0.01, n_estimators=50) was immediately applied to minimize the Train/Test R² gap.

- Impact on Qualifying Metrics: This change stabilized the model, improving generalization but increasing error substantially.

- New QualiTime MAE: 6.4620 seconds

- New QualiTime RMSE: 7.7984 seconds.

- Qualifying Validation: 5-Fold Cross-Validation confirmed stability with a Standard Deviation of R² scores of 0.0098 (low variance).

- Scope Expansion: The stable Gradient Boosting model configuration was then compared again to Random Forest to predict the total race duration (RaceTime). In the end, Random Forest was the best model in this case.

- Race Prediction Features: The feature set was expanded to include historical Driver/Team RaceTime metrics and, crucially, the predicted QualiTime from the first stage.

- RaceTime Metrics (Test Set): The model achieved average results on the race data, with a large margin error (for Formula 1).

  - RaceTime Metric (Test R²): 0.6487
  - RaceTime Metric (MAE): 237.9089 seconds (Approx. 4 minutes of error)
  - RaceTime Metric (RMSE): 370.5408 seconds (Approx. 6 minutes of error)
 
### 5. Conclusion and Constraints

The project successfully developed a stable Gradient Boosting model pipeline for Qualifying prediction and a Random Forest model for Race Time prediction. Although, we achieved a lower R² score and acceptable MAE/RMSE figures for Qualifying Times through aggressive regularization, we managed to avoid having an overfitting model. The results for the Race Time prediction are also contrasted, although the R² score is slightly better. Overall, these predictions are just a shallow overview of the teams' and drivers' performances and won't have a real business impact for F1 stakeholders.

Indeed, predicting Formula 1 outcomes remains inherently challenging due to major unpredictable constraints:

- Unforeseen Events: The models cannot account for non-deterministic events such as Safety Cars, Virtual Safety Cars, red flags, or unexpected mechanical failures, which dramatically alter race time and position.

- Track-Specific Variance: The Las Vegas street circuit is relatively new, limiting the amount of historical data and potentially increasing prediction uncertainty compared to long-established tracks.

- External Factors: Factors like small rule changes, rapid in-season development, and minor driver errors are difficult to quantify and integrate into a feature set, representing a fundamental limit to prediction accuracy.
