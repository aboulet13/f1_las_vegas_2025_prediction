import fastf1
import pandas as pd
import numpy as np

# save fastf1 data in a 'cache' folder
fastf1.Cache.enable_cache('fastf1_cache')

# %%
# STEP 1: EXTRACT QUALIFYING DATA FROM 2024 AND 2025 F1 SEASONS

def get_qualifying_features(year, event):
    quali = fastf1.get_session(year, event, 'Q')
    quali.load()
    
    quali_results = quali.results.copy()
    
    # Convert lap times to seconds (they're timedelta objects)
    def time_to_seconds(time_val):
        if pd.isna(time_val):
            return np.nan
        return time_val.total_seconds()
    
    quali_features = pd.DataFrame({
        'DriverNumber': quali_results['DriverNumber'],
        'Driver': quali_results['Abbreviation'],
        'Team': quali_results['TeamName'],
        'GridPosition': quali_results['Position'],
        'Q1_time': quali_results['Q1'].apply(time_to_seconds),
        'Q2_time': quali_results['Q2'].apply(time_to_seconds),
        'Q3_time': quali_results['Q3'].apply(time_to_seconds),
    })
    # best quali time overall
    quali_features['Best_Quali_Time'] = quali_features[['Q1_time', 'Q2_time', 'Q3_time']].min(axis=1)
    
    return quali_features

# TEST quali function (uncomment next 2  lines)
# brazil_2025_quali = get_qualifying_features(2025, 'Brazil')
# print(brazil_2025_quali)

# %%
# STEP 2: EXTRACT FREE PRACTICE DATA (FP1, FP2, FP3) FROM 2024 AND 2025 F1 SEASONS
##    NB: regular race weekend has FP1, FP2, FP3, quali, race // sprint weekend has FP1, sprint quali, sprint race, quali, race.

def get_practice_features(year, event):
    practice_data = []
    
    # check what sessions are available (regular weekend OR sprint weekend)
    event_schedule = fastf1.get_event_schedule(year)
    event_info = event_schedule[event_schedule['EventName'].str.contains(event, case=False, na=False)]
    
    # Try different practice session combinations
    # for sprint weekends: only FP1
    # for regular weekends: FP1, FP2, FP3
    possible_sessions = ['FP1', 'FP2', 'FP3']
    
    for session_name in possible_sessions:
        try:
            print(f"Loading {session_name}")
            fp = fastf1.get_session(year, event, session_name)
            fp.load()
            
            # get all laps from the session
            laps = fp.laps
            
            # check if laps is a DataFrame
            if not isinstance(laps, pd.DataFrame):
                print(f"{session_name}: Laps data is not a DataFrame, skipping")
                continue
            
            if laps.empty:
                print(f"{session_name}: No lap data available")
                continue
            
            # Filter out invalid laps (pit stops, incomplete laps, etc.)
            valid_laps = laps[(laps['LapTime'].notna())].copy()
            
            if valid_laps.empty:
                print(f"{session_name}: No valid laps found")
                continue
            
            print(f"{session_name}: {len(valid_laps)} valid laps")
            
            # Calculate driver statistics
            driver_stats = valid_laps.groupby('Driver').agg({
                'LapTime': ['mean', 'std', 'min', 'count'],
                'SpeedI1': 'mean',  # speedtrap sector 1
                'SpeedI2': 'mean',  # speedtrap sector 2
                'SpeedFL': 'mean',  # speedtrap at finish line
                'SpeedST': 'mean'   # speedtrap on longest straight
            }).reset_index()
            
            # Flatten column names
            driver_stats.columns = ['Driver', 
                                   f'{session_name}_AvgLapTime',
                                   f'{session_name}_StdLapTime',
                                   f'{session_name}_BestLapTime',
                                   f'{session_name}_LapCount',
                                   f'{session_name}_AvgSpeedI1',
                                   f'{session_name}_AvgSpeedI2',
                                   f'{session_name}_AvgSpeedFL',
                                   f'{session_name}_AvgSpeedST']
            
            practice_data.append(driver_stats)
            print(f"{session_name}: success")
            
        except Exception as e:
            print(f"{session_name}: Could not load - {e}")
            continue
    
    # Merge all practice sessions
    if practice_data:
        combined = practice_data[0]
        for df in practice_data[1:]:
            combined = combined.merge(df, on='Driver', how='outer')
        
        # Convert timedelta to seconds
        time_cols = [col for col in combined.columns if 'LapTime' in col]
        for col in time_cols:
            combined[col] = combined[col].apply(
                lambda x: x.total_seconds() if pd.notna(x) else np.nan
            )
        
        print(f"shape: {combined.shape}")
        return combined
    else:
        print("No practice data available")
        return pd.DataFrame()
    
# TEST free practice function (uncomment next 2  lines)
# brazil_2025_practice = get_practice_features(2025, 'Brazil')
# print(brazil_2025_practice)

# %%
# STEP 3: EXTRACT RACE DATA

def get_race_results(year, event):
    race = fastf1.get_session(year, event, 'R')
    race.load()
    
    race_results = race.results.copy()
    
    results_df = pd.DataFrame({
        'Driver': race_results['Abbreviation'],
        'DriverNumber': race_results['DriverNumber'],
        'Position': race_results['Position'],
        'Points': race_results['Points'],
        'Status': race_results['Status'],  # Finished or DNF.
        'Time': race_results['Time'].apply(
            lambda x: x.total_seconds() if pd.notna(x) else np.nan
        )
    })
    
    return results_df

# TEST race function (uncomment next 2  lines)
# brazil_2025_race = get_race_results(2025, 'Brazil')
# print(brazil_2025_race)

# %%
# STEP 4: COMBINE ALL FEATURES FOR ONE RACE

def create_race_dataset(year, event):

    print(f"Collecting data for {event} {year}:")
    
    # get all features
    quali_features = get_qualifying_features(year, event)
    practice_features = get_practice_features(year, event)
    race_results = get_race_results(year, event)
    
    # merge everything
    # 1. Quali features
    dataset = quali_features.copy()
    
    # 2. Practice features
    if not practice_features.empty:
        dataset = dataset.merge(practice_features, on='Driver', how='left')
    
    # 3. Race results (target)
    dataset = dataset.merge(
        race_results[['Driver', 'Position', 'Time', 'Status']], 
        on='Driver', 
        how='left',
        suffixes=('', '_Race')    #to distinguish from quali
    )
    
    # Rename for clarity
    dataset.rename(columns={
        'Position': 'RacePosition', #to distinguish from quali ('GridPosition')
        'Time': 'RaceTime'
    }, inplace=True)
    
    # Add metadata
    dataset['Year'] = year
    dataset['Event'] = event
    
    return dataset

# TEST race dataset function (uncomment next 2 lines)
# brazil_2025 = create_race_dataset(2025, 'Brazil')
# print(brazil_2025)

#%%
# ## STEP 5: BUILD FULL DATASET FOR MACHINE LEARNING

import os
import time    # to prevent crashing

# Append a single race's data to our master database
def save_race_to_database(race_data, filename='f1_master_database.csv'):
    
    if race_data.empty:
        return False
    
    # Check if file exists
    if os.path.exists(filename):
        # Append to existing file
        race_data.to_csv(filename, mode='a', header=False, index=False)
        print(f"    ✓ Appended to {filename}")
    else:
        # Create new file
        race_data.to_csv(filename, mode='w', header=True, index=False)
        print(f"    ✓ Created new {filename}")
    
    return True

def collect_single_race(year, event_name):
    """
    Collect data for a single race with better error handling
    """
    try:
        print(f"\n--- {event_name} {year} ---")
        race_data = create_race_dataset(year, event_name)
        
        if not race_data.empty and 'RacePosition' in race_data.columns:
            # Only include drivers who have race results
            race_data = race_data[race_data['RacePosition'].notna()]
            
            if len(race_data) > 0:
                print(f"    Found {len(race_data)} drivers")
                return race_data
        
        print(f"    ✗ No valid data")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return pd.DataFrame()

def list_available_races(year):
    """
    Get list of races for a given year
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter out testing
        races = schedule[schedule['EventFormat'] != 'testing']
        return races['EventName'].tolist()
    except Exception as e:
        print(f"Error getting schedule for {year}: {e}")
        return []

# TEST: List races for 2024
print("Available races in 2024:")
races_2024 = list_available_races(2024)
for i, race in enumerate(races_2024, 1):
    print(f"{i:2d}. {race}")


# Collect one race's data at a time

def collect_races_incrementally(year, race_list=None, database_file='f1_master_database.csv'):
    """
    Collect races one at a time, saving after each
    
    Parameters:
    - year: which year to collect from
    - race_list: list of race names, or None for all races
    - database_file: where to save the data
    """
    
    # Get available races if not provided
    if race_list is None:
        race_list = list_available_races(year)
    
    if not race_list:
        print("No races found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Collecting {len(race_list)} races from {year}")
    print(f"{'='*60}")
    
    success_count = 0
    failed_races = []
    
    for i, race_name in enumerate(race_list, 1):
        print(f"\n[{i}/{len(race_list)}] Processing: {race_name}")
        
        # Collect the race
        race_data = collect_single_race(year, race_name)
        
        # Save to database
        if not race_data.empty:
            if save_race_to_database(race_data, database_file):
                success_count += 1
        else:
            failed_races.append(race_name)
        
        # Small delay so the API doesn't crash
        time.sleep(1)
    
    # Summary of race collection
    print(f"\n{'='*60}")
    print(f"COLLECTION SUMMARY for {year}")
    print(f"{'='*60}")
    print(f"Successfully collected: {success_count}/{len(race_list)} races")
    
    if failed_races:
        print(f"\nFailed races:")
        for race in failed_races:
            print(f"  - {race}")
    
    return success_count

# TEST: collect just 3 races from 2024 to check if it works (uncomment next 6 lines)
# test_races_2024 = [
#     'Bahrain Grand Prix',
#     'Saudi Arabian Grand Prix', 
#     'Australian Grand Prix'
# ]
# collect_races_incrementally(2024, race_list=test_races_2024, database_file='f1_master_database.csv')


# COLLECT ONE FULL SEASON one race at a time: 2024

print("\n" + "="*60)
print("COLLECTING 2024 SEASON")
print("="*60)
collect_races_incrementally(2024, race_list=None, database_file='f1_master_database.csv')


# COLLECT ONE FULL SEASON one race at a time: 2025

print("\n" + "="*60)
print("COLLECTING 2025 SEASON")
print("="*60)
collect_races_incrementally(2025, race_list=None, database_file='f1_master_database.csv')
# %%

# ## 🥳 Congrats! the dataset is built!