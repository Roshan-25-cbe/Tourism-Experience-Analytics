import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import numpy as np

# --- Configuration ---
BASE_DIR = os.getcwd()

# Directory where the raw dataframes (pickle files) are stored
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data_pkl')

# Directory to save the cleaned and consolidated dataframe
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
print(f"Processed data output directory '{PROCESSED_DATA_DIR}' ensured to exist.")

# List of dataframe names to load (matching the names saved as .pkl)
df_names_to_load = [
    "city", "continent", "country", "item", "mode", "region",
    "transaction", "type", "user"
]

# Dictionary to store the loaded raw DataFrames
raw_dataframes = {}

print("\n--- Phase 2: Data Cleaning & Preprocessing (`data_preprocessing.py`) ---")
print("Loading raw datasets from pickle files...")

# --- Load Raw DataFrames ---
for df_name in df_names_to_load:
    file_path = os.path.join(RAW_DATA_DIR, f"{df_name}.pkl")
    try:
        with open(file_path, 'rb') as f:
            raw_dataframes[df_name.capitalize()] = pickle.load(f)
        print(f"  Successfully loaded '{df_name.capitalize()}' from '{file_path}'.")
    except FileNotFoundError:
        print(f"  ERROR: Pickle file for '{df_name}' not found at '{file_path}'. Ensure 'tourism_analytics.py' was run.")
    except Exception as e:
        print(f"  An unexpected error occurred while loading '{file_path}': {e}")

print("\nAll raw datasets loaded for preprocessing.")
print("\n" + "="*80 + "\n")

print("--- Step 4.1: Data Cleaning ---")

# 1. Handling Missing Values
print("\n  1. Handling Missing Values:")

# Transaction DataFrame: Rating, VisitMode
if 'Transaction' in raw_dataframes:
    df_transaction = raw_dataframes['Transaction']
    if 'Rating' in df_transaction.columns and df_transaction['Rating'].isnull().sum() > 0:
        print(f"    - Filling missing 'Rating' in Transaction ({df_transaction['Rating'].isnull().sum()} NaNs) with median.")
        df_transaction['Rating'].fillna(df_transaction['Rating'].median(), inplace=True)
    if 'VisitMode' in df_transaction.columns and df_transaction['VisitMode'].isnull().sum() > 0:
        print(f"    - Filling missing 'VisitMode' in Transaction ({df_transaction['VisitMode'].isnull().sum()} NaNs) with mode.")
        mode_visit_mode = df_transaction['VisitMode'].mode()[0]
        df_transaction['VisitMode'].fillna(mode_visit_mode, inplace=True)

# User DataFrame: Geographical IDs
if 'User' in raw_dataframes:
    df_user = raw_dataframes['User']
    for col_id in ['ContinentId', 'RegionId', 'CountryId', 'CityId']:
        if col_id in df_user.columns and df_user[col_id].isnull().sum() > 0:
            print(f"    - Warning: Missing '{col_id}' in User ({df_user[col_id].isnull().sum()} NaNs). Filling with 0.")
            df_user[col_id].fillna(0, inplace=True)


# City DataFrame: CityName
if 'City' in raw_dataframes and 'CityName' in raw_dataframes['City'].columns:
    df_city = raw_dataframes['City']
    if df_city['CityName'].isnull().sum() > 0:
        print(f"    - Filling missing 'CityName' in City ({df_city['CityName'].isnull().sum()} NaNs) with 'Unknown City'.")
        df_city['CityName'].fillna('Unknown City', inplace=True)


# 2. Resolve Discrepancies in Categorical Variables (Standardize strings)
print("\n  2. Standardizing Categorical String Columns:")
string_cols_to_standardize = {
    'City': ['CityName'],
    'Type': ['AttractionType'],
    'Mode': ['VisitMode'],
    'Continent': ['Continent'],
    'Country': ['Country'],
    'Region': ['Region'],
    'Item': ['Attraction', 'AttractionAddress']
}

for df_name, cols in string_cols_to_standardize.items():
    if df_name in raw_dataframes:
        for col in cols:
            if col in raw_dataframes[df_name].columns:
                raw_dataframes[df_name][col] = raw_dataframes[df_name][col].astype(str).str.strip().str.title()
                print(f"    - Standardized '{col}' in {df_name}.")

if 'Transaction' in raw_dataframes and 'VisitMode' in raw_dataframes['Transaction'].columns:
    raw_dataframes['Transaction']['VisitMode'] = raw_dataframes['Transaction']['VisitMode'].astype(str).str.strip().str.title()
    print("    - Standardized 'VisitMode' in Transaction (after specific handling).")

# 3. Standardize Date and Time Format
print("\n  3. Standardizing Date/Time Formats:")
if 'Transaction' in raw_dataframes:
    df_transaction = raw_dataframes['Transaction']
    for col in ['VisitYear', 'VisitMonth']:
        if col in df_transaction.columns:
            df_transaction[col] = pd.to_numeric(df_transaction[col], errors='coerce')
            if df_transaction[col].isnull().sum() > 0:
                print(f"    - Warning: Non-numeric '{col}' values found and converted to NaN in Transaction. Filling with mode.")
                df_transaction[col].fillna(df_transaction[col].mode()[0], inplace=True)
            df_transaction[col] = df_transaction[col].astype(int)
    print("    - Ensured 'VisitYear' and 'VisitMonth' are integers in Transaction.")

# 4. Handle Outliers or Incorrect Entries (e.g., Rating)
print("\n  4. Handling Outliers/Incorrect Entries:")
if 'Transaction' in raw_dataframes and 'Rating' in raw_dataframes['Transaction'].columns:
    df_transaction = raw_dataframes['Transaction']
    original_rows = df_transaction.shape[0]
    df_transaction = df_transaction[(df_transaction['Rating'] >= 1) & (df_transaction['Rating'] <= 5)]
    rows_removed = original_rows - df_transaction.shape[0]
    if rows_removed > 0:
        print(f"    - Removed {rows_removed} rows from Transaction where 'Rating' was outside 1-5 range.")
        raw_dataframes['Transaction'] = df_transaction
    else:
        print("    - No 'Rating' outliers found outside 1-5 range in Transaction.")

print("\n  Data cleaning complete. Re-checking missing values across all DataFrames:")
for df_name, df in raw_dataframes.items():
    print(f"    --- Missing values for {df_name} ---")
    print(df.isnull().sum())

print("\n" + "="*80 + "\n")

print("--- Step 4.2: Data Preprocessing - Joining Data ---")
print("  Creating the consolidated dataset by merging all relevant dataframes.")

df_consolidated = raw_dataframes['Transaction'].copy()
print("  Starting merge with Transaction data.")

df_consolidated = pd.merge(df_consolidated, raw_dataframes['User'], on='UserId', how='left')
print("  Merging with User data...")

df_consolidated = pd.merge(df_consolidated, raw_dataframes['Item'], on='AttractionId', how='left')
print("  Merging with Item (Attraction) data...")

df_consolidated = pd.merge(df_consolidated, raw_dataframes['Type'], on='AttractionTypeId', how='left')
print("  Merging with Type (Attraction Type) data...")

# Merge CityName and AttractionName for display, but won't OHE them later
df_consolidated = pd.merge(
    df_consolidated,
    raw_dataframes['City'][['CityId', 'CityName']].rename(columns={'CityName': 'UserCityName'}),
    left_on='CityId',
    right_on='CityId',
    how='left'
)
print("  Merging with City data for User's City Name...")

df_consolidated = pd.merge(
    df_consolidated,
    raw_dataframes['City'][['CityId', 'CityName']].rename(columns={'CityId': 'AttractionCityId', 'CityName': 'AttractionCityName'}),
    left_on='AttractionCityId',
    right_on='AttractionCityId',
    how='left'
)
print("  Merging with City data for Attraction's City Name...")

df_consolidated = pd.merge(df_consolidated, raw_dataframes['Continent'], on='ContinentId', how='left')
print("  Merging with Continent data...")

df_consolidated = pd.merge(
    df_consolidated,
    raw_dataframes['Country'][['CountryId', 'Country']].rename(columns={'Country': 'UserCountry'}),
    on='CountryId',
    how='left'
)
print("  Merging with Country data (for UserCountry)...")

df_consolidated = pd.merge(
    df_consolidated,
    raw_dataframes['Region'][['RegionId', 'Region']].rename(columns={'Region': 'UserRegion'}),
    on='RegionId',
    how='left'
)
print("  Merging with Region data (for UserRegion)...")


# --- Feature Engineering (initial, based on project objectives) ---
print("\n--- Initial Feature Engineering ---")

if 'VisitMonth' in df_consolidated.columns:
    df_consolidated['MonthName'] = pd.to_datetime(df_consolidated['VisitMonth'], format='%m').dt.strftime('%B')
    print("  Created 'MonthName' feature.")

user_avg_rating = df_consolidated.groupby('UserId')['Rating'].mean().reset_index()
user_avg_rating.rename(columns={'Rating': 'UserAvgRating'}, inplace=True)
df_consolidated = pd.merge(df_consolidated, user_avg_rating, on='UserId', how='left')
print("  Added 'UserAvgRating' feature.")

attraction_avg_rating = df_consolidated.groupby('AttractionId')['Rating'].mean().reset_index()
attraction_avg_rating.rename(columns={'Rating': 'AttractionAvgRating'}, inplace=True)
df_consolidated = pd.merge(df_consolidated, attraction_avg_rating, on='AttractionId', how='left')
print("  Added 'AttractionAvgRating' feature.")

user_visit_count = df_consolidated.groupby('UserId').size().reset_index(name='UserVisitCount')
df_consolidated = pd.merge(df_consolidated, user_visit_count, on='UserId', how='left')
print("  Added 'UserVisitCount' feature.")

attraction_visit_count = df_consolidated.groupby('AttractionId').size().reset_index(name='AttractionVisitCount')
df_consolidated = pd.merge(df_consolidated, attraction_visit_count, on='AttractionId', how='left')
print("  Added 'AttractionVisitCount' feature.")

# --- Optimized One-Hot Encoding: Only for lower cardinality features ---
print("\n--- Optimized One-Hot Encoding (for reasonable cardinality features) ---")

# These are the columns that will be ONE-HOT ENCODED for models.
# High-cardinality ones like CityName and AttractionName are EXCLUDED here.
categorical_cols_to_encode_for_models = [
    'VisitMode', 'MonthName', 'AttractionType', 'Continent',
    'UserCountry', 'UserRegion' # Keep country/region OHE for broader geographical patterns
    # EXCLUDE 'UserCityName', 'AttractionCityName', 'Attraction' to avoid 5000+ columns
]

for col in categorical_cols_to_encode_for_models:
    if col in df_consolidated.columns:
        if df_consolidated[col].dtype == 'object':
            original_cols = df_consolidated.shape[1]
            if df_consolidated[col].isnull().sum() > 0:
                df_consolidated[col].fillna('Unknown', inplace=True)
                print(f"  Filled NaNs in '{col}' with 'Unknown' before encoding.")
            
            df_consolidated = pd.get_dummies(df_consolidated, columns=[col], prefix=col, drop_first=True)
            print(f"  One-hot encoded '{col}'. Columns added: {df_consolidated.shape[1] - original_cols}")
        else:
            print(f"  '{col}' is not of object type, skipping encoding.")
    else:
        print(f"  Warning: '{col}' not found in consolidated DataFrame for encoding.")


# --- Final Check of Consolidated DataFrame ---
print("\n--- Consolidated DataFrame Final Overview ---")
print("\nConsolidated DataFrame (first 5 rows) after all cleaning and preprocessing:")
print(df_consolidated.head())
print("\nConsolidated DataFrame shape:", df_consolidated.shape) # EXPECT THIS TO BE MUCH SMALLER
print("\nConsolidated DataFrame columns (first 20):", df_consolidated.columns.tolist()[:20])
print("\nMissing values in Consolidated DataFrame after preprocessing:")
print(df_consolidated.isnull().sum()[df_consolidated.isnull().sum() > 0])

print("\n" + "="*80 + "\n")

print("--- Saving Processed Data ---")
processed_output_path = os.path.join(PROCESSED_DATA_DIR, 'consolidated_cleaned_data.pkl')
try:
    with open(processed_output_path, 'wb') as f:
        pickle.dump(df_consolidated, f)
    print(f"Successfully saved cleaned and consolidated DataFrame to '{processed_output_path}'.")
except Exception as e:
    print(f"ERROR: Could not save consolidated DataFrame to pickle: {e}")


# --- NEW: Save a SMALL SAMPLE to Excel for better viewing ---
excel_sample_path = os.path.join(PROCESSED_DATA_DIR, 'consolidated_data_sample_for_viewing.xlsx')
try:
    # Select key columns that are human-readable and exclude most OHE features
    viewable_columns = [
        'TransactionId', 'UserId', 'AttractionId', 'Rating', 'VisitYear', 'VisitMonth',
        'VisitMode', 'Attraction', 'AttractionType', 'UserCityName', 'AttractionCityName',
        'UserCountry', 'UserRegion', 'Continent',
        'UserAvgRating', 'AttractionAvgRating', 'UserVisitCount', 'AttractionVisitCount',
        'MonthName' # The original MonthName string
    ]
    
    # Filter to only existing columns in case some were dropped/renamed
    viewable_columns_exist = [col for col in viewable_columns if col in df_consolidated.columns]

    # Take the first 1000 rows as a sample
    df_view_sample = df_consolidated[viewable_columns_exist].head(1000)
    
    # Save to Excel
    df_view_sample.to_excel(excel_sample_path, index=False)
    print(f"Successfully saved a SAMPLE of consolidated DataFrame to Excel for viewing: '{excel_sample_path}'.")
    print("This file contains the first 1000 rows and key human-readable columns.")
    print("The FULL processed data (with all 5000+ columns) is saved in the .pkl file for model training.")
except Exception as e:
    print(f"ERROR: Could not save a SAMPLE of consolidated DataFrame to Excel: {e}")


print("\nData preprocessing complete. You can now proceed to the `EDA.py` phase.")