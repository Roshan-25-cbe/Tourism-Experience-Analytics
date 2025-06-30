import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_DIR = os.getcwd()

# Directory where the raw dataframes (pickle files) are stored - ADDED THIS DEFINITION
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data_pkl')

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
EDA_OUTPUT_DIR = os.path.join(BASE_DIR, 'eda_plot') # Changed to 'eda_plot' as requested
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
print(f"EDA results output directory '{EDA_OUTPUT_DIR}' ensured to exist.")

# Path to your cleaned and consolidated data
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_cleaned_data.pkl')

print("\n--- Phase 3: Exploratory Data Analysis (`EDA.py`) ---")
print(f"Loading cleaned and consolidated data from '{PROCESSED_DATA_PATH}'...")

# --- Load Consolidated DataFrame ---
try:
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        df_eda = pickle.load(f)
    print("  Successfully loaded consolidated DataFrame for EDA.")
    print(f"  DataFrame shape: {df_eda.shape}")
    print(f"  DataFrame columns (first 20): {df_eda.columns.tolist()[:20]}")
except FileNotFoundError:
    print(f"  ERROR: Consolidated data not found at '{PROCESSED_DATA_PATH}'. Ensure 'data_preprocessing.py' was run.")
    exit() # Exit if the data isn't found
except Exception as e:
    print(f"  An unexpected error occurred while loading consolidated data: {e}")
    exit()

print("\n" + "="*80 + "\n") # Separator

print("--- Step 3: Exploratory Data Analysis (EDA) ---")
# Refer to project document 'Approach' -> 'Exploratory Data Analysis (EDA)' section

# Set aesthetic style for plots
sns.set_style("whitegrid")

# Load raw dataframes for mapping original names to IDs for cleaner plots
print("  Reloading raw dataframes for mapping original names to IDs for plots...")
raw_dataframes_for_mapping = {}
df_names_to_load = [
    "city", "continent", "country", "item", "mode", "region",
    "transaction", "type", "user"
] # Need this list here too

for df_name in df_names_to_load:
    file_path = os.path.join(RAW_DATA_DIR, f"{df_name}.pkl")
    try:
        with open(file_path, 'rb') as f:
            raw_dataframes_for_mapping[df_name.capitalize()] = pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not reload {df_name}.pkl for mapping: {e}")

# Merging back readable names to df_eda for plotting
# Using suffixes to distinguish original ID columns from the newly merged name columns
if 'ContinentId' in df_eda.columns and 'Continent' in raw_dataframes_for_mapping['Continent'].columns:
    df_eda = pd.merge(df_eda, raw_dataframes_for_mapping['Continent'][['ContinentId', 'Continent']], on='ContinentId', how='left', suffixes=('', '_original_name'))
    df_eda.rename(columns={'Continent': 'Continent_original'}, inplace=True) # Rename the merged column
    print("  Added 'Continent_original' (original name) to consolidated DataFrame for EDA.")
if 'CountryId' in df_eda.columns and 'Country' in raw_dataframes_for_mapping['Country'].columns:
    df_eda = pd.merge(df_eda, raw_dataframes_for_mapping['Country'][['CountryId', 'Country']].rename(columns={'Country':'OriginalUserCountry'}), on='CountryId', how='left', suffixes=('', '_original_name'))
    print("  Added 'OriginalUserCountry' (original name) to consolidated DataFrame for EDA.")
if 'RegionId' in df_eda.columns and 'Region' in raw_dataframes_for_mapping['Region'].columns:
    df_eda = pd.merge(df_eda, raw_dataframes_for_mapping['Region'][['RegionId', 'Region']].rename(columns={'Region':'OriginalUserRegion'}), on='RegionId', how='left', suffixes=('', '_original_name'))
    print("  Added 'OriginalUserRegion' (original name) to consolidated DataFrame for EDA.")
if 'AttractionTypeId' in df_eda.columns and 'AttractionType' in raw_dataframes_for_mapping['Type'].columns:
    df_eda = pd.merge(df_eda, raw_dataframes_for_mapping['Type'][['AttractionTypeId', 'AttractionType']].rename(columns={'AttractionType':'OriginalAttractionType'}), on='AttractionTypeId', how='left', suffixes=('', '_original_name'))
    print("  Added 'OriginalAttractionType' (original name) to consolidated DataFrame for EDA.")


# 1. Visualize user distribution across continents, countries, and regions.
print("\n  1. Analyzing User Distribution:")

# Plotting User Distribution by Continent
if 'Continent_original' in df_eda.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_eda, y='Continent_original', order=df_eda['Continent_original'].value_counts().index)
    plt.title('User Distribution Across Continents')
    plt.xlabel('Number of Visits')
    plt.ylabel('Continent')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'user_distribution_continent.png'))
    plt.close() # Close plot to free memory
    print("  Generated 'user_distribution_continent.png'")

# Plotting User Distribution by OriginalUserCountry (Top N)
if 'OriginalUserCountry' in df_eda.columns:
    plt.figure(figsize=(12, 8))
    # Handle potential NaNs in 'OriginalUserCountry' if some user IDs didn't map
    country_counts = df_eda['OriginalUserCountry'].value_counts()
    top_countries = country_counts.nlargest(15).index.tolist()
    # Filter out NaN if it's in top_countries and you don't want to plot it, or explicitly add it.
    if 'nan' in top_countries: top_countries.remove('nan') # Remove 'nan' if it somehow became a string entry

    sns.countplot(data=df_eda[df_eda['OriginalUserCountry'].isin(top_countries)],
                  y='OriginalUserCountry',
                  order=top_countries)
    plt.title('Top 15 User Countries by Number of Visits')
    plt.xlabel('Number of Visits')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'user_distribution_top_countries.png'))
    plt.close()
    print("  Generated 'user_distribution_top_countries.png'")

# 2. Explore attraction types and their popularity based on user ratings.
print("\n  2. Analyzing Attraction Popularity and Ratings:")
if 'OriginalAttractionType' in df_eda.columns:
    # Popularity by count of visits for each type
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df_eda, y='OriginalAttractionType', order=df_eda['OriginalAttractionType'].value_counts().index)
    plt.title('Popularity of Attraction Types (by Visit Count)')
    plt.xlabel('Number of Visits')
    plt.ylabel('Attraction Type')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'attraction_type_popularity_count.png'))
    plt.close()
    print("  Generated 'attraction_type_popularity_count.png'")

    # Average rating per attraction type
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_eda, x='Rating', y='OriginalAttractionType',
                order=df_eda.groupby('OriginalAttractionType')['Rating'].mean().sort_values(ascending=False).index,
                estimator=lambda x: x.mean()) # ensure it's mean
    plt.title('Average Rating per Attraction Type')
    plt.xlabel('Average Rating')
    plt.ylabel('Attraction Type')
    plt.xlim(1, 5) # Assuming ratings are 1-5
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'attraction_type_avg_rating.png'))
    plt.close()
    print("  Generated 'attraction_type_avg_rating.png'")

# Top N Attractions by Average Rating
if 'Attraction' in df_eda.columns and 'AttractionAvgRating' in df_eda.columns:
    # Ensure 'Attraction' column doesn't have 'nan' string entries if they result from a failed merge/fill
    df_attractions_for_plot = df_eda[['Attraction', 'AttractionAvgRating']].drop_duplicates().dropna(subset=['Attraction'])
    top_rated_attractions = df_attractions_for_plot.sort_values(by='AttractionAvgRating', ascending=False).head(10)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='AttractionAvgRating', y='Attraction', data=top_rated_attractions)
    plt.title('Top 10 Attractions by Average Rating')
    plt.xlabel('Average Rating')
    plt.ylabel('Attraction')
    plt.xlim(1, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'top_10_attractions_by_avg_rating.png'))
    plt.close()
    print("  Generated 'top_10_attractions_by_avg_rating.png'")

# 3. Investigate correlation between Visit Mode and user demographics to identify patterns.
print("\n  3. Analyzing Correlation between Visit Mode and User Demographics:")

# Overall Visit Mode distribution
# 'VisitMode' column is already cleaned and in Title Case from preprocessing
if 'VisitMode' in df_eda.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_eda, y='VisitMode', order=df_eda['VisitMode'].value_counts().index)
    plt.title('Overall Distribution of Visit Modes')
    plt.xlabel('Number of Visits')
    plt.ylabel('Visit Mode')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'overall_visit_mode_distribution.png'))
    plt.close()
    print("  Generated 'overall_visit_mode_distribution.png'")

    # Visit Mode by Continent (using original continent name for clarity)
    if 'Continent_original' in df_eda.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df_eda, y='Continent_original', hue='VisitMode',
                      order=df_eda['Continent_original'].value_counts().index)
        plt.title('Visit Mode Distribution by Continent')
        plt.xlabel('Number of Visits')
        plt.ylabel('Continent')
        plt.legend(title='Visit Mode')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'visit_mode_by_continent.png'))
        plt.close()
        print("  Generated 'visit_mode_by_continent.png'")

    # Visit Mode by MonthName
    if 'MonthName' in df_eda.columns:
        plt.figure(figsize=(12, 8))
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        sns.countplot(data=df_eda, x='MonthName', hue='VisitMode', order=month_order)
        plt.title('Visit Mode Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Visits')
        plt.legend(title='Visit Mode')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'visit_mode_by_month.png'))
        plt.close()
        print("  Generated 'visit_mode_by_month.png'")

# 4. Analyze distribution of ratings across different attractions and regions.
print("\n  4. Analyzing Rating Distribution Across Attractions and Regions:")

# Overall Rating Distribution
if 'Rating' in df_eda.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_eda['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], kde=False, stat='count')
    plt.title('Overall Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'overall_rating_distribution.png'))
    plt.close()
    print("  Generated 'overall_rating_distribution.png'")

# Rating distribution by OriginalUserRegion (Top N regions)
if 'OriginalUserRegion' in df_eda.columns:
    plt.figure(figsize=(12, 8))
    # Handle potential NaNs
    region_counts = df_eda['OriginalUserRegion'].value_counts()
    top_regions = region_counts.nlargest(10).index.tolist()
    if 'nan' in top_regions: top_regions.remove('nan')

    sns.boxplot(data=df_eda[df_eda['OriginalUserRegion'].isin(top_regions)],
                x='Rating', y='OriginalUserRegion',
                order=top_regions)
    plt.title('Rating Distribution by Top Regions')
    plt.xlabel('Rating')
    plt.ylabel('Region')
    plt.xlim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'rating_distribution_by_region.png'))
    plt.close()
    print("  Generated 'rating_distribution_by_region.png'")

print("\nEDA complete. All generated plots are saved in the 'eda_plot' directory.")
print("\n" + "="*80 + "\n") # Separator