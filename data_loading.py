import pandas as pd
import os
import pickle

# --- Configuration ---
# Define the base directory where your script and datasets are located
# os.getcwd() gets the current working directory, which should be your project folder
BASE_DIR = os.getcwd()

# Directory to save the loaded raw dataframes as pickle files
OUTPUT_DIR = os.path.join(BASE_DIR, 'raw_data_pkl')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' ensured to exist.")

# List of your dataset filenames (ensure these match your actual file names exactly)
# Based on your screenshot, using the larger 'Updated_Item.xlsx' for 'Item' data.
dataset_files = {
    "City": "City.xlsx",
    "Continent": "Continent.xlsx",
    "Country": "Country.xlsx",
    "Item": "Updated_Item.xlsx", # Using this assuming it's the comprehensive Item data
    "Mode": "Mode.xlsx",
    "Region": "Region.xlsx",
    "Transaction": "Transaction.xlsx",
    "Type": "Type.xlsx",
    "User": "User.xlsx"
}

# Dictionary to store all loaded DataFrames temporarily
raw_dataframes = {}

print("\n--- Phase 1: Data Loading (`data_loading.py`) ---")
print("Loading datasets from Excel files and saving them as pickle files.")

# --- Data Loading ---
for df_name, file_name in dataset_files.items():
    file_path = os.path.join(BASE_DIR, file_name)
    output_pickle_path = os.path.join(OUTPUT_DIR, f"{df_name.lower()}.pkl") # Save as lowercase .pkl

    try:
        # Load from Excel
        df = pd.read_excel(file_path)
        raw_dataframes[df_name] = df
        print(f"  Successfully loaded '{df_name}' from '{file_name}'. Shape: {df.shape}")

        # Save as pickle file
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"  Saved '{df_name}' as '{output_pickle_path}' for later use.")

    except FileNotFoundError:
        print(f"  ERROR: '{file_name}' not found at '{file_path}'. Please check the filename and path.")
    except Exception as e:
        print(f"  An unexpected error occurred while loading or saving '{file_name}': {e}")

print("\nAll raw datasets loaded and saved as pickle files.")
print("You can now proceed to the `data_preprocessing.py` phase.")

# Optional: Display the first few rows of a couple of dataframes to verify content
if "Transaction" in raw_dataframes:
    print("\n--- Quick Look: Transaction Data (first 3 rows) ---")
    print(raw_dataframes["Transaction"].head(3))

if "User" in raw_dataframes:
    print("\n--- Quick Look: User Data (first 3 rows) ---")
    print(raw_dataframes["User"].head(3))