import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight

# For Regression
from lightgbm import LGBMRegressor # Using LightGBM Regressor for regression

# For Classification
from sklearn.ensemble import RandomForestClassifier # Using RandomForest Classifier for classification

# For Recommendation
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import SVD
from surprise import accuracy as surprise_accuracy

# --- Configuration ---
BASE_DIR = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models output directory '{MODELS_DIR}' ensured to exist.")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_cleaned_data.pkl')

print("\n--- Phase 4: Model Training (`model_training.py`) ---")
print(f"Loading cleaned and consolidated data from '{PROCESSED_DATA_PATH}'...")

try:
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        df_model = pickle.load(f)
    print("  Successfully loaded consolidated DataFrame for model training.")
    print(f"  DataFrame shape: {df_model.shape}")
except FileNotFoundError:
    print(f"  ERROR: Consolidated data not found at '{PROCESSED_DATA_PATH}'. Ensure 'data_preprocessing.py' was run.")
    exit()
except Exception as e:
    print(f"  An unexpected error occurred while loading consolidated data: {e}")
    exit()

raw_dataframes_for_mapping = {}
df_names_to_load = ["transaction"]

RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data_pkl')
for df_name in df_names_to_load:
    file_path = os.path.join(RAW_DATA_DIR, f"{df_name}.pkl")
    try:
        with open(file_path, 'rb') as f:
            raw_dataframes_for_mapping[df_name.capitalize()] = pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not reload {df_name}.pkl for VisitMode mapping: {e}")


print("\n" + "="*80 + "\n")

# --- 4. Model Training Tasks ---

# --- Task 1: Regression (Predicting Attraction Ratings) ---
print("\n--- Task 1: Regression - Predicting Attraction Ratings ---")

numerical_features_to_scale = [
    'UserAvgRating', 'AttractionAvgRating', 'UserVisitCount', 'AttractionVisitCount'
]

final_regression_features = ['UserAvgRating', 'AttractionAvgRating', 'UserVisitCount', 'AttractionVisitCount']
ohe_prefixes_reg = ['MonthName_', 'AttractionType_', 'UserCityName_', 'AttractionCityName_', 'Continent_', 'UserCountry_', 'UserRegion_', 'Attraction_']
for col in df_model.columns:
    for prefix in ohe_prefixes_reg:
        if col.startswith(prefix):
            final_regression_features.append(col)
            break

X_reg = df_model[final_regression_features]
y_reg = df_model['Rating']

for col in numerical_features_to_scale:
    if col in X_reg.columns:
        X_reg.loc[:, col] = X_reg[col].astype(float)


scaler_reg = StandardScaler()
numerical_features_to_scale_existing = [col for col in numerical_features_to_scale if col in X_reg.columns]
if numerical_features_to_scale_existing:
    X_reg.loc[:, numerical_features_to_scale_existing] = scaler_reg.fit_transform(X_reg[numerical_features_to_scale_existing])
    print(f"  Numerical features {numerical_features_to_scale_existing} scaled for regression task.")
else:
    print("  No numerical features found for scaling in regression task.")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
print(f"  Regression data split: Train {X_train_reg.shape}, Test {X_test_reg.shape}")

reg_features_path = os.path.join(MODELS_DIR, 'regression_features.pkl')
with open(reg_features_path, 'wb') as f:
    pickle.dump(X_train_reg.columns.tolist(), f)
print(f"  Regression feature names saved to '{reg_features_path}'.")

reg_scaler_path = os.path.join(MODELS_DIR, 'regression_scaler.pkl')
with open(reg_scaler_path, 'wb') as f:
    pickle.dump(scaler_reg, f)
print(f"  Regression scaler saved to '{reg_scaler_path}'.")


# Train Regression Model (LGBMRegressor)
print("  Training LGBMRegressor for rating prediction...")
reg_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
reg_model.fit(X_train_reg, y_train_reg)
print("  LGBMRegressor trained.")

reg_model_path = os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl') # Changed model name
with open(reg_model_path, 'wb') as f:
    pickle.dump(reg_model, f)
print(f"  Regression model saved to '{reg_model_path}'.")

print("\n" + "="*80 + "\n")


# --- Task 2: Classification (Predicting User Visit Mode) ---
print("\n--- Task 2: Classification - Predicting User Visit Mode ---")

raw_transaction_df = raw_dataframes_for_mapping['Transaction']
df_model_clf = pd.merge(df_model, raw_transaction_df[['TransactionId', 'VisitMode']].rename(columns={'VisitMode':'OriginalVisitMode'}),
                        on='TransactionId', how='left')
print("  Original 'VisitMode' (as OriginalVisitMode) merged for classification target.")


le = LabelEncoder()
df_model_clf['EncodedVisitMode'] = le.fit_transform(df_model_clf['OriginalVisitMode'])
print("  Classification target 'OriginalVisitMode' encoded to 'EncodedVisitMode'.")


features_for_classification = [
    col for col in final_regression_features
    if not col.startswith('VisitMode_')
]

X_clf = df_model_clf[features_for_classification]
y_clf = df_model_clf['EncodedVisitMode']


for col in numerical_features_to_scale:
    if col in X_clf.columns:
        X_clf.loc[:, col] = X_clf[col].astype(float)


if numerical_features_to_scale_existing:
    X_clf.loc[:, numerical_features_to_scale_existing] = scaler_reg.transform(X_clf[numerical_features_to_scale_existing])
    print(f"  Numerical features {numerical_features_to_scale_existing} scaled for classification task.")
else:
    print("  No numerical features found for scaling in classification task.")


X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
print(f"  Classification data split: Train {X_train_clf.shape}, Test {X_test_clf.shape}")

clf_features_path = os.path.join(MODELS_DIR, 'classification_features.pkl')
with open(clf_features_path, 'wb') as f:
    pickle.dump(X_train_clf.columns.tolist(), f)
print(f"  Classification feature names saved to '{clf_features_path}'.")

clf_label_encoder_path = os.path.join(MODELS_DIR, 'classification_label_encoder.pkl')
with open(clf_label_encoder_path, 'wb') as f:
    pickle.dump(le, f)
print(f"  Classification LabelEncoder saved to '{clf_label_encoder_path}'.")


class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_clf),
    y=y_train_clf
)
class_weights_dict = dict(zip(np.unique(y_train_clf), class_weights))
print(f"  Computed class weights for classification: {class_weights_dict}")

sample_weights = y_train_clf.map(class_weights_dict)


# Train Classification Model (RandomForestClassifier)
print("  Training RandomForestClassifier for VisitMode prediction (with class weighting)...")
# For RandomForestClassifier, 'class_weight' parameter can be set to 'balanced' or a dict,
# or 'sample_weight' can be passed to the fit method. Using sample_weight for consistency.
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_model.fit(X_train_clf, y_train_clf, sample_weight=sample_weights) # Pass sample_weight
print("  RandomForestClassifier trained.")

clf_model_path = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl') # Changed model name
with open(clf_model_path, 'wb') as f:
    pickle.dump(clf_model, f)
print(f"  Classification model saved to '{clf_model_path}'.")

print("\n" + "="*80 + "\n")


# --- Task 3: Recommendation (Personalized Attraction Suggestions) ---
print("\n--- Task 3: Recommendation - Personalized Attraction Suggestions ---")

print("  Preparing data for Recommendation System (Surprise library)...")
reader = Reader(rating_scale=(1, 5))
ratings_df = df_model[['UserId', 'AttractionId', 'Rating']]
data = Dataset.load_from_df(ratings_df, reader)

trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
print(f"  Recommendation data split: Trainset size {trainset.n_ratings}, Testset size {len(testset)}")

print("  Training SVD model for recommendation...")
algo_svd = SVD(random_state=42)
algo_svd.fit(trainset)
print("  SVD model trained.")

svd_model_path = os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl')
with open(svd_model_path, 'wb') as f:
    pickle.dump(algo_svd, f)
print(f"  Recommendation model (SVD) saved to '{svd_model_path}'.")

print("\nModel Training complete. Trained models are saved in the 'models' directory.")
print("\n" + "="*80 + "\n")