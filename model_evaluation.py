import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# For Regression Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# For Classification Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# For Recommendation Evaluation
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy as surprise_accuracy

# --- Configuration ---
BASE_DIR = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
print(f"Evaluation results output directory '{EVALUATION_RESULTS_DIR}' ensured to exist.")

# Paths to your data and trained models
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_cleaned_data.pkl')
REG_MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl')
CLF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')
REC_MODEL_PATH = os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl')

# Paths to saved feature lists, scaler, and label encoder
REG_FEATURES_PATH = os.path.join(MODELS_DIR, 'regression_features.pkl')
REG_SCALER_PATH = os.path.join(MODELS_DIR, 'regression_scaler.pkl')
CLF_FEATURES_PATH = os.path.join(MODELS_DIR, 'classification_features.pkl')
CLF_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'classification_label_encoder.pkl')


print("\n--- Phase 5: Model Evaluation (`model_evaluation.py`) ---")
print(f"Loading cleaned and consolidated data from '{PROCESSED_DATA_PATH}'...")

# --- Load Consolidated DataFrame ---
try:
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        df_eval = pickle.load(f)
    print("  Successfully loaded consolidated DataFrame for model evaluation.")
    print(f"  DataFrame shape: {df_eval.shape}")
except FileNotFoundError:
    print(f"  ERROR: Consolidated data not found at '{PROCESSED_DATA_PATH}'. Ensure 'data_preprocessing.py' was run.")
    exit()
except Exception as e:
    print(f"  An unexpected error occurred while loading consolidated data: {e}")
    exit()

print("\n" + "="*80 + "\n")

# --- Load Trained Models, Feature Lists, Scaler, and Label Encoder ---
reg_model = None
clf_model = None
rec_model = None
reg_features = None
reg_scaler = None
clf_features = None
clf_label_encoder = None

try:
    with open(REG_MODEL_PATH, 'rb') as f:
        reg_model = pickle.load(f)
    print(f"  Successfully loaded Regression Model from '{REG_MODEL_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Regression Model: {e}")

try:
    with open(CLF_MODEL_PATH, 'rb') as f:
        clf_model = pickle.load(f)
    print(f"  Successfully loaded Classification Model from '{CLF_MODEL_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Classification Model: {e}")

try:
    with open(REC_MODEL_PATH, 'rb') as f:
        rec_model = pickle.load(f)
    print(f"  Successfully loaded Recommendation Model from '{REC_MODEL_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Recommendation Model: {e}")

try:
    with open(REG_FEATURES_PATH, 'rb') as f:
        reg_features = pickle.load(f)
    print(f"  Successfully loaded Regression features from '{REG_FEATURES_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Regression features: {e}")

try:
    with open(REG_SCALER_PATH, 'rb') as f:
        reg_scaler = pickle.load(f)
    print(f"  Successfully loaded Regression scaler from '{REG_SCALER_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Regression scaler: {e}")

try:
    with open(CLF_FEATURES_PATH, 'rb') as f:
        clf_features = pickle.load(f)
    print(f"  Successfully loaded Classification features from '{CLF_FEATURES_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Classification features: {e}")

try:
    with open(CLF_LABEL_ENCODER_PATH, 'rb') as f:
        clf_label_encoder = pickle.load(f)
    print(f"  Successfully loaded Classification LabelEncoder from '{CLF_LABEL_ENCODER_PATH}'.")
except Exception as e:
    print(f"  ERROR: Could not load Classification LabelEncoder: {e}")


if not all([reg_model, clf_model, rec_model, reg_features, reg_scaler, clf_features, clf_label_encoder]):
    print("\n  Warning: One or more models, feature lists, scaler, or label encoder failed to load. Evaluation might be incomplete.")
    exit()
print("\n" + "="*80 + "\n")

# --- Prepare Data for Evaluation (Re-split test sets and ensure feature order) ---

numerical_features_to_scale = [
    'UserAvgRating', 'AttractionAvgRating', 'UserVisitCount', 'AttractionVisitCount'
]

# --- Prepare Regression Data ---
X_reg_full_df = df_eval[reg_features].copy()
y_reg_full = df_eval['Rating']

for col in numerical_features_to_scale:
    if col in X_reg_full_df.columns:
        X_reg_full_df.loc[:, col] = X_reg_full_df[col].astype(float)

numerical_features_to_scale_existing_reg = [col for col in numerical_features_to_scale if col in X_reg_full_df.columns]
if numerical_features_to_scale_existing_reg:
    X_reg_full_df.loc[:, numerical_features_to_scale_existing_reg] = reg_scaler.transform(X_reg_full_df[numerical_features_to_scale_existing_reg])
    print("  Numerical features transformed using loaded scaler for regression evaluation.")
else:
    print("  No numerical features found for scaling in regression task.")

_, X_test_reg, _, y_test_reg = train_test_split(X_reg_full_df, y_reg_full, test_size=0.2, random_state=42)
print(f"  Regression test data shape: {X_test_reg.shape}")


# --- Prepare Classification Data ---
raw_dataframes_for_clf_mapping = {}
df_names_for_clf = ["transaction"]
# CORRECTED TYPO: os.out to os.path.join
RAW_DATA_DIR_FOR_EVAL = os.path.join(BASE_DIR, 'raw_data_pkl')
for df_name_clf in df_names_for_clf:
    file_path_clf = os.path.join(RAW_DATA_DIR_FOR_EVAL, f"{df_name_clf}.pkl")
    try:
        with open(file_path_clf, 'rb') as f:
            raw_dataframes_for_clf_mapping[df_name_clf.capitalize()] = pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not reload {df_name_clf}.pkl for VisitMode mapping in evaluation: {e}")

df_eval_clf_temp = pd.merge(df_eval, raw_dataframes_for_clf_mapping['Transaction'][['TransactionId', 'VisitMode']].rename(columns={'VisitMode':'OriginalVisitMode'}),
                            on='TransactionId', how='left')

y_clf_full_original_modes = df_eval_clf_temp['OriginalVisitMode']
y_clf_full = clf_label_encoder.transform(y_clf_full_original_modes)

X_clf_full_df = df_eval_clf_temp[clf_features].copy()

for col in numerical_features_to_scale:
    if col in X_clf_full_df.columns:
        X_clf_full_df.loc[:, col] = X_clf_full_df[col].astype(float)

numerical_features_to_scale_existing_clf = [col for col in numerical_features_to_scale if col in X_clf_full_df.columns]
if numerical_features_to_scale_existing_clf:
    X_clf_full_df.loc[:, numerical_features_to_scale_existing_clf] = reg_scaler.transform(X_clf_full_df[numerical_features_to_scale_existing_clf])
    print("  Numerical features transformed using loaded scaler for classification evaluation.")
else:
    print("  No numerical features found for scaling in classification task.")

_, X_test_clf, _, y_test_clf = train_test_split(X_clf_full_df, y_clf_full, test_size=0.2, random_state=42, stratify=y_clf_full)
print(f"  Classification test data shape: {X_test_clf.shape}")


# Prepare data for Recommendation Task
reader_rec = Reader(rating_scale=(1, 5))
ratings_df_rec = df_eval[['UserId', 'AttractionId', 'Rating']]
data_rec = Dataset.load_from_df(ratings_df_rec, reader_rec)
_, testset_rec = surprise_train_test_split(data_rec, test_size=0.2, random_state=42)
print(f"  Recommendation test data size: {len(testset_rec)}")

print("\n  Data prepared for evaluation.")
print("\n" + "="*80 + "\n")


# --- 5. Model Evaluation ---

# --- Evaluation for Regression Task ---
if reg_model:
    print("\n--- Evaluation: Regression Model (LGBMRegressor) ---")
    y_pred_reg = reg_model.predict(X_test_reg)
    mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
    rmse_reg = np.sqrt(mse_reg)
    r2_reg = r2_score(y_test_reg, y_pred_reg)

    print(f"  Mean Squared Error (MSE): {mse_reg:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_reg:.4f}")
    print(f"  R-squared (R2): {r2_reg:.4f}")

    reg_eval_report_path = os.path.join(EVALUATION_RESULTS_DIR, 'regression_evaluation_report.txt')
    with open(reg_eval_report_path, 'w') as f:
        f.write("Regression Model Evaluation Report (LGBMRegressor)\n")
        f.write(f"MSE: {mse_reg:.4f}\n")
        f.write(f"RMSE: {rmse_reg:.4f}\n")
        f.write(f"R2: {r2_reg:.4f}\n")
    print(f"  Regression evaluation report saved to '{reg_eval_report_path}'.")
else:
    print("\n--- Skipping Regression Model Evaluation (Model not loaded) ---")


# --- Evaluation for Classification Task ---
if clf_model:
    print("\n--- Evaluation: Classification Model (RandomForestClassifier) ---")
    y_pred_clf_encoded = clf_model.predict(X_test_clf)

    y_pred_clf = clf_label_encoder.inverse_transform(y_pred_clf_encoded)
    y_test_clf_decoded = clf_label_encoder.inverse_transform(y_test_clf)

    clf_accuracy = accuracy_score(y_test_clf_decoded, y_pred_clf)
    precision = precision_score(y_test_clf_decoded, y_pred_clf, average='weighted', zero_division=0)
    recall = recall_score(y_test_clf_decoded, y_pred_clf, average='weighted', zero_division=0)
    f1 = f1_score(y_test_clf_decoded, y_pred_clf, average='weighted', zero_division=0)
    clf_report = classification_report(y_test_clf_decoded, y_pred_clf, zero_division=0)

    print(f"  Accuracy: {clf_accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print("\nClassification Report:\n", clf_report)

    clf_eval_report_path = os.path.join(EVALUATION_RESULTS_DIR, 'classification_evaluation_report.txt')
    with open(clf_eval_report_path, 'w') as f:
        f.write("Classification Model Evaluation Report (RandomForestClassifier)\n")
        f.write(f"Accuracy: {clf_accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(clf_report)
    print(f"  Classification evaluation report saved to '{clf_eval_report_path}'.")
else:
    print("\n--- Skipping Classification Model Evaluation (Model not loaded) ---")


# --- Evaluation for Recommendation Task ---
if rec_model:
    print("\n--- Evaluation: Recommendation System (SVD) ---")
    predictions_rec = rec_model.test(testset_rec)

    rmse_rec = surprise_accuracy.rmse(predictions_rec, verbose=False)
    mae_rec = surprise_accuracy.mae(predictions_rec, verbose=False)
    
    print(f"  Recommendation RMSE: {rmse_rec:.4f}")
    print(f"  Recommendation MAE: {mae_rec:.4f}")

    rec_eval_report_path = os.path.join(EVALUATION_RESULTS_DIR, 'recommendation_evaluation_report.txt')
    with open(rec_eval_report_path, 'w') as f:
        f.write("Recommendation System Evaluation Report (SVD)\n")
        f.write(f"RMSE: {rmse_rec:.4f}\n")
        f.write(f"MAE: {mae_rec:.4f}\n")
    print(f"  Recommendation evaluation report saved to '{rec_eval_report_path}'.")
else:
    print("\n--- Skipping Recommendation System Evaluation (Model not loaded) ---")

print("\nModel Evaluation complete. Reports are saved in the 'evaluation_results' directory.")
print("\n" + "="*80 + "\n")