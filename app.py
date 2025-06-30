import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data_pkl')
EDA_PLOT_DIR = os.path.join(BASE_DIR, 'eda_plot')

# Paths to models, scalers, and feature lists
REG_MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl')
CLF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')
REC_MODEL_PATH = os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl')
REG_FEATURES_PATH = os.path.join(MODELS_DIR, 'regression_features.pkl')
REG_SCALER_PATH = os.path.join(MODELS_DIR, 'regression_scaler.pkl')
CLF_FEATURES_PATH = os.path.join(MODELS_DIR, 'classification_features.pkl')
CLF_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'classification_label_encoder.pkl')
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_cleaned_data.pkl')

# --- Load Data and Models (Cache for efficiency in Streamlit) ---
@st.cache_data
def load_data():
    try:
        with open(PROCESSED_DATA_PATH, 'rb') as f:
            df = pickle.load(f)
        return df
    except Exception as e:
        st.error(f"Failed to load processed data: {e}. Please ensure `data_preprocessing.py` ran successfully.")
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        with st.spinner("Loading Regression Model..."):
            with open(REG_MODEL_PATH, 'rb') as f:
                models['regressor'] = pickle.load(f)
        with st.spinner("Loading Classification Model..."):
            with open(CLF_MODEL_PATH, 'rb') as f:
                models['classifier'] = pickle.load(f)
        with st.spinner("Loading Recommendation Model..."):
            with open(REC_MODEL_PATH, 'rb') as f:
                models['recommender'] = pickle.load(f)
        with st.spinner("Loading Feature Lists and Scaler..."):
            with open(REG_FEATURES_PATH, 'rb') as f:
                models['reg_features'] = pickle.load(f)
            with open(REG_SCALER_PATH, 'rb') as f:
                models['reg_scaler'] = pickle.load(f)
            with open(CLF_FEATURES_PATH, 'rb') as f:
                models['clf_features'] = pickle.load(f)
            with open(CLF_LABEL_ENCODER_PATH, 'rb') as f:
                models['clf_label_encoder'] = pickle.load(f)
        st.success("All models and supporting files loaded!")
        return models
    except Exception as e:
        st.error(f"Error loading models or supporting files: {e}. Ensure `model_training.py` ran successfully.")
        return None

@st.cache_data
def load_raw_data_for_mapping():
    raw_dfs = {}
    df_names_to_load = ["city", "continent", "country", "item", "mode", "region", "transaction", "type", "user"]
    for df_name in df_names_to_load:
        file_path = os.path.join(RAW_DATA_DIR, f"{df_name}.pkl")
        try:
            with open(file_path, 'rb') as f:
                raw_dfs[df_name.capitalize()] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load raw '{df_name}.pkl' for mapping: {e}")
    return raw_dfs

df = load_data()
models = load_models()
raw_dfs_map = load_raw_data_for_mapping()

if df is None or models is None:
    st.stop() # Stop the app if essential data/models didn't load

reg_model = models['regressor']
clf_model = models['classifier']
rec_model = models['recommender']
reg_features = models['reg_features'] # Features list for regression
reg_scaler = models['reg_scaler']
clf_features = models['clf_features'] # Features list for classification
clf_label_encoder = models['clf_label_encoder']

# --- Prepare data for predictions (helper functions) ---

def get_id_to_name_map(df_name, id_col, name_col):
    if df_name in raw_dfs_map and id_col in raw_dfs_map[df_name].columns and name_col in raw_dfs_map[df_name].columns:
        return {k: str(v) for k, v in raw_dfs_map[df_name].set_index(id_col)[name_col].to_dict().items()}
    return {}

def get_name_to_id_map(df_name, id_col, name_col):
    if df_name in raw_dfs_map and id_col in raw_dfs_map[df_name].columns and name_col in raw_dfs_map[df_name].columns:
        return {str(k): v for k, v in raw_dfs_map[df_name].set_index(name_col)[id_col].to_dict().items()}
    return {}
    
original_visit_modes = sorted(clf_label_encoder.classes_.tolist())

attraction_types_map = get_id_to_name_map('Type', 'AttractionTypeId', 'AttractionType')
attraction_types = sorted(attraction_types_map.values())

cities_map = get_id_to_name_map('City', 'CityId', 'CityName')
all_city_names = sorted(cities_map.values())

continents_map = get_id_to_name_map('Continent', 'ContinentId', 'Continent')
all_continent_names = sorted(continents_map.values())

countries_map = get_id_to_name_map('Country', 'CountryId', 'Country')
all_country_names = sorted(countries_map.values())

regions_map = get_id_to_name_map('Region', 'RegionId', 'Region')
all_region_names = sorted(regions_map.values())

attractions_df = raw_dfs_map.get('Item')
if attractions_df is not None:
    attractions_df['Attraction'] = attractions_df['Attraction'].astype(str)
    attractions_list = attractions_df[['AttractionId', 'Attraction']].drop_duplicates().sort_values('Attraction')
    attraction_name_to_id = attractions_list.set_index('Attraction')['AttractionId'].to_dict()
    # CORRECTED LINE 471: Missing closing bracket and .to_dict()
    attraction_id_to_name = attractions_list.set_index('AttractionId')['Attraction'].to_dict() 
else:
    st.warning("Item data not loaded, recommendation system might be limited.")
    attraction_name_to_id = {}
    attraction_id_to_name = {}
    attractions_list = pd.DataFrame(columns=['AttractionId', 'Attraction'])


numerical_features_used = [
    'UserAvgRating', 'AttractionAvgRating', 'UserVisitCount', 'AttractionVisitCount'
]

# --- Streamlit UI ---
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a slightly more attractive look
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    color: #4CAF50; /* A nice green */
    text-align: center;
    margin-bottom: 0.5em;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1em;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    border: 1px solid #4CAF50;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #45a049;
    border: 1px solid #45a049;
}
.stTextInput>div>div>input, .stSelectbox>div>div>div>div {
    border-radius: 5px;
    border: 1px solid #ccc;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-header'>🗺️ Tourism Experience Analytics</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
Welcome to the **Tourism Experience Analytics Platform!** This application leverages machine learning to
help tourism agencies and travelers gain valuable insights, predict user behavior, and provide personalized
recommendations.
""", unsafe_allow_html=True)
st.markdown("---")

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics Dashboard", "🔮 Predict Visit Mode", "⭐ Predict Attraction Rating", "🌟 Get Recommendations"]) # NEW TAB HERE

with tab1:
    st.header("📊 Explore Tourism Trends")
    st.markdown("""
    Dive into key insights and visualizations derived from tourism data. Understand popular attractions,
    user demographics, and visit patterns across different regions.
    """)

    st.subheader("Visualizations")
    eda_plots = [f for f in os.listdir(EDA_PLOT_DIR) if f.endswith('.png')]
    
    if eda_plots:
        st.info(f"Displaying {len(eda_plots)} pre-generated analytics plots.")
        col1, col2 = st.columns(2)
        for i, plot_file in enumerate(sorted(eda_plots)):
            with (col1 if i % 2 == 0 else col2):
                with st.expander(f"View Plot: {plot_file.replace('_', ' ').replace('.png', '').title()}", expanded=False):
                    st.image(os.path.join(EDA_PLOT_DIR, plot_file), caption=plot_file.replace('_', ' ').replace('.png', ''), use_column_width=True)
    else:
        st.warning("No EDA plots found. Please ensure `EDA.py` was run and plots were saved.")


with tab2:
    st.header("🔮 Predict User's Visit Mode")
    st.markdown("""
    Based on user demographics and attraction features, this model predicts the most likely
    mode of visit (e.g., Business, Family, Couples, Friends). This can help tailor marketing
    campaigns and resource planning.
    """)

    with st.form("predict_visit_mode_form", clear_on_submit=False):
        st.subheader("User & Attraction Context")
        
        col_user, col_attr = st.columns(2)
        with col_user:
            st.markdown("#### User Demographics")
            user_continent_name = st.selectbox("Continent 🌍", options=[''] + all_continent_names, key="clf_continent")
            user_country_name = st.selectbox("Country 🗺️", options=[''] + all_country_names, key="clf_country")
            user_region_name = st.selectbox("Region 📍", options=[''] + all_region_names, key="clf_region")
            user_city_name = st.selectbox("City 🏙️", options=[''] + all_city_names, key="clf_city")

        with col_attr:
            st.markdown("#### Attraction Details")
            attraction_type_name = st.selectbox("Attraction Type 🏛️", options=[''] + attraction_types, key="clf_attraction_type")
            attraction_city_name = st.selectbox("Attraction City 🌆", options=[''] + all_city_names, key="clf_attraction_city")
            visit_month = st.selectbox("Visit Month 🗓️", options=[''] + sorted([pd.to_datetime(m, format='%m').strftime('%B') for m in range(1,13)]), key="clf_month")

        st.markdown("#### Simulated User & Attraction History (for demonstration)")
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            user_avg_rating = st.slider("User's Avg Rating (1-5) ⭐", 1.0, 5.0, 3.5, 0.1, key="clf_user_avg_rating")
            user_visit_count = st.number_input("User's Total Visits (#)", min_value=0, value=5, key="clf_user_visit_count")
        with col_sim2:
            attraction_avg_rating = st.slider("Attraction's Avg Rating (1-5) ✨", 1.0, 5.0, 3.5, 0.1, key="clf_attraction_avg_rating")
            attraction_visit_count = st.number_input("Attraction's Total Visits (#)", min_value=0, value=100, key="clf_attraction_visit_count")
        
        st.markdown("---")
        submitted_clf = st.form_submit_button("🚀 Predict Visit Mode")

        if submitted_clf:
            if not all([user_continent_name, user_country_name, user_region_name, user_city_name,
                        attraction_type_name, attraction_city_name, visit_month]):
                st.error("❗ Please select a value for ALL input fields to get a prediction.")
            else:
                with st.spinner("Predicting..."):
                    input_df_clf = pd.DataFrame(index=[0])

                    input_df_clf['UserAvgRating'] = user_avg_rating
                    input_df_clf['AttractionAvgRating'] = attraction_avg_rating
                    input_df_clf['UserVisitCount'] = user_visit_count
                    input_df_clf['AttractionVisitCount'] = attraction_visit_count

                    # Create a template of all OHE columns from clf_features
                    ohe_cols_template = pd.DataFrame(0, index=[0], columns=clf_features)
                    input_df_clf = pd.concat([input_df_clf, ohe_cols_template], axis=1)
                    input_df_clf = input_df_clf.loc[:,~input_df_clf.columns.duplicated()].copy()

                    # Set 1 for selected OHE features
                    if f'MonthName_{visit_month}' in input_df_clf.columns:
                        input_df_clf[f'MonthName_{visit_month}'] = 1
                    if f'AttractionType_{attraction_type_name}' in input_df_clf.columns:
                         input_df_clf[f'AttractionType_{attraction_type_name}'] = 1
                    # City/Attraction City/Attraction are not OHE features in optimized pipeline
                    # Only include if they are part of `clf_features`
                    if f'UserCityName_{user_city_name}' in input_df_clf.columns and f'UserCityName_{user_city_name}' in clf_features:
                        input_df_clf[f'UserCityName_{user_city_name}'] = 1
                    if f'AttractionCityName_{attraction_city_name}' in input_df_clf.columns and f'AttractionCityName_{attraction_city_name}' in clf_features:
                         input_df_clf[f'AttractionCityName_{attraction_city_name}'] = 1
                    if f'Continent_{user_continent_name}' in input_df_clf.columns: # Continent IS OHE in optimized
                        input_df_clf[f'Continent_{user_continent_name}'] = 1
                    if f'UserCountry_{user_country_name}' in input_df_clf.columns: # UserCountry IS OHE in optimized
                        input_df_clf[f'UserCountry_{user_country_name}'] = 1
                    if f'UserRegion_{user_region_name}' in input_df_clf.columns: # UserRegion IS OHE in optimized
                        input_df_clf[f'UserRegion_{user_region_name}'] = 1
                    
                    # Ensure columns are in the exact order as clf_features (critical for scikit-learn models)
                    X_input_clf = input_df_clf[clf_features]
                    
                    for col in numerical_features_used:
                        if col in X_input_clf.columns:
                            X_input_clf.loc[:, col] = X_input_clf[col].astype(float)
                    X_input_clf.loc[:, numerical_features_used] = reg_scaler.transform(X_input_clf[numerical_features_used])


                    try:
                        predicted_mode_encoded = clf_model.predict(X_input_clf)[0]
                        predicted_mode = clf_label_encoder.inverse_transform([predicted_mode_encoded])[0]
                        st.success(f"🎉 Predicted Visit Mode: **{predicted_mode}**")
                        st.info(f"*(Note: Model Accuracy for classification is around 50% based on evaluation.)*") 
                    except Exception as e:
                        st.error(f"Prediction failed: {e}. Please check inputs and ensure model compatibility.")
                        st.write("Input features shape:", X_input_clf.shape)
                        st.write("Model expected features shape (from clf_features):", len(clf_features))
                        st.write("Missing features in input (compared to expected):")
                        st.write(set(clf_features) - set(X_input_clf.columns))
                        st.write("Extra features in input (compared to expected):")
                        st.write(set(X_input_clf.columns) - set(clf_features))

# --- NEW TAB: Predict Attraction Rating ---
with tab3: # This will be the third tab now
    st.header("⭐ Predict Attraction Rating")
    st.markdown("""
    Estimate the rating a user might give to a tourist attraction based on user and attraction characteristics.
    This can help travel platforms gauge satisfaction or identify areas for improvement.
    """)

    with st.form("predict_rating_form", clear_on_submit=False):
        st.subheader("User & Attraction Context for Rating Prediction")
        
        col_user_reg, col_attr_reg = st.columns(2)
        with col_user_reg:
            st.markdown("#### User Demographics")
            user_continent_name_reg = st.selectbox("Continent 🌍", options=[''] + all_continent_names, key="reg_continent")
            user_country_name_reg = st.selectbox("Country 🗺️", options=[''] + all_country_names, key="reg_country")
            user_region_name_reg = st.selectbox("Region 📍", options=[''] + all_region_names, key="reg_region")
            user_city_name_reg = st.selectbox("City 🏙️", options=[''] + all_city_names, key="reg_city")

        with col_attr_reg:
            st.markdown("#### Attraction Details")
            attraction_type_name_reg = st.selectbox("Attraction Type 🏛️", options=[''] + attraction_types, key="reg_attraction_type")
            attraction_city_name_reg = st.selectbox("Attraction City 🌆", options=[''] + all_city_names, key="reg_attraction_city")
            visit_month_reg = st.selectbox("Visit Month 🗓️", options=[''] + sorted([pd.to_datetime(m, format='%m').strftime('%B') for m in range(1,13)]), key="reg_month")

        st.markdown("#### Simulated User & Attraction History (for demonstration)")
        col_sim1_reg, col_sim2_reg = st.columns(2)
        with col_sim1_reg:
            user_avg_rating_reg = st.slider("User's Avg Rating (1-5) ⭐", 1.0, 5.0, 3.5, 0.1, key="reg_user_avg_rating")
            user_visit_count_reg = st.number_input("User's Total Visits (#)", min_value=0, value=5, key="reg_user_visit_count")
        with col_sim2_reg:
            attraction_avg_rating_reg = st.slider("Attraction's Avg Rating (1-5) ✨", 1.0, 5.0, 3.5, 0.1, key="reg_attraction_avg_rating")
            attraction_visit_count_reg = st.number_input("Attraction's Total Visits (#)", min_value=0, value=100, key="reg_attraction_visit_count")
        
        st.markdown("---")
        submitted_reg = st.form_submit_button("💡 Predict Rating")

        if submitted_reg:
            if not all([user_continent_name_reg, user_country_name_reg, user_region_name_reg, user_city_name_reg,
                        attraction_type_name_reg, attraction_city_name_reg, visit_month_reg]):
                st.error("❗ Please select a value for ALL input fields to get a prediction.")
            else:
                with st.spinner("Predicting rating..."):
                    input_df_reg = pd.DataFrame(index=[0])

                    input_df_reg['UserAvgRating'] = user_avg_rating_reg
                    input_df_reg['AttractionAvgRating'] = attraction_avg_rating_reg
                    input_df_reg['UserVisitCount'] = user_visit_count_reg
                    input_df_reg['AttractionVisitCount'] = attraction_visit_count_reg

                    # Create a template of all OHE columns from reg_features
                    ohe_cols_template_reg = pd.DataFrame(0, index=[0], columns=reg_features)
                    input_df_reg = pd.concat([input_df_reg, ohe_cols_template_reg], axis=1)
                    input_df_reg = input_df_reg.loc[:,~input_df_reg.columns.duplicated()].copy()

                    # Set 1 for selected OHE features
                    if f'MonthName_{visit_month_reg}' in input_df_reg.columns:
                        input_df_reg[f'MonthName_{visit_month_reg}'] = 1
                    if f'AttractionType_{attraction_type_name_reg}' in input_df_reg.columns:
                         input_df_reg[f'AttractionType_{attraction_type_name_reg}'] = 1
                    # Handle City/Attraction City OHE only if they are actually in reg_features
                    if f'UserCityName_{user_city_name_reg}' in input_df_reg.columns and f'UserCityName_{user_city_name_reg}' in reg_features:
                        input_df_reg[f'UserCityName_{user_city_name_reg}'] = 1
                    if f'AttractionCityName_{attraction_city_name_reg}' in input_df_reg.columns and f'AttractionCityName_{attraction_city_name_reg}' in reg_features:
                         input_df_reg[f'AttractionCityName_{attraction_city_name_reg}'] = 1
                    if f'Continent_{user_continent_name_reg}' in input_df_reg.columns:
                        input_df_reg[f'Continent_{user_continent_name_reg}'] = 1
                    if f'UserCountry_{user_country_name_reg}' in input_df_reg.columns:
                        input_df_reg[f'UserCountry_{user_country_name_reg}'] = 1
                    if f'UserRegion_{user_region_name_reg}' in input_df_reg.columns:
                        input_df_reg[f'UserRegion_{user_region_name_reg}'] = 1
                    
                    # Ensure columns are in the exact order as reg_features
                    X_input_reg = input_df_reg[reg_features]
                    
                    for col in numerical_features_used:
                        if col in X_input_reg.columns:
                            X_input_reg.loc[:, col] = X_input_reg[col].astype(float)
                    X_input_reg.loc[:, numerical_features_used] = reg_scaler.transform(X_input_reg[numerical_features_used])

                    try:
                        predicted_rating_value = reg_model.predict(X_input_reg)[0]
                        st.success(f"📈 Predicted Attraction Rating: **{predicted_rating_value:.2f} out of 5** ⭐")
                        st.info(f"*(Note: Model RMSE for regression is around 0.50 based on evaluation.)*")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}. Please check inputs and ensure model compatibility.")
                        st.write("Input features shape:", X_input_reg.shape)
                        st.write("Model expected features shape (from reg_features):", len(reg_features))
                        st.write("Missing features in input (compared to expected):")
                        st.write(set(reg_features) - set(X_input_reg.columns))
                        st.write("Extra features in input (compared to expected):")
                        st.write(set(X_input_reg.columns) - set(reg_features))


# --- Original Tab 3 (now Tab 4): Get Recommendations ---
with tab4: # This will be the fourth tab now
    st.header("🌟 Personalized Attraction Recommendations")
    st.markdown("""
    Discover new attractions tailored to your preferences! Our system suggests places you'll love
    based on your past visits and similar users' choices.
    """)

    st.subheader("Find Recommendations by User ID")
    col_user_id, col_filler = st.columns([0.4, 0.6])
    with col_user_id:
        sample_user_id_default = df['UserId'].sample(1).iloc[0] if not df['UserId'].empty else 1 
        target_user_id = st.number_input("Enter a User ID 👤", min_value=1, value=int(sample_user_id_default), help="Try a User ID from the dataset (e.g., " + str(int(sample_user_id_default)) + ").")
        get_rec_button = st.button("✨ Get Recommendations")

    if get_rec_button:
        if target_user_id:
            st.info(f"Generating recommendations for User ID: **{target_user_id}**...")

            with st.spinner("Finding the best attractions for you..."):
                full_ratings_df = df[['UserId', 'AttractionId', 'Rating']]

                user_visited_attractions = full_ratings_df[full_ratings_df['UserId'] == target_user_id]['AttractionId'].unique().tolist()
                
                if len(user_visited_attractions) == 0:
                    st.warning(f"😔 User {target_user_id} has no historical visits in our dataset. Cannot provide personalized recommendations based on history.")
                    st.write("However, here are some generally highly-rated attractions across the dataset:")
                    if 'AttractionAvgRating' in df.columns and 'Attraction' in df.columns:
                        # Need to get Attraction Name from original item data, not df
                        attraction_details = raw_dfs_map['Item'][['AttractionId', 'Attraction']].drop_duplicates()
                        top_attractions_by_avg_rating = df.groupby('AttractionId')['Rating'].mean().reset_index()
                        top_attractions_by_avg_rating = pd.merge(top_attractions_by_avg_rating, attraction_details, on='AttractionId', how='left')
                        top_attractions_by_avg_rating = top_attractions_by_avg_rating.sort_values(by='Rating', ascending=False).head(10)
                        
                        # Format for display
                        display_data = []
                        for i, row in top_attractions_by_avg_rating.iterrows():
                            display_data.append({"Rank": len(display_data)+1, "Attraction Name": row['Attraction'], "Average Rating": f"{row['Rating']:.2f} ⭐"})
                        st.dataframe(pd.DataFrame(display_data).set_index('Rank'), use_container_width=True)
                    else:
                        st.info("No general popular attractions data available.")

                else:
                    all_attraction_ids = full_ratings_df['AttractionId'].unique().tolist()

                    predictions = []
                    for attr_id in all_attraction_ids:
                        if attr_id not in user_visited_attractions:
                            try:
                                predicted_rating = rec_model.predict(str(target_user_id), str(attr_id)).est
                                predictions.append((attr_id, predicted_rating))
                            except KeyError:
                                continue 
                            except ValueError:
                                continue

                    predictions.sort(key=lambda x: x[1], reverse=True)

                    st.subheader(f"Top Recommended Attractions for User {target_user_id}:")
                    if predictions:
                        recommendation_data = []
                        for i, (attr_id, predicted_rating) in enumerate(predictions[:10]):
                            attr_name = attraction_id_to_name.get(attr_id, f"Attraction ID: {attr_id} (Name not found)")
                            recommendation_data.append({"Rank": i+1, "Attraction Name": attr_name, "Predicted Rating": f"{predicted_rating:.2f} ⭐"})
                        
                        st.dataframe(pd.DataFrame(recommendation_data).set_index('Rank'), use_container_width=True)

                        st.markdown("---")
                        st.subheader("Attractions already visited by this user:")
                        visited_names = [attraction_id_to_name.get(aid, f"Attraction ID: {aid}") for aid in user_visited_attractions]
                        st.write(", ".join(visited_names))
                    else:
                        st.info("No new attractions to recommend for this user. Perhaps they have visited most, or the model cannot find relevant recommendations.")
        else:
          st.warning("Please enter a valid User ID.")