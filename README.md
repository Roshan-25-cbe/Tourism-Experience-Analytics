# 🗺️ Tourism Experience Analytics: Explore, Predict & Recommend

## ✨ Introduction

This project delivers an interactive platform for the tourism industry, powered by machine learning and data analytics. It allows users to **explore tourism trends**, **predict user behavior**, and receive **personalized attraction recommendations**, aiming to enhance traveler experiences and inform business strategies.

## 🚀 Key Features & Objectives

* **Interactive Analytics Dashboard:** Visualize popular attractions, user demographics, and travel patterns across regions.
* **Predict User Visit Mode:** Forecast the purpose of a user's visit (e.g., Business, Family, Couples, Friends) based on their profile and trip details.
* **Predict Attraction Rating:** Estimate the rating a user is likely to give an attraction, helping gauge satisfaction and identify areas for improvement.
* **Personalized Recommendations:** Suggest attractions tailored to individual user preferences and historical data.

## 📊 Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application.
* **Pandas, NumPy:** For efficient data manipulation and numerical operations.
* **Scikit-learn:** For machine learning model development (preprocessing, regression, classification).
* **LightGBM (`LGBMRegressor`):** A high-performance gradient boosting framework used for predicting attraction ratings.
* **RandomForest (`RandomForestClassifier`):** A robust ensemble learning method used for predicting visit modes.
* **Scikit-Surprise (`SVD`):** A specialized library for building and evaluating recommendation systems.
* **Matplotlib, Seaborn:** For data visualization in EDA.

## 📂 Project Structure

The project is logically organized into several Python scripts for clarity and modularity:

* `data_loading.py`: Loads raw Excel datasets and converts them to optimized `.pkl` format.
* `data_preprocessing.py`: Cleans, preprocesses, merges data, performs feature engineering, and applies optimized one-hot encoding. Saves processed data (`.pkl` and a viewable `.xlsx` sample).
* `eda.py`: Conducts Exploratory Data Analysis, generating and saving insightful visualizations.
* `model_training.py`: Trains the machine learning models (Regression, Classification, Recommendation) and saves them along with necessary components (scalers, feature lists, label encoder).
* `model_evaluation.py`: Evaluates the performance of the trained models and generates detailed reports.
* `app.py`: The main Streamlit application, serving as the interactive user interface.
* `raw_data_pkl/`: Directory for raw data in `.pkl` format. (Auto-generated)
* `processed_data/`: Directory for cleaned and processed data. (Auto-generated)
* `eda_plot/`: Directory for saved EDA visualizations. (Auto-generated)
* `models/`: Directory for trained machine learning models and supporting files. (Auto-generated)
* `evaluation_results/`: Directory for model performance reports. (Auto-generated)
* `.gitignore`: Specifies files and directories to be excluded from version control.
* `requirements.txt`: Lists all Python libraries required to run the project.

## ⚙️ How to Run Locally

Follow these steps to set up and launch the project on your machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Roshan-25-cbe/Tourism-Experience-Analytics.git](https://github.com/Roshan-25-cbe/Tourism-Experience-Analytics.git)
    cd Tourism-Experience-Analytics
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    ```
    * **Windows (PowerShell/CMD):** `.\.venv\Scripts\activate`
    * **macOS/Linux (Bash/Zsh):** `source ./.venv/bin/activate`

3.  **Place Raw Data Files:**
    Ensure all your original Excel dataset files (`City.xlsx`, `Continent.xlsx`, `Country.xlsx`, `Updated_Item.xlsx`, `Mode.xlsx`, `Region.xlsx`, `Transaction.xlsx`, `Type.xlsx`, `User.xlsx`) are placed directly in the project's root directory.

4.  **Install Dependencies:**
    With your virtual environment active, install all required Python libraries.
    ```bash
    pip install -r requirements.txt
    ```
    * *If you encounter `numpy` compatibility issues (e.g., `AttributeError: float object has no attribute 'rmse'` or `ImportError: numpy.core.multiarray failed to import`), try reinstalling `numpy` and `scikit-surprise` specifically:*
        ```bash
        pip uninstall scikit-surprise numpy -y
        pip install numpy==1.26.4 # Compatible version
        pip install scikit-surprise
        ```

5.  **Execute Project Pipeline:**
    Run the following scripts sequentially from your terminal (with the virtual environment activated) to process data, train models, and evaluate:

    ```bash
    python data_loading.py
    python data_preprocessing.py
    python eda.py
    python model_training.py
    python model_evaluation.py
    ```

6.  **Launch the Streamlit Application:**
    Once all the above scripts have completed successfully, launch the interactive dashboard:
    ```bash
    streamlit run app.py
    ```
    This command will automatically open the application in your default web browser (usually `http://localhost:8501`).

## 📈 Model Performance Highlights

*(Metrics based on evaluation on a test set)*

* **Regression Model (LGBMRegressor) for Attraction Rating:**
    * Mean Squared Error (MSE): **0.2450**
    * Root Mean Squared Error (RMSE): **0.4950** (Average prediction error of ~0.5 stars on a 1-5 scale)
    * R-squared (R2): **0.7398** (Explains ~74% of rating variance)

* **Classification Model (RandomForestClassifier) for Visit Mode:**
    * Accuracy: **0.4975** (~50% correct predictions)
    * Precision (weighted): **0.4857**
    * Recall (weighted): **0.4975**
    * F1-Score (weighted): **0.4838**
    * *Note: The classification model's accuracy, while functional, indicates challenges due to significant class imbalance and dataset complexity. Performance varies across different visit modes. Further optimization would focus here.*

* **Recommendation System (SVD):**
    * Root Mean Squared Error (RMSE): **0.9244**
    * Mean Absolute Error (MAE): **0.7186**

## 💡 Future Enhancements

* **Hyperparameter Optimization:** Conduct a more exhaustive search for optimal model parameters to boost accuracy.
* **Advanced Imbalance Handling:** Implement sophisticated techniques like SMOTE or ADASYN for the classification task.
* **Feature Enrichment:** Explore creating more complex features or integrating external data sources.
* **Alternative Algorithms:** Experiment with deep learning models or other specialized algorithms for further performance gains.
* **Scalability:** Adapt the pipeline for even larger datasets, potentially using Dask or Spark.
* **Real-time Recommendations:** Integrate with a database and build a system for real-time recommendation generation.
* **User Interface Refinements:** Further enhance the Streamlit UI with more interactive plots, user profiles, and personalized dashboards.

## 🧑‍💻 Project Author

* **Roshan**
* GitHub: [https://github.com/Roshan-25-cbe/Tourism-Experience-Analytics.git](https://github.com/Roshan-25-cbe/Tourism-Experience-Analytics.git)
* LinkedIn: [www.linkedin.com/in/roshan-angamuthu-195ba230a](www.linkedin.com/in/roshan-angamuthu-195ba230a)

## 📧 Contact

For any inquiries, collaboration opportunities, or feedback, feel free to connect:
* Email: [roshana36822@gmail.com](mailto:roshana36822@gmail.com)
