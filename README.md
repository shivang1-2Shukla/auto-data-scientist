# Auto Data Scientist

![Platform Interface Overview](frontend/style.css)

**Auto Data Scientist** is an end-to-end autonomous Machine Learning platform. It acts as a "Data Scientist in a Box," enabling users with zero coding experience to upload raw data, automatically train machine learning models, and generate predictions on new data via a professional web dashboard.

Built using an Agentic AI architecture, the platform automatically cleans data, engineers features, trains multiple algorithms simultaneously, and deploys the most accurate model through a REST API.

---

## 🌟 Key Features

*   **Autonomous Data Pipeline**: Fully automated handling of missing values, duplicate removal, one-hot encoding for categorical text, and feature scaling.
*   **AutoML Leaderboard**: The system trains multiple algorithms (e.g., Random Forest, Logistic Regression, Linear Regression) in parallel and ranks them on a leaderboard to select the best performer.
*   **Dynamic Data Explorer**: Instantly visualizes your uploaded CSV files with auto-generated bar charts and histograms.
*   **Model Accuracy Breakdown**: Provides a visual comparison of True Positives vs. Errors to prove the model's learning capability.
*   **Batch Prediction Engine**: Upload thousands of rows of new, untested data, and the platform will instantly process them, append AI predictions, and return a downloadable CSV file.
*   **Enterprise UI**: A clean, responsive AWS-inspired light theme designed for business intelligence presentations.

---

## 🏗️ Architecture

The backend is powered by a multi-agent workflow orchestrated in Python. The workflow is split into specialized algorithmic agents:

1.  **Cleaning Agent**: Identifies missing data and removes corrupt rows/duplicates.
2.  **Feature Engineering Agent**: Handles mathematical transformations like Standard Scaling for numerical data and Label/One-Hot Encoding for text categories.
3.  **AutoML Agent**: Trains, cross-validates, and tunes various Scikit-Learn algorithms based on the selected task (Classification or Regression).
4.  **Evaluation Agent**: Ranks the models based on Accuracy or RMSE scores.
5.  **Deployment Agent**: Packages the winning model and its pipelines into reusable `.pkl` artifacts and exposes them via the `/batch-predict` endpoint.

---

## 💻 Tech Stack

*   **Backend**: Python, FastAPI, Uvicorn, Pandas, Numpy, Scikit-Learn
*   **Frontend**: HTML5, Vanilla CSS (Custom Enterprise Theme), Vanilla JavaScript
*   **Visualizations**: Chart.js, PapaParse (Client-side CSV parsing)

---

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/auto-data-scientist.git
    cd auto-data-scientist
    ```

2.  **Create and activate a Virtual Environment:**
    ```bash
    python -m venv venv
    
    # On Windows:
    venv\Scripts\activate
    
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
    ```

5.  **Access the Dashboard:**
    Open your web browser and navigate to `http://localhost:8000`.

---

## 📖 Usage Guide

### 1. Training a Model
*   Navigate to `http://localhost:8000`.
*   Drag and drop your historical training data (e.g., `data/employee_attrition.csv`) into the upload zone.
*   Select the **Task Type**:
    *   *Classification*: For predicting categories (e.g., Will Quit? Yes/No)
    *   *Regression*: For predicting continuous numbers (e.g., Housing Prices).
*   Select the **Target Column** you wish to predict.
*   Click **Run Pipeline**.

### 2. Batch Prediction (Testing on New Data)
*   Once the model finishes training, scroll down to the **Batch Prediction** panel.
*   Upload a brand new CSV file that does *not* contain the target column (e.g., `data/new_employees_untested.csv`).
*   Click **Upload & Predict**.
*   The system will instantly download a new CSV file to your machine containing a newly appended `AI_Prediction` column with the results.

---

## 📁 Sample Datasets Included

To help you test the platform immediately, two sample datasets are provided in the `/data` folder:

*   **`employee_attrition.csv`**: A historical dataset of 1,000 employees. Used for **Training**.
    *   *Features*: Age, Years_At_Company, Salary, Job_Satisfaction_Score, Department
    *   *Target*: Will_Quit (1/0)
*   **`new_employees_untested.csv`**: A dataset of 100 brand new employees. The `Will_Quit` column is missing. Used for **Batch Prediction** testing.