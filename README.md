# 🚢 Titanic Survival Prediction — End-to-End Machine Learning Project

An end-to-end machine learning project that predicts whether a passenger would survive the Titanic disaster using passenger and ticket information.

The project covers the complete data science workflow, including exploratory data analysis, data preprocessing, feature engineering, model comparison, hyperparameter tuning, repeated cross-validation, error analysis, model interpretation, model serialization, automated testing, and a Streamlit prediction application.

---

## 📌 Project Overview

The goal of this project is to build a binary classification model that predicts passenger survival using the Kaggle Titanic dataset.

The target variable is:

* `Survived = 1` → Passenger survived
* `Survived = 0` → Passenger did not survive

Rather than focusing only on model accuracy, this project follows an end-to-end machine learning workflow and evaluates models using multiple metrics and repeated cross-validation.

---

## 🎯 Objectives

The main objectives of this project are to:

* Understand the Titanic dataset and its structure.
* Perform exploratory data analysis.
* Identify missing values and important patterns.
* Clean and preprocess the data.
* Engineer meaningful features.
* Compare multiple classification algorithms.
* Tune model hyperparameters.
* Evaluate model stability using repeated stratified cross-validation.
* Analyze model errors.
* Interpret important model features.
* Save and reload the trained model.
* Build a reusable prediction pipeline.
* Create a Streamlit web application.
* Add automated tests for important project components.

---

## 📊 Dataset

The project uses the **Kaggle Titanic dataset**.

The dataset contains passenger information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Port of embarkation
* Name
* Cabin
* Ticket

The prediction target is `Survived`.

---

## 🔍 Exploratory Data Analysis

The exploratory analysis examined:

* Dataset dimensions and data types
* Missing values
* Duplicate records
* Numerical feature distributions
* Categorical feature distributions
* Survival distribution
* Survival patterns by sex
* Survival patterns by passenger class
* Age and fare distributions
* Family-related patterns
* Ticket and cabin information

Some of the strongest survival patterns were associated with passenger sex and passenger class.

---

## 🛠️ Feature Engineering

Several features were created to capture additional information from the original dataset.

### Family Size

```python
FamilySize = SibSp + Parch + 1
```

This represents the total number of family members associated with a passenger.

### Is Alone

```python
IsAlone = 1
```

when `FamilySize == 1`, otherwise:

```python
IsAlone = 0
```

### Title

Titles were extracted from passenger names and grouped into meaningful categories such as:

* Mr
* Miss
* Mrs
* Master
* Rare

### Deck

The first character of the cabin value was used to represent the passenger's deck.

Passengers without cabin information were assigned:

```text
Unknown
```

### Ticket Group Size

The number of passengers sharing the same ticket was calculated.

### Fare Per Person

```python
FarePerPerson = Fare / TicketGroupSize
```

This attempts to provide a more informative representation of the fare when multiple passengers share a ticket.

### Fare Group

`FarePerPerson` was divided into four groups:

* Low
* Medium-Low
* Medium-High
* High

---

## ⚙️ Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

### Numerical Features

Missing numerical values are handled using median imputation and then standardized.

### Categorical Features

Missing categorical values are handled using most-frequent imputation and categorical variables are transformed using one-hot encoding.

A `ColumnTransformer` combines the numerical and categorical preprocessing pipelines.

This preprocessing is integrated with the machine learning models using Scikit-learn `Pipeline`.

---

## 🤖 Models

Four classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Logistic Regression was used as an interpretable baseline, while tree-based models were evaluated for their ability to capture nonlinear relationships.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV` with:

```text
Cross-validation: 5 folds
Scoring metric: ROC-AUC
```

The tuned models were then evaluated on the held-out test set.

---

## 📈 Repeated Cross-Validation

To obtain a more reliable estimate of model performance, repeated stratified cross-validation was performed using:

```text
5 folds × 10 repeats
```

### Results

| Model                 | Mean ROC-AUC | Std ROC-AUC |
| --------------------- | -----------: | ----------: |
| **Gradient Boosting** |   **0.8960** |      0.0278 |
| Random Forest         |       0.8878 |      0.0338 |
| Logistic Regression   |       0.8650 |      0.0332 |
| Decision Tree         |       0.8561 |      0.0325 |

Gradient Boosting achieved the highest mean ROC-AUC and was therefore selected as the final predictive model.

---

## 🏆 Final Model

### Gradient Boosting

The final model was selected based on repeated cross-validation performance rather than relying on a single test-set score.

The final project decision was:

> Logistic Regression was used as the interpretable baseline, while Gradient Boosting was selected as the final predictive model based on repeated cross-validation performance.

The trained model and preprocessing pipeline were serialized using `joblib`.

---

## 🔎 Model Interpretation

Feature importance analysis showed that the model relied strongly on several passenger characteristics.

For the Gradient Boosting model, the most influential grouped features included:

| Feature         | Importance |
| --------------- | ---------: |
| Title           |     0.2941 |
| Sex             |     0.2051 |
| Age             |     0.0904 |
| FarePerPerson   |     0.0871 |
| Pclass          |     0.0860 |
| Fare            |     0.0665 |
| Deck            |     0.0611 |
| TicketGroupSize |     0.0487 |
| FamilySize      |     0.0434 |

These values describe the model's relative feature importance and should not be interpreted as causal effects.

---

## 🧪 Error Analysis

Error analysis was performed on the Logistic Regression model to investigate patterns among incorrectly classified passengers.

A total of 32 test-set predictions were misclassified.

Important observations included:

* False negatives were dominated by male passengers.
* Third-class passengers represented a large portion of false negatives and false positives.
* Passengers with the `Mr` title accounted for many errors.
* `Master` passengers had a relatively high error rate, although this group was small.
* Missing cabin information (`Unknown` deck) accounted for many errors.
* `IsAlone` showed similar error rates for both groups, suggesting that it was not a major source of prediction errors.

Small categories should be interpreted cautiously because their error rates are based on relatively few observations.

---

## 💾 Model Saving

The final trained pipeline is saved as:

```text
models/titanic_gradient_boosting_pipeline.pkl
```

Prediction metadata is stored separately as:

```text
models/titanic_metadata.pkl
```

The model can therefore be loaded without retraining:

```python
import joblib

model = joblib.load(
    'models/titanic_gradient_boosting_pipeline.pkl'
)
```

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

The application allows users to enter passenger information and receive:

* Predicted survival outcome
* Estimated survival probability
* Input validation messages

The application also contains the feature engineering logic required to transform raw passenger information into the features expected by the trained model.

### Run the application

From the project root:

```bash
streamlit run app.py
```

---

## 🧪 Testing

Automated tests are included using `pytest`.

Tests cover:

* Feature engineering
* Input validation
* Model loading
* Metadata loading
* Prediction output
* Prediction probability

Run the tests with:

```bash
pytest
```

All tests should pass before deployment.

---

## 📁 Project Structure

```text
Titanic-ML-Project/
│
├── data/
│   └── train.csv
│
├── models/
│   ├── titanic_gradient_boosting_pipeline.pkl
│   └── titanic_metadata.pkl
│
├── notebooks/
│   └── Titanic_ML_Project.ipynb
│
├── src/
│   ├── __init__.py
│   └── features.py
│
├── tests/
│   ├── test_features.py
│   └── test_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---


## Live Demo

The Titanic Survival Prediction application is deployed using Streamlit Community Cloud.

**Live Application:**
`https://nwxhjrlz62askfshgsicx8.streamlit.app/`

You can enter passenger information and receive a predicted survival class along with the model's estimated survival probability.
---

---


## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Move into the project directory

```bash
cd Titanic-ML-Project
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Usage

After installing the dependencies, launch the Streamlit application:

```bash
streamlit run app.py
```

Enter passenger information in the application and click:

```text
🔮 Predict Survival
```

The application will process the input, generate the required engineered features, and return the model's prediction and estimated survival probability.

---

## ⚠️ Limitations

This project has several limitations:

* The Titanic dataset is relatively small.
* Some cabin information is missing.
* The model is trained on historical Titanic data and should not be considered a real-world survival predictor.
* Ticket-based features depend on information available in the training data.
* The current feature-engineering metadata workflow can be improved further for production use.
* Fare grouping and ticket-group information require careful handling when applying the model to completely new datasets.

There is also room to improve the separation between training-time feature fitting and inference-time transformation to further reduce the risk of data leakage.

---

## 🔮 Future Improvements

Possible improvements include:

* Build a fully unified custom transformation pipeline.
* Fit feature-engineering metadata using training data only.
* Add model explainability using SHAP.
* Add more extensive automated tests.
* Add logging and error handling.
* Improve the Streamlit interface.
* Deploy the application publicly.
* Add prediction monitoring.
* Experiment with additional machine learning algorithms.
* Perform more systematic feature selection.
* Create a production-ready inference API.

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Pytest
* Streamlit
* Jupyter Notebook

---

## 📚 Key Skills Demonstrated

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Data Preprocessing
* Classification
* Model Comparison
* Hyperparameter Tuning
* Cross-Validation
* Error Analysis
* Model Interpretation
* Model Serialization
* Automated Testing
* Streamlit Application Development
* Git/GitHub Project Organization

---

## 👨‍💻 Project Purpose

This project was developed as a portfolio project to demonstrate an end-to-end machine learning workflow, from raw data exploration to model development and deployment.

It focuses on both **machine learning performance** and **software engineering practices**, including reusable feature engineering, model serialization, testing, and application development.
