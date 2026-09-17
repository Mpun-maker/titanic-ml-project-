# 🚢 Titanic Survival Prediction

An end-to-end machine learning project that predicts passenger survival using the classic Titanic dataset.

This project demonstrates a complete machine learning workflow — from problem definition and exploratory data analysis to feature engineering, preprocessing, model comparison, hyperparameter tuning, repeated cross-validation, error analysis, model serialization, automated testing, and Streamlit deployment.

## 🌐 Live Demo

**Streamlit Application:**
`https://nwxhjrlz62askfshgsicx8.streamlit.app/`

---

## 📌 Project Overview

The objective of this project is to build a machine learning classification system that predicts whether a Titanic passenger survived based on passenger and travel-related information.

Rather than focusing only on model training, the project follows a production-oriented workflow that includes:

* Problem understanding
* Exploratory data analysis
* Data preprocessing
* Feature engineering
* Train/test splitting
* Multiple model development
* Hyperparameter tuning
* Repeated cross-validation
* Model evaluation
* Error analysis
* Model serialization
* Prediction pipeline
* Automated testing
* Streamlit deployment
* Project documentation

---

## 🎯 Problem Statement

The Titanic dataset contains information about passengers aboard the RMS Titanic, including demographic and travel-related attributes.

The machine learning task is:

> **Given the available passenger information, predict whether the passenger survived.**

### Target Variable

`Survived`

| Value | Meaning         |
| ----: | --------------- |
|   `0` | Did not survive |
|   `1` | Survived        |

This is a **binary classification problem**.

---

## 📊 Dataset

The project uses the Titanic training dataset from Kaggle.

The original dataset contains **891 passenger records** and includes information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Port of embarkation
* Name
* Cabin
* Ticket

### Missing Values

The main missing-value challenges were:

* `Age`
* `Embarked`
* `Cabin`

`Cabin` contains substantial missing information, so the project extracts available deck information while representing missing cabin information as `Unknown`.

---

# 🔎 Exploratory Data Analysis

Exploratory data analysis was performed to understand:

* Target distribution
* Numerical feature distributions
* Categorical feature distributions
* Survival patterns
* Missing values
* Relationships between passenger characteristics and survival

Some of the major variables investigated included:

* `Sex`
* `Age`
* `Pclass`
* `Fare`
* `SibSp`
* `Parch`
* `Embarked`
* `Name`
* `Cabin`
* `Ticket`

### Example EDA

![EDA Visualization](screenshots/eda-survival.png)

---

# 🛠️ Feature Engineering

Additional features were created from the original variables to capture potentially useful passenger and ticket-level information.

| Feature           | Description                                         |
| ----------------- | --------------------------------------------------- |
| `FamilySize`      | Total number of family members traveling together   |
| `IsAlone`         | Indicates whether the passenger was traveling alone |
| `Title`           | Extracted passenger title from the name             |
| `Deck`            | First character of the cabin information            |
| `TicketGroupSize` | Number of passengers sharing the same ticket        |
| `FarePerPerson`   | Fare divided by the ticket group size               |
| `FareGroup`       | Quartile-based group of fare per person             |

### Example

For family-related features:

```python
dataframe['FamilySize'] = (
    dataframe['SibSp']
    + dataframe['Parch']
    + 1
)

dataframe['IsAlone'] = (
    dataframe['FamilySize'] == 1
).astype(int)
```

For ticket-level information:

```python
dataframe['TicketGroupSize'] = (
    dataframe
    .groupby('Ticket')['Ticket']
    .transform('count')
)
```

And:

```python
dataframe['FarePerPerson'] = (
    dataframe['Fare']
    / dataframe['TicketGroupSize']
)
```

These engineered variables were designed to capture information that is not directly represented by the original individual columns.

---

# 🧹 Data Preprocessing

Separate preprocessing pipelines were created for numerical and categorical features.

### Numerical Features

Numerical missing values were handled using median imputation, followed by standardization.

```text
Numerical Features
        ↓
Median Imputation
        ↓
Standard Scaling
```

### Categorical Features

Categorical missing values were handled using the most frequent category, followed by one-hot encoding.

```text
Categorical Features
        ↓
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The preprocessing was implemented using Scikit-learn's `Pipeline` and `ColumnTransformer`.

This ensures that the same transformations are consistently applied during both training and prediction.

---

# 🤖 Machine Learning Models

Four classification algorithms were developed and compared:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**

Logistic Regression was used as an interpretable baseline, while tree-based models were evaluated to capture potentially non-linear relationships.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV`.

The models were tuned using:

* 5-fold cross-validation
* ROC-AUC as the optimization metric
* Parallel processing with `n_jobs=-1`

This allowed the models to be compared using a consistent optimization strategy.

---

# 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

ROC-AUC was used as the primary metric for model comparison because it evaluates the model's ability to distinguish between the two classes across classification thresholds.

---

## 🔁 Repeated Cross-Validation

To obtain a more robust estimate of model performance, repeated stratified cross-validation was performed using:

* **5 folds**
* **10 repeats**
* **50 validation runs in total**

### Results

| Model                 | Mean ROC-AUC | Std ROC-AUC |
| --------------------- | -----------: | ----------: |
| **Gradient Boosting** |   **0.8960** |      0.0278 |
| Random Forest         |       0.8878 |      0.0338 |
| Logistic Regression   |       0.8650 |      0.0332 |
| Decision Tree         |       0.8561 |      0.0325 |

### Model Selection

Logistic Regression was retained as an interpretable baseline.

**Gradient Boosting was selected as the final predictive model based on its repeated cross-validation performance.**

The final model was then retrained on the training data and saved for deployment.

![Model Performance](screenshots/model-performance.png)

---

# 🔍 Model Interpretation

Feature importance analysis was performed to understand which engineered and original variables contributed most strongly to the tree-based models.

For Gradient Boosting, the most influential feature groups included:

| Feature         | Importance |
| --------------- | ---------: |
| `Title`         |     0.2941 |
| `Sex`           |     0.2051 |
| `Age`           |     0.0904 |
| `FarePerPerson` |     0.0871 |
| `Pclass`        |     0.0860 |
| `Fare`          |     0.0665 |
| `Deck`          |     0.0611 |

These values describe the model's feature importance and should not be interpreted as causal effects.

---

# 🧪 Error Analysis

Error analysis was performed to understand where the model made incorrect predictions.

The analysis examined errors across variables such as:

* Sex
* Passenger class
* Title
* Deck
* Fare group
* Family status

For example, the analysis showed that false negatives were concentrated more heavily among male passengers, while false positives included a larger number of female passengers.

The analysis also highlighted the limitations of features with substantial missing information, particularly cabin-related information.

Error analysis was used to identify potential areas for future feature engineering and model improvement.

---

# 💾 Model Serialization

The final Gradient Boosting pipeline was saved using `joblib`.

```python
joblib.dump(
    final_model,
    'models/titanic_gradient_boosting_pipeline.pkl'
)
```

Additional inference metadata was saved separately:

```text
models/
├── titanic_gradient_boosting_pipeline.pkl
└── titanic_metadata.pkl
```

The metadata contains information required to reproduce engineered features during prediction, including:

* Ticket group sizes
* Fare group boundaries

---

# 🔮 Prediction Pipeline

A reusable feature-generation module was created in:

```text
src/features.py
```

The prediction pipeline performs feature engineering on new passenger information before passing it to the trained model.

The process is:

```text
Passenger Input
       ↓
Input Validation
       ↓
Feature Engineering
       ↓
Preprocessing Pipeline
       ↓
Gradient Boosting Model
       ↓
Prediction
       ↓
Estimated Survival Probability
```

This separates application logic from the Streamlit interface and makes the feature-generation process reusable.

---

# 🖥️ Streamlit Application

The trained model was integrated into an interactive Streamlit application.

Users can enter passenger information such as:

* Passenger class
* Sex
* Age
* Siblings/spouses
* Parents/children
* Fare
* Embarkation port
* Name
* Cabin
* Ticket

The application validates the input, generates the required features, and returns:

* Predicted survival class
* Estimated survival probability

### Application Preview

![Streamlit Application](screenshots/streamlit-home.png)

### Prediction Result

![Prediction Result](screenshots/prediction-result.png)

---

# 🧪 Testing

Automated tests were added using `pytest`.

The test suite covers important parts of the application, including:

### Feature Engineering

* Correct feature generation
* Family size calculation
* `IsAlone`
* Title extraction
* Deck extraction
* Ticket group size
* Fare per person

### Input Validation

* Valid passenger input
* Invalid passenger class
* Invalid sex
* Invalid age
* Invalid fare
* Invalid embarkation port
* Invalid name format
* Invalid cabin format

### Model Testing

* Model loading
* Metadata loading
* Prediction output shape
* Probability output shape
* Valid prediction values

Run the tests with:

```bash
pytest
```

---

# 📁 Project Structure

```text
Titanic-ML-Project/
│
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
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
├── screenshots/
│   ├── streamlit-home.png
│   ├── prediction-result.png
│   ├── eda-survival.png
│   └── model-performance.png
│
├── src/
│   ├── __init__.py
│   └── features.py
│
└── tests/
    ├── test_features.py
    └── test_model.py
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-ml-project.git
```

```bash
cd titanic-ml-project
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project pins the Scikit-learn version used to serialize the trained model:

```text
scikit-learn==1.6.1
```

Keeping the compatible Scikit-learn version helps prevent model-loading compatibility issues.

---

## 4. Run the tests

```bash
pytest
```

---

## 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:**
`https://nwxhjrlz62askfshgsicx8.streamlit.app/`

The application uses the trained model and metadata stored in the repository to generate predictions for new passenger inputs.

---

# ⚠️ Limitations

This project has several limitations:

* The dataset represents a historical event and should not be interpreted as a general-purpose survival model.
* Some original variables contain substantial missing information.
* `Cabin` information is incomplete for many passengers.
* Ticket-level features depend on information available in the training dataset.
* The model's predicted probability is an estimated model output, not a guaranteed probability of an individual's actual outcome.
* The model is intended as a portfolio and educational machine learning project rather than a real-world decision-making system.

---

# 🔮 Future Improvements

Potential improvements include:

* Develop a unified custom transformer for all feature engineering.
* Improve inference-time handling of unseen ticket groups.
* Explore additional feature engineering strategies.
* Perform more systematic feature selection.
* Investigate probability calibration.
* Add SHAP-based model explanations.
* Add more comprehensive automated tests.
* Add application logging.
* Introduce API-based model serving.
* Add monitoring for deployed predictions.
* Containerize the application using Docker.
* Add continuous integration with GitHub Actions.

---

# 🧰 Technologies

| Category          | Tools                     |
| ----------------- | ------------------------- |
| Programming       | Python                    |
| Data Manipulation | Pandas, NumPy             |
| Visualization     | Matplotlib, Seaborn       |
| Machine Learning  | Scikit-learn              |
| Model Persistence | Joblib                    |
| Application       | Streamlit                 |
| Testing           | Pytest                    |
| Development       | Jupyter Notebook, VS Code |
| Version Control   | Git, GitHub               |
| Deployment        | Streamlit Community Cloud |

---

# 📚 Key Skills Demonstrated

This project demonstrates practical experience with:

* Python programming
* Data cleaning
* Exploratory data analysis
* Feature engineering
* Missing-value handling
* Categorical encoding
* Feature scaling
* Scikit-learn pipelines
* ColumnTransformer
* Binary classification
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Repeated stratified cross-validation
* ROC-AUC evaluation
* Model interpretation
* Error analysis
* Model serialization
* Prediction pipelines
* Automated testing
* Streamlit development
* Git and GitHub
* Machine learning deployment

---

# 🎓 Project Purpose

This project was developed as a practical demonstration of an end-to-end machine learning workflow.

The primary goal was not simply to achieve a high score on the Titanic dataset, but to understand and implement the complete process required to take a machine learning problem from **raw data to a tested and deployed application**.

---

## 👤 Author

**Milan Pun**

GitHub: `https://github.com/Mpun-maker`

LinkedIn: `https://www.linkedin.com/in/milan-punmp/`

---

## ⭐ Acknowledgements

Dataset: **Titanic — Machine Learning from Disaster**, provided through Kaggle.

---

## 📄 License

This project is intended for educational and portfolio purposes.





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
