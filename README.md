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




