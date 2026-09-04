# UCI Adult Income Prediction

## Project Objective

The objective of this project is to build a supervised machine learning classification system that predicts whether an individual's annual income is **greater than $50K** based on demographic, educational, occupational, and financial attributes.

The project focuses on developing a reproducible machine learning workflow covering data preprocessing, feature engineering, model comparison, hyperparameter tuning, threshold selection, evaluation, model interpretation, and production-ready inference.

---

## Dataset Description

The project uses the **UCI Adult Income Dataset**, containing **48,842 observations**.

The dataset includes demographic, education, employment, and financial attributes such as:

* Age
* Workclass
* Education
* Education number
* Marital status
* Occupation
* Relationship
* Race
* Sex
* Capital gain
* Capital loss
* Hours per week
* Native country

The target variable is imbalanced, with approximately **23.93% positive-class observations**.

---

## Target Variable

The target represents annual income:

* `0` → Income ≤ $50K
* `1` → Income > $50K

The positive class (`>50K`) is the class of interest.

---

## Feature Engineering

Eight engineered features were created:

1. `age_group` — groups individuals into age ranges.
2. `hours_group` — groups weekly working hours into meaningful categories.
3. `capital_net` — calculates capital gain minus capital loss.
4. `has_capital_gain` — indicates whether an individual has capital gain.
5. `has_capital_loss` — indicates whether an individual has capital loss.
6. `education_age_ratio` — relates education level to age.
7. `work_hours_age` — combines working hours and age.
8. `is_married` — identifies married individuals.

Feature engineering was integrated into the machine learning pipeline so that the same transformations are automatically applied during inference.

---

## Preprocessing

The preprocessing workflow was implemented using a `ColumnTransformer` and included:

### Numerical Features

* Median imputation for missing values
* StandardScaler normalization

### Categorical Features

* Most-frequent imputation
* One-hot encoding
* `handle_unknown="ignore"` to safely process unseen categories during inference

The complete preprocessing workflow is included inside the final pipeline, preventing manual preprocessing during deployment.

---

## Models Tested

Three classification algorithms were evaluated using stratified 5-fold cross-validation:

* Logistic Regression
* Random Forest
* HistGradientBoostingClassifier

The models were evaluated primarily using **F1 score** and **ROC AUC**.

### Cross-Validation Results

| Model                |    Mean F1 | F1 Std | Mean ROC AUC | ROC AUC Std |
| -------------------- | ---------: | -----: | -----------: | ----------: |
| Logistic Regression  |     0.6693 | 0.0035 |       0.9127 |      0.0036 |
| Random Forest        |     0.6677 | 0.0018 |       0.9024 |      0.0023 |
| HistGradientBoosting | **0.7063** | 0.0069 |   **0.9259** |      0.0039 |

HistGradientBoosting produced the strongest cross-validation performance.

---

## Hyperparameter Tuning

Hyperparameter tuning was performed on the HistGradientBoosting model to improve predictive performance.

The selected parameters were:

```text
learning_rate = 0.0986849497
max_depth = 7
max_iter = 234
l2_regularization = 8.2443121909
random_state = 42
```

Randomness was controlled using `random_state=42` to improve reproducibility.

---

## Selected Final Model

The selected final model is:

**HistGradientBoostingClassifier**

The complete final pipeline consists of:

```text
Feature Engineering
        ↓
Preprocessing
        ↓
HistGradientBoostingClassifier
```

The pipeline was saved as:

```text
final_model.joblib
```

---

## Classification Threshold

The default classification threshold of 0.50 was replaced with a selected threshold of:

```text
0.35
```

The threshold was selected using the validation data to improve F1 performance.

During inference:

```text
probability >= 0.35 → class 1 (>50K)
probability < 0.35  → class 0 (≤50K)
```

The threshold was saved separately as:

```text
final_threshold.joblib
```

---

## Final Test Performance

The final model was evaluated once on the held-out test set.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.8613** |
| Precision | **0.6850** |
| Recall    | **0.7784** |
| F1 Score  | **0.7287** |
| ROC AUC   | **0.9280** |
| PR AUC    | **0.8295** |

### Test Confusion Matrix

```text
                Predicted
                0       1

Actual 0       6594    837
Actual 1        518   1820
```

The model correctly identified **1,820 positive cases** while producing **837 false positives** and **518 false negatives**.

---

## Important Features

Permutation importance was used to interpret the HistGradientBoosting model.

The most influential features were:

| Rank | Feature        | Permutation Importance |
| ---: | -------------- | ---------------------: |
|    1 | marital-status |                 0.1420 |
|    2 | capital-gain   |                 0.1130 |
|    3 | education-num  |                 0.0837 |
|    4 | age            |                 0.0541 |
|    5 | occupation     |                 0.0385 |
|    6 | capital-loss   |                 0.0276 |
|    7 | hours-per-week |                 0.0237 |
|    8 | relationship   |                 0.0108 |
|    9 | sex            |                 0.0052 |
|   10 | workclass      |                 0.0038 |

Education and age showed generally positive relationships with predicted income probability. Marital status and occupation also had substantial effects on model predictions.

These relationships represent patterns learned by the model and **should not be interpreted as causal relationships**.

---

## Known Limitations

* The dataset reflects historical income patterns and may contain demographic or socioeconomic biases.
* Features such as `sex`, `race`, and `marital-status` may raise fairness concerns because the model uses demographic information when making predictions.
* Permutation importance measures predictive usefulness, not causality.
* Capital gain and capital loss are sparse and show non-linear relationships with predictions.
* The model may perform differently on populations or data distributions that differ from the original dataset.
* The selected threshold of 0.35 optimizes the chosen validation objective and may not be optimal for every real-world application.
* The model should therefore be treated as a predictive system rather than a tool for making high-stakes decisions about individuals.

---

## How to Reproduce Training

1. Install the required Python libraries listed in the **Environment** section.
2. Load the UCI Adult dataset.
3. Perform the documented train/validation/test split.
4. Apply the feature engineering and preprocessing pipeline.
5. Train and compare Logistic Regression, Random Forest, and HistGradientBoosting models.
6. Perform hyperparameter tuning for HistGradientBoosting.
7. Select the classification threshold using validation data.
8. Fit the final pipeline.
9. Evaluate the final model on the held-out test set.
10. Save the artifacts:

```text
final_model.joblib
final_threshold.joblib
```

A fixed `random_state=42` is used where applicable to support reproducibility.

---

## How to Run Inference

Load the saved artifacts:

```python
import joblib

model = joblib.load("final_model.joblib")
threshold = joblib.load("final_threshold.joblib")
```

Provide new observations using the original raw feature format:

```python
probability = model.predict_proba(new_data)[:, 1]

prediction = (probability >= threshold).astype(int)
```

The saved pipeline automatically performs:

```text
Raw Input
   ↓
Feature Engineering
   ↓
Missing-value Imputation
   ↓
Scaling / Encoding
   ↓
HistGradientBoosting
   ↓
Prediction Probability
   ↓
Threshold = 0.35
   ↓
Final Prediction
```

No manual preprocessing is required during inference.

The inference workflow was successfully tested on **10 unseen examples**.

---

## Saved Artifacts

```text
final_model.joblib
final_threshold.joblib
```

`final_model.joblib` contains the complete feature-engineering, preprocessing, and model pipeline.

`final_threshold.joblib` contains the selected classification threshold.

---

## Environment

The project was developed and tested using:

| Component    | Version |
| ------------ | ------- |
| Python       | 3.13.15 |
| NumPy        | 2.1.3   |
| Pandas       | 2.2.3   |
| Matplotlib   | 3.10.0  |
| Seaborn      | 0.13.2  |
| Scikit-learn | 1.6.1   |
| SciPy        | 1.16.3  |
| Joblib       | 1.5.3   |

These versions should be used when reproducing the project to minimize environment-related differences.

---

## Reproducibility

The project controls model randomness using `random_state=42` where applicable. The complete preprocessing and feature-engineering workflow is stored within the final pipeline, ensuring that training and inference use consistent transformations.

The final model and threshold are saved as serialized Joblib artifacts for reproducible inference.
