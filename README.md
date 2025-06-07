# Diabetes Prediction using Logistic Regression

This project focuses on predicting the onset of diabetes based on a set of diagnostic medical measurements. An Exploratory Data Analysis (EDA) was performed to understand the relationships between different features, followed by the development of a machine learning model to make predictions.

## Project Overview

1.  **Data Cleaning & Preparation**: Loaded the dataset and handled missing or zero values in critical columns.
2.  **Exploratory Data Analysis (EDA)**: Analyzed the data statistically and visually to identify patterns, correlations, and outliers.
3.  **Feature Engineering & Preprocessing**: Scaled the features to prepare them for the machine learning model.
4.  **Model Building**: Trained a Logistic Regression model with L1 regularization.
5.  **Model Evaluation**: Assessed the model's performance on a held-out test set using accuracy and a detailed classification report.

---

## Dataset

The dataset used is the **PIMA Indians Diabetes Database**, which is a classic dataset for binary classification problems. It includes several medical predictor variables and one target variable, `Outcome`.

**Features:**
* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`
* `Outcome` (Target Variable: 0 = No Diabetes, 1 = Diabetes)

---

## Exploratory Data Analysis (EDA)

A thorough EDA was conducted to gain insights from the data.

* **Initial Analysis**: Used `pandas` functions like `.info()`, `.describe()`, and `.isnull().sum()` to get a summary of the data and check for missing values.
* **Handling Missing Data**: Identified that columns like `Glucose`, `BloodPressure`, and `BMI` had zero values, which are physiologically impossible. These were replaced with the mean of the respective columns to avoid data loss.
* **Visualization**:
    * Used **`seaborn`** and **`matplotlib`** to create histograms and density plots to visualize the distribution of each feature.
    * A correlation heatmap was generated to understand the relationships between different variables. This revealed a strong correlation between `Glucose`, `Age`, `BMI`, and the `Outcome`.

---

## Feature Engineering & Preprocessing

To ensure the model performs optimally, the features were preprocessed as follows:

* **Feature Scaling**: The numerical features were scaled using `StandardScaler` from **`scikit-learn`**. This standardizes features by removing the mean and scaling to unit variance, which is crucial for distance-based algorithms and models that use regularization, like Logistic Regression.

---

## Modeling

A **Logistic Regression** model was chosen for this classification task due to its interpretability and efficiency.

* **Model**: `LogisticRegression` from `scikit-learn`.
* **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets using `train_test_split`.
* **Regularization**: **L1 (Lasso) regularization** (`penalty='l1'`) was used to help prevent overfitting by adding a penalty equivalent to the absolute value of the magnitude of coefficients. This can also perform feature selection by shrinking less important feature coefficients to zero.
* **Solver**: The `liblinear` solver was chosen as it is well-suited for smaller datasets and works well with L1 regularization.

---

## Results & Evaluation

The model was evaluated on the unseen test data, and it achieved strong performance metrics.

* **Final Accuracy**: **81.88%**

### Classification Report

The classification report provides a detailed breakdown of the model's performance for each class.

```
                     precision    recall  f1-score   support

           0              0.84      0.91      0.87       102
           1              0.76      0.62      0.68        47

    accuracy                                0.82       149
   macro avg            0.80      0.76      0.78       149
weighted avg            0.81      0.82      0.81       149
```

**Key Takeaways from the Report:**
* The model is particularly good at identifying patients **without** diabetes (Class 0), with a recall of 0.91.
* It correctly identifies patients **with** diabetes (Class 1) 62% of the time (recall of 0.62).
* The overall precision and F1-score demonstrate a balanced and effective model.

---

## Technologies Used

* **Python 3**
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For machine learning (modeling, preprocessing, and evaluation).
* **Matplotlib & Seaborn**: For data visualization.
* **Jupyter Notebook**: For interactive development.

---
