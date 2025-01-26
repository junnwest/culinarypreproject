# Cuisine Classification Project

## Project Overview
This project focuses on building a machine learning pipeline to classify cuisines based on their ingredients. Using a dataset of recipes, the goal is to predict the type of cuisine (e.g., Italian, Indian, Chinese) given a list of ingredients. The project explores various machine learning algorithms, feature engineering techniques, and dimensionality reduction methods to achieve optimal classification performance.

---

## Dataset Description

The dataset is provided in the JSON file: `train.json`.

### Dataset (`train.json`)
- Contains three columns:
  - **`id`**: Unique identifier for each recipe.
  - **`cuisine`**: The type of cuisine (target variable).
  - **`ingredients`**: A list of ingredients for each recipe.

---

## Project Workflow

### 1. Data Preprocessing
- Cleaned the ingredients by:
  - Removing descriptive modifiers (e.g., `"chopped onions"` → `"onions"`).
  - Standardizing ingredient names using mappings (e.g., `"kosher salt"` → `"salt"`).
- Combined ingredients into a single string, separated by `|||`, for text vectorization.
- Split the `train.json` file into training and test sets (since the provided `test.json` lacks labels).

### 2. Feature Engineering
- Applied **TF-IDF Vectorization** to convert the list of ingredients into numerical vectors suitable for machine learning models.
- Reduced the dimensionality of the TF-IDF vectors using **Principal Component Analysis (PCA)**:
  - Retained 90% of the variance by reducing to **1468 components**.

### 3. Clustering
- Performed **K-Means Clustering** on the PCA-reduced data to group cuisines into clusters.
- Used the cluster labels as additional features for supervised classification models.

### 4. Machine Learning Models
Trained and evaluated the following models:
- **K-Nearest Neighbors (KNN)**
- **Naïve Bayes (NB)**
- **Logistic Regression (LR)**
- **Random Forest (RF)**
- **Decision Tree (DT)**
- **Support Vector Machine (SVM)**
- **Stochastic Gradient Descent (SGD)**

### 5. Hyperparameter Tuning
- Optimized model performance using **GridSearchCV** for hyperparameter tuning.
- For example, tuned `C`, `gamma`, and `kernel` for SVM.

---

## Results

### Model Comparison
| Model                  | Accuracy     |
|------------------------|--------------|
| Support Vector Machine (SVM) | **78.75%**  |
| Logistic Regression    | 77.28%       |
| Stochastic Gradient Descent (SGD) | 76.89% |
| Random Forest          | 64.49%       |
| K-Nearest Neighbors    | 55.93%       |
| Decision Tree          | 48.37%       |

### Clustering Visualization
- Performed 2D visualization of cuisines in PCA-reduced space with cluster labels from K-Means.
- Provided insights into how cuisines group based on their ingredients.