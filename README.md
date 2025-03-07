# Ridge Regression with Regularization

## Project Overview
This project implements **Ridge Regression**, a regularized linear regression model, to prevent overfitting and improve prediction accuracy. It explores different values of the regularization parameter (alpha) and evaluates its impact on model performance. The dataset is preprocessed, standardized, and analyzed for optimal hyperparameter selection.

Additionally, the notebook includes:
- **Automatic Hyperparameter Selection**: Choosing the best alpha value using cross-validation.
- **Feature Scaling Techniques**: Standardizing features to improve model stability.
- **Regularization Impact Analysis**: Understanding how different alpha values affect model coefficients.
- **Performance Evaluation**: Using mean squared error (MSE) to compare different models.
- **Visualization of Regression Coefficients**: Analyzing coefficient shrinkage with increasing regularization.


## Folder Structure
Ridge_Regression_with_Regularization/
      
   ├── MLProject2-RidgeRegression.ipynb  # Jupyter Notebook implementing Ridge Regression
   
   ├── README.md  # Project documentation
   
   ├── requirements.txt  # Dependencies required to run the project

## Installation
Ensure you have Python installed (preferably Python 3.8 or above). Install the required dependencies using the following command:
```sh
pip install -r requirements.txt 
```

## Usage
1. Open the Jupyter Notebook:
```sh
   jupyter notebook MLProject2-RidgeRegression.ipynb
```

2. Execute the cells step by step to:
- Load and preprocess the dataset.
- Implement Ridge Regression with different regularization parameters.
- Evaluate model performance using cross-validation.
- Visualize coefficient shrinkage and model predictions.

## Dataset Description
- **application_data.csv**: Contains details of loan applications, including applicant income, credit amount, down payment, contract status, and decision timelines.
- **previous_application.csv**: Historical records of previous loan applications.
- **value_dict.csv**: Mapping file for categorical values used in datasets.

## Methodology

### 1. Data Preprocessing
- Handling missing values (if any).
- Standardizing numerical features for better model performance.

### 2. Ridge Regression Implementation
- Using **scikit-learn's Ridge** model.
- Exploring different values of **alpha** (regularization strength).

### 3. Cross-Validation for Hyperparameter Tuning
- Implementing **K-Fold CV** to select the best **alpha**.
- Evaluating models based on **Mean Squared Error (MSE)**.

### 4. Results & Analysis
- **Performance Visualization**: Line plots showing the effect of alpha on MSE.
- **Optimal Alpha Selection**: Identifying the best regularization strength.
- **Coefficient Analysis**: Visualizing how feature coefficients shrink with higher alpha values.


## Notes
- Adjust **alpha** range if working with a different dataset.
- Compare Ridge Regression with **Linear Regression** to observe overfitting effects.

