# Support Vector Machines & Linear Regression (CS446)

This project implements several machine learning algorithms from scratch for CS446 Homework 3.

## Implemented Methods

### 1. Support Vector Machine (SVM)
Implemented the dual formulation of SVM using projected gradient descent.

Features:
- Kernel SVM implementation
- Dual optimization
- Equality constraint projection
- Hard-margin and soft-margin support
- Prediction using support vectors

File:
- `hw3_q2.py`

### 2. Linear Regression Models

A full regression pipeline is implemented for the Ames Housing dataset.

Steps:
1. Data preprocessing
2. OLS regression
3. Ridge regression
4. Lasso regression using ISTA
5. Log-transform of target variable
6. Duan's smearing estimator for bias correction

File:
- `hw3_q4.py`


## Data Processing Pipeline

The regression pipeline includes:

- Train / validation / test split (70 / 15 / 15)
- Missing value imputation  
  - Numerical: median  
  - Categorical: mode
- One-hot encoding for categorical variables
- Z-score normalization for numerical variables
- Bias term added to the design matrix

Dataset:
- Ames Housing dataset (OpenML)


## Implemented Algorithms

### Ordinary Least Squares (OLS)

Closed-form solution:

\[
w^* = (X^TX)^{-1}X^Ty
\]

### Ridge Regression

Closed-form solution:

\[
w^* = (X^TX + \lambda I)^{-1}X^Ty
\]

Bias term is not regularized.

### Lasso Regression (ISTA)

Objective:

\[
\frac{1}{n}\|Xw-y\|^2 + \lambda\|w\|_1
\]

Solved using **Iterative Shrinkage-Thresholding Algorithm (ISTA)**.

Features:
- gradient step
- soft-thresholding
- early stopping
- Lipschitz step size

### Log Transform Motivation

The target variable `SalePrice` is highly right-skewed.

We apply

\[
\log(1 + y)
\]

to stabilize variance and improve regression assumptions.

### Duan's Smearing Estimator

Used to correct bias when transforming predictions back to the original scale.

\[
\hat y = \exp(\hat y_{log}) \times E[e^{\epsilon}]
\]


## Usage
```bash 
pip install pandas numpy matplotlib scikit-learn torch
python hw3_q2.py
python hw3_q4.py
```

## Output
* Test MSE
* Test RMSE
* Model comparison (OLS, Ridge, Lasso)
* Log-transform regression performance
* Feature sparsity for Lasso


© 2026 by KyoSook Shin
