# Ridge Regression Streamlit App
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Synthetic Data Generator
def generate_synthetic_data(n_samples=100, noise_std=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-0.3, 0.3, size=(n_samples, 1))
    epsilon = rng.normal(0, noise_std, size=n_samples)
    y = (np.cos(3 * np.pi * X).ravel() / 2) + 3 * X.ravel() + epsilon
    return X, y

# --- Polynomial Transformer
def polynomial_transform(X, degree=5):
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)

# --- Ridge Regression via Gradient Descent
class RidgeRegressionGD:
    def __init__(self, eta=0.01, lambda_=0.1, max_iter=1000, tol=1e-6, stochastic=False, batch_size=1, random_state=None):
        self.eta = eta
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.stochastic = stochastic
        self.batch_size = batch_size
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.max_iter):
            if self.stochastic:
                idx = rng.choice(n_samples, size=self.batch_size, replace=False)
                Xb, yb = X[idx], y[idx]
                grad = -(Xb.T @ (yb - Xb @ self.coef_)) / self.batch_size + self.lambda_ * self.coef_
            else:
                grad = -(X.T @ (y - X @ self.coef_)) / n_samples + self.lambda_ * self.coef_
            new_coef = self.coef_ - self.eta * grad
            if np.linalg.norm(new_coef - self.coef_) < self.tol:
                self.coef_ = new_coef
                break
            self.coef_ = new_coef
        return self

    def predict(self, X):
        return X @ self.coef_

# --- Lambda Range
def make_lambda_values(n=101):
    return [10 ** (-10 + 9 * i / (n-1)) for i in range(n)]

# --- Evaluate Ridge Regression
def evaluate_ridge(X_train, y_train, X_test, y_test, lambdas):
    train_errs, test_errs = [], []
    for lam in lambdas:
        model = RidgeRegressionGD(lambda_=lam)
        model.fit(X_train, y_train)
        train_errs.append(mean_squared_error(y_train, model.predict(X_train)))
        test_errs.append(mean_squared_error(y_test, model.predict(X_test)))
    return train_errs, test_errs

# --- Main App
def main():
    st.title('Ridge Regression Dashboard')
    # Introduction
    st.markdown(
        """
        **Objective:** Explore how Ridge Regression, a linear model with L2 regularization, can capture non-linear patterns via polynomial feature expansion.  
        We generate synthetic data (a cosine-based function plus noise) to provide a controlled environment for testing model behavior across different regularization strengths and polynomial degrees.  
        Ridge Regression penalizes large coefficients to prevent overfitting, striking a balance between bias and variance.  
        This interactive dashboard lets you adjust data noise, sample size, train/test split, learning rate, polynomial degree, and regularization parameter (λ) to see their effects on model fit, error curves, and predictions.
        """
    )

    # Sidebar controls
    st.sidebar.header('Settings')
    n_samples = st.sidebar.slider('Number of samples', 50, 1000, 100, step=50)
    noise = st.sidebar.slider('Noise std dev', 0.0, 1.0, 0.1, step=0.05)
    train_ratio = st.sidebar.slider('Train/Test Split', 0.5, 0.9, 0.8, step=0.05)
    degree = st.sidebar.slider('Polynomial degree', 1, 10, 5)
    eta = st.sidebar.slider('Learning rate', 0.001, 0.1, 0.01)
    lam = st.sidebar.select_slider('Lambda (regularization)', options=make_lambda_values(), value=0.1)

    # Generate and split data
    X, y = generate_synthetic_data(n_samples=n_samples, noise_std=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    # Transform features
    X_train_poly = polynomial_transform(X_train, degree)
    X_test_poly = polynomial_transform(X_test, degree)

    # Fit model
    model = RidgeRegressionGD(eta=eta, lambda_=lam)
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

        # 1. Data Overview & EDA
    st.subheader('Feature & Target Distribution')
    st.markdown(
        'This section shows the distribution of your input feature X and the target y. ' 
        'Histograms help identify skewness, modality, and outliers.'
    )
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(X, bins=20, edgecolor='k'); ax1.set_title('X distribution')
    ax2.hist(y, bins=20, edgecolor='k'); ax2.set_title('y distribution')
    st.pyplot(fig1)
    st.markdown(
        "The histograms above allow you to assess the distribution of input and output.\
        Look for skewness (long tails), multiple modes, or outliers that could affect model performance."
    )

    # 2. Model Fit
    st.subheader('Model Fit on Training Data')
    fig2, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(X_train, y_train, label='True')
    xs = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ys = model.predict(polynomial_transform(xs, degree))
    ax3.plot(xs, ys, color='r', label='Prediction')
    ax3.legend(); ax3.set_title('Ridge Regression Fit')
    st.pyplot(fig2)
    st.markdown(
        "This scatter plot overlays the true training data with the ridge regression curve.\
        Check how well the model follows the data: gaps indicate underfitting, excessive wiggles indicate overfitting."
    )

    # 3. Error vs Lambda
    st.subheader('Training & Testing Error vs Lambda')
    lambdas = make_lambda_values()
    train_errs, test_errs = evaluate_ridge(X_train_poly, y_train, X_test_poly, y_test, lambdas)
    fig3, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(np.log10(lambdas), np.log10(train_errs), label='Train')
    ax4.plot(np.log10(lambdas), np.log10(test_errs), label='Test')
    ax4.set_xlabel('log10(lambda)'); ax4.set_ylabel('log10(MSE)'); ax4.legend(); ax4.grid(True)
    st.pyplot(fig3)
    st.markdown(
        "This log–log plot of mean squared error versus regularization shows the bias–variance tradeoff.\
        The minimum test curve point indicates the optimal λ. Test performance rising on both ends signals under/overfitting regions."
    )

    # 4. Interactive Prediction
    st.subheader('Interactive Prediction')
    x_new = st.number_input('Input new X value', float(X.min()), float(X.max()), value=float(0.0), step=0.01)
    x_new_arr = np.array([[x_new]])
    x_new_poly = polynomial_transform(x_new_arr, degree)
    y_new_pred = model.predict(x_new_poly)[0]
    std_res = np.std(y_train - y_train_pred)
    st.write(f"Prediction for X={x_new:.2f}: {y_new_pred:.2f} ± {std_res:.2f}")
    st.markdown(
        "This shows the predicted output for a custom input X plus an uncertainty estimate (training residual std).\
        Use it to gauge confidence: narrower intervals mean more reliable predictions. {y_new_pred:.2f} ± {std_res:.2f}")

    # 5. Partial Dependence Curve
    st.subheader('Partial Dependence Curve')
    xs_pd = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ys_pd = model.predict(polynomial_transform(xs_pd, degree))
    fig4, ax5 = plt.subplots(figsize=(6, 4))
    ax5.plot(xs_pd, ys_pd, color='blue')
    ax5.set_xlabel('X'); ax5.set_ylabel('Predicted y'); ax5.set_title('Partial Dependence of y on X')
    st.pyplot(fig4)
    st.markdown(
        "The partial dependence curve illustrates how the predicted y changes as X varies.\
        It visualizes the learned non-linear relationship; flat regions indicate insensitivity, steep slopes show high sensitivity."
    )

if __name__ == '__main__':
    main()
