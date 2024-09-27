import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV file
file_path = 'inc_utf.csv'
data = pd.read_csv(file_path)

# Explicitly handle the '100+' entry
data['age'] = data['age'].apply(lambda x: x.replace(' years', '').replace('100+', '100')).astype(int)

# Extract the age and income columns
age = data['age'].values.reshape(-1, 1)
income = data['2020'].values


# Function to plot polynomial regression for different degrees
def plot_polynomial_regression(age, income, max_degree=10):
    plt.figure(figsize=(10, 6))

    # Scatter plot of data points
    plt.scatter(age, income, color='blue', label='data points')

    mse_list = []
    # Try polynomial degrees from 1 to max_degree
    for degree in range(1, max_degree + 1):
        poly_features = PolynomialFeatures(degree=degree)
        age_poly = poly_features.fit_transform(age)

        # Fit Linear Regression model
        model = LinearRegression()
        model.fit(age_poly, income)
        income_pred = model.predict(age_poly)

        # Calculate MSE
        mse = mean_squared_error(income, income_pred)
        mse_list.append(mse)

        # Plot the polynomial regression line
        plt.plot(age, income_pred, label=f'degree = {degree}, MSE = {mse:.6f}', linestyle='--')

    # Add plot labels and title
    plt.xlabel('Age')
    plt.ylabel('Average Income')
    plt.title('Linear Regression with Polynomial Features')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the polynomial regression with degrees up to 10
plot_polynomial_regression(age, income)
