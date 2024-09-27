import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'inc_subset.csv'

# Function to load CSV data
def load_csv_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
        headers = lines[0].strip().split(',')
        #pdb.set_trace() start and stop the debugger
        for line in lines[1:]:
            row = line.strip().split(',')
            data.append([float(x) for x in row])
    return data, headers


# Convert the data into a pandas DataFrame
def save_data_in_df(data, headers):
    return pd.DataFrame(data, columns=headers)


# Perform linear regression using the Normal Equation
def perform_normal_equation(X, Y):
    # Add bias term (column of ones for the intercept) to X
    X_b = np.c_[np.ones((len(X), 1)), X]  # X_b becomes a matrix with a column of 1's for intercept

    # Normal Equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return theta


# Prediction function
def predict(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias term for predictions
    return X_b.dot(theta)
# Calculate Mean Squared Error (MSE)

def calculate_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def plot_regression(X, Y, predictions,title):
    plt.scatter(X, Y, color='blue', label=f'{title}')
    plt.plot(X, predictions, color='red', label='Regression Line')  # Regression line
    plt.xlabel('age')
    plt.ylabel('Average Income')
    plt.title(title)  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid lines
    plt.show()  # Display plot



# Task 1--Load and print the file, segregate X and Y axes and save in pandas data frame
data, headers = load_csv_data(file_path)
df = save_data_in_df(data, headers)

# Assuming the first column is X (feature) and the second is Y (target)
X = df[headers[1]].astype(float).values
Y = df[headers[2]].astype(float).values

# Task 2.1 -- Split the data into 80-20 train-validation sets using sample()
train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
val_df = df.drop(train_df.index)  # Remaining 20% for validation

# Separate features and target for training and validation
X_train = train_df[headers[1]].values
Y_train = train_df[headers[2]].values

X_val = val_df[headers[1]].values
Y_val = val_df[headers[2]].values

# Task 2 -- Perform linear regression using the Normal Equation on the training data
theta = perform_normal_equation(X_train, Y_train)

# Task 2.3  Predict values and print them based on the model for both training and validation sets
predictions_train = predict(X_train, theta)
predictions_val = predict(X_val, theta)
train_df['Predictions'] = predictions_train
val_df['Predictions'] = predictions_val
print("Training Data:\n", train_df)
print("\nValidation Data:\n", val_df)

# Task 2.4 Evaluate model using MSE
mse_train = calculate_mse(Y_train, predictions_train)
mse_val = calculate_mse(Y_val, predictions_val)
# 7. Plot the scatter plot and regression line for training and validation data
print(f"Mean Squared Error (Training Set): {mse_train:.4f}")
print(f"Mean Squared Error (Validation Set): {mse_val:.4f}")

# Task 2.2 Create Scatter plot
plot_regression(X_train, Y_train, predictions_train,'Training Data')
plot_regression(X_val, Y_val, predictions_val,'Validation Data')


''' Debugger is used in the scenarios when a step by step investigation in a complex program or large data set
    Print is used when the final output is needed , or the code is simple doesn't need step by step investigaton.'''

# Predict Y for new input values Additional
def predict_new_values(new_values, theta):
    new_values = np.array(new_values)
    return predict(new_values, theta)

# Example new input values for prediction
new_input = np.array([[20], [30], [35], [37]])
predicted_Y = predict_new_values(new_input, theta)

# Display the predictions for the new input values
for age, predicted_income in zip(new_input, predicted_Y):
    print(f"Predicted Average Income for age {age}: {predicted_income:.2f}")
