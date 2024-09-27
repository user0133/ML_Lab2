import numpy as np
import pdb
import re
import pandas as pd
import matplotlib.pyplot as plt


# Task 3.1 Data Cleaning
def extract_age(age_str):
    # Extract the age from the string using regex
    age = int(re.search(r'\d+', age_str).group())
    return age

#Function to load CSV data
def load_csv_data(file_path):
    data = []
    clean_data = []
    with (open(file_path, 'r', encoding='utf-8-sig') as file):
        lines = file.readlines()
        header = lines[0].strip().split(',')
        # remove the first column as it is the header of the index
        del header[0]
        # pdb.set_trace() # start and stop the debugger
        for line in lines[1:]:
            row = line.strip().split(',')
            # remove the first column as it is the index
            del row[0]
            clean_row = row.copy()
            print(clean_row)
            clean_row[1] = extract_age(clean_row[1])
            data.append([x for x in row])
            clean_data.append(x for x in clean_row)
    return data, header, clean_data

# Convert the data into a pandas DataFrame
def save_data_in_df(data, header):
    return pd.DataFrame(data, columns=header)

file_path = 'inc_utf.csv'

# 1. Load
data, headers, clean_data = load_csv_data(file_path)
# Create a dataframe from the data and header
df = save_data_in_df(data, headers)
# Create a dataframe from the clean data and header
clean_df = save_data_in_df(clean_data, headers)

# To print the first 10 rows of the dataset
# To verify the data is loaded correctly as expected
print(df.head(10))

# To print the last 10 rows of the dataset
# To verify the the full data is loaded correctly
print(df.tail(10))

# Task 3.1 Data Cleaning
grouped_by_age_mean = clean_df.groupby('region')['age'].mean()
print (grouped_by_age_mean)

# Task 3.2
# Assuming the first column is X (feature) and the second is Y (target)
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

X = clean_df[headers[1]].astype(float).values
Y = clean_df[headers[2]].astype(float).values

def plot_regression(X, Y, predictions,title):
    plt.scatter(X, Y, color='blue', label=f'{title}')
    plt.plot(X, predictions, color='red', label='Regression Line')  # Regression line
    plt.xlabel('age')
    plt.ylabel('Average Income')
    plt.title(title)  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid lines
    plt.show()  # Display plot

# 2. Split the data into 80-20 train-validation sets using sample()
train_df = clean_df.sample(frac=0.8, random_state=42)  # 80% for training
val_df = clean_df.drop(train_df.index)  # Remaining 20% for validation

# 3. Separate features and target for training and validation
X_train = train_df[headers[1]].astype(float).values
Y_train = train_df[headers[2]].astype(float).values

X_val = val_df[headers[1]].astype(float).values
Y_val = val_df[headers[2]].astype(float).values

# Task 3.2 Perform linear regression using the Normal Equation on the training data
theta = perform_normal_equation(X_train, Y_train)

# Task 3.4 Predict based on the model for both training and validation sets
predictions_train = predict(X_train, theta)
predictions_val = predict(X_val, theta)
#  Add predictions to the training and validation DataFrames
train_df['Predictions'] = predictions_train
val_df['Predictions'] = predictions_val
print("Training Data:\n", train_df)
print("\nValidation Data:\n", val_df)

#Task 3.5 Evaluate model using MSE
mse_train = calculate_mse(Y_train, predictions_train)
mse_val = calculate_mse(Y_val, predictions_val)
print(f"Mean Squared Error (Training Set): {mse_train:.4f}")
print(f"Mean Squared Error (Validation Set): {mse_val:.4f}")

# Task 3.3 Plot the scatter plot and regression line for training and validation data
plot_regression(X_train, Y_train, predictions_train,'Training Data')
plot_regression(X_val, Y_val, predictions_val,'Validation Data')

# Predict Y for new input values
def predict_new_values(new_values, theta):
    new_values = np.array(new_values)
    return predict(new_values, theta)

# Example new input values for prediction
new_input = np.array([[20], [30], [35], [37]])
predicted_Y = predict_new_values(new_input, theta)

# Display the predictions for the new input values
for age, predicted_income in zip(new_input, predicted_Y):
    print(f"Predicted Average Income for age {age}: {predicted_income:.2f}")

