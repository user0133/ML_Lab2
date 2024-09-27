import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pdb
import re
import pandas as pd

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
            clean_row[1] = extract_age(clean_row[1])
            data.append([x for x in row])
            clean_data.append(x for x in clean_row)
    return data, header, clean_data

# Convert the data into a pandas DataFrame
def save_data_in_df(data, header):
    return pd.DataFrame(data, columns=header)


file_path = 'inc_utf.csv'

# 1. Load and print the file, segregate X and Y axes
data, headers, clean_data = load_csv_data(file_path)

df = save_data_in_df(clean_data, headers)

# 2. input feature(s) and target variable
X = df[['age']]  # Input feature(s)
y = df['2020']     # Target variable

# 3. Generate Polynomial Features
degree = 8 # Set the degree for the polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)  # Generate polynomial features

# 4. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_poly, y)

# 5. Make Predictions
y_pred = model.predict(X_poly)

# 6. Optional, split the data into training and testing sets if needed
#X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 7. Visualization (as X has only one feature age )
if X.shape[1] == 1:
    #plt.scatter(X, y, color='blue', label='Original Data')
    plt.plot(X, y_pred, color='red', label=f'Polynomial Regression (degree={degree})')
    plt.xlabel('Age')
    plt.ylabel('Income of 2020 (Year)')
    plt.title('Polynomial Regression of Age and Income (no split)')
    plt.legend()
    plt.show()
