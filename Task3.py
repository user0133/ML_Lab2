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
