import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Read the CSV file
input_file = '../data/imputed_data.csv'
data = pd.read_csv(input_file)

# Identify numerical and categorical columns
numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

# Normalize the numerical data
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# One-hot encode the categorical data
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_data = encoder.fit_transform(data[categorical_columns])

# Create DataFrame for encoded categorical data
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Combine numerical and encoded categorical data
normalized_data = pd.concat([pd.DataFrame(data[numerical_columns], columns=numerical_columns), encoded_categorical_df], axis=1)

# Write the result to a new CSV file
output_file = '../data/normalized_encoded_output.csv'
normalized_data.to_csv(output_file, index=False)

print(f"Normalized and encoded data has been written to {output_file}")
