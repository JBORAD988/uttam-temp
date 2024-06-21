import joblib
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif  # Adjust import based on your feature selection method
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Load the data
df = pd.read_csv('../data/all1.csv')
print(df.head(10))

# Print the number of Rows and Columns
print('Number of Rows:', df.shape[0])
print('Number of Columns:', df.shape[1])

# Check number of missing values
missing_data_count = df.isnull().sum().sum()
print(f"Number of missing data points: {missing_data_count}")

# Print only columns with missing values and their percentages
missing_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
print("Columns with missing values:")
for col, pct_missing in missing_percentage.items():
    if pct_missing > 0:
        print(f"{col}: {pct_missing}%")

# Check number of duplicate rows
duplicate_rows_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows_count}")

# Display data information
df.info()

# Display descriptive statistics
print(df.describe())

# Display correlation matrix for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Correlation Matrix:")
print(df[numerical_cols].corr())

# Identify numerical and categorical columns
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
print('Numerical columns:', numerical_cols)
print('Categorical columns:', categorical_cols)

# Print number of unique values in categorical columns
print("Number of unique values in categorical columns:\n", df[categorical_cols].nunique())

# Print the first 50 unique Scores
print('First 50 Unique Scores:', df['Score'].unique()[:50])

# Convert Age and Score to integer if they are float
mean_age = df['Age'].mean()
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(mean_age).astype(int)
mean_score = df['Score'].mean()
df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(mean_score).astype(int)

# Function to handle outliers
def handle_outliers(df, col):
    if df[col].dtype == 'object':
        print(f"Outliers in '{col}' are not applicable for this method.")
        return df
    else:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        print(f'Lower and Upper Bounds for {col}:')
        print(f'Lower Bound ({col}):', lower_bound)
        print(f'Upper Bound ({col}):', upper_bound)
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Handle outliers for numerical columns
df = handle_outliers(df, 'Age')
df = handle_outliers(df, 'Score')

# Map different representations to a standard form for Gender
gender_map = {'m': 'Male', 'male': 'Male', 'M': 'Male', 'Male': 'Male',
              'f': 'Female', 'female': 'Female', 'F': 'Female', 'Female': 'Female'}
df['Gender'] = df['Gender'].map(gender_map)
df = df.dropna(subset=['Gender'])
print(df['Gender'].unique())

# Ensure 'Gender' is included in categorical columns
if 'Gender' not in categorical_cols:
    categorical_cols.append('Gender')

# Ensure Age is between 16 and 17
df = df[(df['Age'] >= 16) & (df['Age'] <= 17)]

# Ensure Score is between 0 and 100
df = df[(df['Score'] > 0) & (df['Score'] <= 100)]

# Check for null values again
print('Null values after imputation and filtering:\n', df.isnull().sum())

print('Data after dropping outliers:\n', df.describe())

# Save the DataFrame after imputation to a new CSV file
df.to_csv('../data/imputed_data.csv', index=False)
print("Imputed data saved to '../data/imputed_data.csv'")

# Separate features (X) and target variable (Y)
# X = df.drop('Score', axis=1)
# Y = df['Score']

# Step 1: Read the CSV file into a DataFrame
# Replace 'your_file.csv' with the path to your actual CSV file
df = pd.read_csv('../data/imputed_data.csv')

# Step 2: Separate the first column from the rest
first_col = df.iloc[:, 0]
rest_df = df.iloc[:, 1:]

# Step 3: Identify numeric and non-numeric columns
numeric_cols = rest_df.select_dtypes(include=['number']).columns
non_numeric_cols = rest_df.select_dtypes(exclude=['number']).columns

# Step 4: Normalize the numeric columns
min_vals = rest_df[numeric_cols].min()
max_vals = rest_df[numeric_cols].max()
rest_df[numeric_cols] = (rest_df[numeric_cols] - min_vals) / (max_vals - min_vals)

# Step 5: Normalize the non-numeric columns
# Convert boolean columns to integers (0 and 1)
bool_cols = rest_df.select_dtypes(include=['bool']).columns
rest_df[bool_cols] = rest_df[bool_cols].astype(int)

# Apply label encoding for object columns
label_encoder = LabelEncoder()
for col in rest_df.select_dtypes(include=['object']).columns:
    rest_df[col] = label_encoder.fit_transform(rest_df[col])

# Step 6: Concatenate the first column back with the normalized DataFrame
normalized_df = pd.concat([first_col, rest_df], axis=1)

# Step 7: Save the DataFrame back to a CSV file (if needed)
normalized_df.to_csv('../data/normalized_encoded_output.csv', index=False)

# Display the DataFrame with the normalized columns
print(normalized_df)

X = normalized_df.drop('Score', axis=1)
Y = normalized_df['Score']

# Apply SelectKBest to select the top 10 features
selector = SelectKBest(score_func=f_regression, k=15)  # Using f_regression as the scoring function
# Perform feature selection
#selector = SelectKBest(score_func=f_classif, k=15)  # Example: Select top 3 features
X_selected = selector.fit_transform(X, Y)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

# Print selected features
print("Selected Features:")
for feature in selected_feature_names:
    print(feature)

# Save dataset containing only selected features to a new CSV file
X[selected_feature_names].to_csv('../data/selected_features_output.csv', index=False)
print("Dataset with selected features saved to '../data/selected_features_output.csv'")

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_column, selected_features):
    """
    Function to load data from a CSV file, select specific features, and preprocess them for modeling.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.
    - target_column (str): Name of the target column (dependent variable).
    - selected_features (list): List of feature names to be used for modeling.

    Returns:
    - X (DataFrame): DataFrame containing the selected features.
    - Y (Series): Series containing the target variable.
    - numerical_cols (list): List of numerical column names in X.
    - categorical_cols (list): List of categorical column names in X.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Select only the features that were used in training
    X = df[selected_features]
    Y = df[target_column]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    return X, Y, numerical_cols, categorical_cols

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path):
    """
    Function to train and evaluate a regression model using GridSearchCV.

    Parameters:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Test features.
    - y_train (Series): Training target.
    - y_test (Series): Test target.
    - preprocessor (ColumnTransformer): Preprocessing steps including scaling and encoding.
    - regressor (estimator): Regression model to be trained.
    - param_grid (dict): Parameter grid for hyperparameter tuning.
    - algo_name (str): Name of the algorithm.
    - model_save_path (str): Path to save the trained model.

    Returns:
    - y_pred (array): Predicted values on the test set.
    - y_test (array): Actual values on the test set.
    """
    # Initialize model in Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(model_save_path, f'best_model_{algo_name.replace(" ", "_").lower()}.joblib'))

    # Predict on the test data
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)

    # Convert regression predictions to classes (e.g., binarizing predictions based on a threshold)
    threshold = 70  # Example threshold for classification
    y_pred_class = np.where(y_pred >= threshold, 1, 0)
    y_test_class = np.where(y_test >= threshold, 1, 0)

    # Calculate evaluation metrics for classification
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, zero_division=0)
    recall = recall_score(y_test_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

    # Print evaluation metrics
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared: {r2}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    return y_pred, y_test

# Function to save results to CSV
def save_results_to_csv(y_test, final_predictions, algo_name, save_re_path):
    """
    Function to save model predictions and actual values to a CSV file.

    Parameters:
    - y_test (array): Actual values (target) from the test set.
    - final_predictions (array): Predicted values from the model.
    - algo_name (str): Name of the algorithm used for predictions.
    - save_re_path (str): Path to save the results CSV file.
    """
    results_df = pd.DataFrame({
        'Actual Score': list(y_test),
        'Predicted Score': final_predictions,
    })
    results_df.to_csv(os.path.join(save_re_path, f'model_predictions_{algo_name.replace(" ", "_").lower()}.csv'),
                      index=False)
    print(f"Results have been exported to 'model_predictions_{algo_name.replace(' ', '_').lower()}.csv'.")

# Additional algorithms to try
algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
     ('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]}),
    ('SVR', SVR(), {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}),
    ('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 5, 10, 20]}),
    ('Elastic Net', ElasticNet(), {'regressor__alpha': [0.1, 0.5, 1.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    #('Lasso', Lasso(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    ('Ridge', Ridge(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
 #('Linear Regression', LinearRegression(), {})
]

# Paths for saving data and models
final_df_path = "../data/normalized_encoded_output.csv"
model_save_path = '../models'
save_re_path = '../save_results'

# Train and evaluate models for each algorithm
for algo_name, regressor, param_grid in algorithms:
    print(f"\nTraining and evaluating using {algo_name}...")

    X, Y, numerical_cols, categorical_cols = load_and_preprocess_data(final_df_path, 'Score', selected_feature_names)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Define preprocessing steps including scaling for numerical columns and one-hot encoding for categorical columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    predictions, y_test = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid,
                                             algo_name, model_save_path)

    save_results_to_csv(y_test, predictions, algo_name, save_re_path)

