import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

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

# Save the DataFrame after imputation to a new CSV file
df.to_csv('../data/imputed_data.csv', index=False)
print("Imputed data saved to '../data/imputed_data.csv'")

# Boxplot for 'Score'
plt.boxplot(df['Score'], vert=False)
plt.xlabel('Score')
plt.title('Box Plot for Score')
plt.show()

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
print('Data after dropping outliers:\n', df.describe())

# Save preprocessed data to a new CSV file
df.to_csv('../data/preprocessed_data.csv', index=False)
print("Preprocessed data saved to '../data/preprocessed_data.csv'")

# Code for best feature selection
final_df = pd.read_csv('../data/preprocessed_data.csv')

X = final_df.drop('Score', axis=1)
Y = final_df['Score']

# Apply SelectKBest to select the top 3 features
selector = SelectKBest(score_func=f_regression, k=3)  # Using f_regression as the scoring function
X_selected = selector.fit_transform(X, Y)

# Get indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Get names of the selected features
selected_feature_names = X.columns[selected_feature_indices]

print("Selected Features:")
for feature in selected_feature_names:
    print(feature)

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_column, selected_features):
    # Load the data
    df = pd.read_csv(file_path)

    # Select only the features that were used in training
    X = df[selected_features]
    y = df[target_column]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    return X, y, numerical_cols, categorical_cols

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path):
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
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Metrics for {algo_name}:")
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')

    return y_pred, y_test

# Additional algorithms to try
algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    ('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]}),
    #('SVR', SVR(), {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}),
    #('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 5, 10, 20]}),
    #('Elastic Net', ElasticNet(), {'regressor__alpha': [0.1, 0.5, 1.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    #('Lasso', Lasso(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    #('Ridge', Ridge(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    #('Linear Regression', LinearRegression(), {})
]

# Paths for saving data and models
