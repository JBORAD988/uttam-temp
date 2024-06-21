import logreg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,classification_report
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

# Load the data
df = pd.read_csv('../data/imputed_data.csv')

# Check the data information
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
df['Age'] = df['Age'].astype(int)
df['Score'] = df['Score'].astype(int)

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

# Ensure 'Gender' is included in categorical columns
if 'Gender' not in categorical_cols:
    categorical_cols.append('Gender')

# Separate features (X) and target variable (Y)
X = df.drop('Score', axis=1)
Y = df['Score']

# Define the preprocessing steps using ColumnTransformer
scaler = MinMaxScaler(feature_range=(0, 1))
num_cols = [col for col in X.columns if X[col].dtype != 'object']
preprocessor = ColumnTransformer(transformers=[
    ('num', scaler, num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Convert to numpy arrays (optional step)
X = X.values
Y = Y.values

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_column):
    # Load the data
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns


    return X, y, numerical_cols, categorical_cols

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(model_save_path, f'best_model_{algo_name.replace(" ", "_").lower()}.joblib'))

    # Predict on the test data
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    recall = recall_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    # Classification report
    #print("\nClassification Report:")
    #print(classification_report(y_test, y_pred.round(), zero_division=0))

   # y_pred_proba = logreg.predict_proba(X_test)[::, 1]
   # false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)
    #auc = metrics.roc_auc_score(y_test, y_pred_proba)
    #plt.plot(false_positive_rate, true_positive_rate, label="AUC=" + str(auc))
    #plt.title('ROC Curve')
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('false Positive Rate')
    #plt.legend(loc=4)
    #plt.show()

    return y_pred, y_test

# Function to save results to CSV
def save_results_to_csv(y_test, final_predictions, algo_name, save_re_path):
    results_df = pd.DataFrame({
        'Actual Score': list(y_test),
        'Predicted Score': final_predictions,
    })
    results_df.to_csv(os.path.join(save_re_path, f'model_predictions_{algo_name.replace(" ", "_").lower()}.csv'),
                      index=False)
    print(f"Results have been exported to 'model_predictions_{algo_name.replace(' ', '_').lower()}.csv'.")

# Define algorithms to try and their hyperparameters
algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    #('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]}),
    #('SVR', SVR(), {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}),
    #('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 5, 10, 20]}),
    #('Elastic Net', ElasticNet(), {'regressor__alpha': [0.1, 0.5, 1.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    #('Lasso', Lasso(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    #('Ridge', Ridge(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    #('Linear Regression', LinearRegression(), {})
]

# Paths for saving data and models
final_df_path = "../data/imputed_data.csv"
model_save_path = '../models'
save_re_path = '../save_results'

# Train and evaluate models for each algorithm
for algo_name, regressor, param_grid in algorithms:
    print(f"\nTraining and evaluating using {algo_name}...")

    X, Y, numerical_cols, categorical_cols = load_and_preprocess_data(final_df_path, 'Score')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    predictions, y_test = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path)

    save_results_to_csv(y_test, predictions, algo_name, save_re_path)

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
    # encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(data[categorical_columns])

    # Create DataFrame for encoded categorical data
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data,
                                          columns=encoder.get_feature_names_out(categorical_columns))

    # Combine numerical and encoded categorical data
    normalized_data = pd.concat(
        [pd.DataFrame(data[numerical_columns], columns=numerical_columns), encoded_categorical_df], axis=1)

    # Write the result to a new CSV file
    output_file = '../data/normalized_encoded_output.csv'
    normalized_data.to_csv(output_file, index=False)

    print(f"Normalized and encoded data has been written to {output_file}")

    # Code for best feature selection#####################

    # Read the CSV file for best selection features
    final_df = pd.read_csv('../data/normalized_encoded_output.csv')
