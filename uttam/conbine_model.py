import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def print_dataset_features(X):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    print("\nNumerical features:")
    for col in numerical_cols:
        print(f" - {col} (dtype: {X[col].dtype})")
    print("Categorical features:")
    for col in categorical_cols:
        print(f" - {col} (dtype: {X[col].dtype})")

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)])
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Save the best model using joblib
    joblib.dump(best_model, os.path.join(model_save_path, f'best_model_{algo_name.replace(" ", "_").lower()}.joblib'))

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    print("MSE:", mse, "R2:", r2)

    return y_pred, y_test

def categorize_scores(scores):
    categories = []
    for score in scores:
        if score < 50:
            categories.append('Fail')
        elif 50 <= score < 60:
            categories.append('Second Class')
        elif 60 <= score < 75:
            categories.append('First Class')
        else:
            categories.append('First Class with Distinction')
    return categories

def save_results_to_csv(y_test, final_predictions, categories, algo_name, save_re_path):
    results_df = pd.DataFrame({
        'Actual Score': list(y_test),
        'Predicted Score': final_predictions,
        'Category': categories
    })

    results_df.to_csv(os.path.join(save_re_path, f'model_predictions_{algo_name.replace(" ", "_").lower()}.csv'), index=False)
    print(f"Results have been exported to 'model_predictions_{algo_name.replace(' ', '_').lower()}.csv'.")

algorithms = [
    # Model 1
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    
    # Model 2
    ('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]})
]
final_df_path = "../data/final.csv"
model_save_path = '../models'
save_re_path = '../save_results'

# Iterate over each model
for algo_name, regressor, param_grid in algorithms:
    print(f"\nTraining and evaluating using {algo_name}...")
    
    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf = load_and_preprocess_data(final_df_path, 'Score')
    
    # Print dataset features
    print("\nFeatures for", algo_name)
    print_dataset_features(X_train_pf)
   
    predictions_pf, y_test_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor, param_grid, algo_name, model_save_path)
   
    # Save predictions for each model
    joblib.dump(predictions_pf, os.path.join(model_save_path, f'predictions_{algo_name.replace(" ", "_").lower()}.joblib'))
    
    # Display results for each model
    final_predictions = np.maximum.reduce([predictions_pf])
    categories = categorize_scores(final_predictions)

    save_results_to_csv(y_test_pf, final_predictions, categories, algo_name, save_re_path)

# Combine predictions from both models (simple averaging)
predictions_gb = joblib.load(os.path.join(model_save_path, 'predictions_gradient_boosting.joblib'))
predictions_rf = joblib.load(os.path.join(model_save_path, 'predictions_random_forest.joblib'))

final_predictions_combined = (predictions_gb + predictions_rf) / 2

# Categorize scores for combined predictions
categories_combined = categorize_scores(final_predictions_combined)

# Additional function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

# Save results for combined predictions
save_results_to_csv(y_test_pf, final_predictions_combined, categories_combined, 'combined_models', save_re_path)

# Evaluate combined models MSE and R2
mse_combined = mean_squared_error(y_test_pf, final_predictions_combined)
actual_categories = categorize_scores(y_test_pf)
accuracy_combined = calculate_accuracy(actual_categories, categories_combined)

print("Combined Models MSE:", mse_combined, "R2:", accuracy_combined)
