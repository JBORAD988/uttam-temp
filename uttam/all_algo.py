from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import os
from sklearn.feature_selection import SelectKBest, f_regression


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    ####### PCA CODE START #######
    n_samples = X.shape[0]
    n_features = len(numerical_cols)
    n_components = min(n_features, n_samples, 2)  # Limiting to 2 to avoid the error

    numerical_transformer_steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    if n_components > 1:
        numerical_transformer_steps.append(('pca', PCA(n_components=n_components)))
    ####### PCA CODE END #######
    numerical_transformer = Pipeline(steps=numerical_transformer_steps)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols

def print_dataset_features(X, pca, numerical_cols):
    if pca is not None:
        print("\nPCA Component Analysis:")
        components = pca.components_
        for i, component in enumerate(components):
            feature_contributions = zip(numerical_cols, component)
            sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
            print(f"\nComponent {i + 1}:")
            print(" Best fit features:")
            for feature, loading in sorted_features[:3]:
                print(f"   {feature}: {loading}")
            print(" Low fit features:")
            for feature, loading in sorted_features[-3:]:
                print(f"   {feature}: {loading}")

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name, model_save_path):
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor)])

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Save the best model using joblib
    joblib.dump(best_model, os.path.join(model_save_path, f'best_model_{algo_name.replace(" ", "_").lower()}.joblib'))

    #y_pred = best_model.predict(X_test)

    # Before calculating MSE, perform feature selection
    selector = SelectKBest(score_func=f_regression, k=10)  # Select the top 10 features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train and predict using the selected features
    regressor.fit(X_train_selected, y_train)
    y_pred = regressor.predict(X_test_selected)

    mse = mean_squared_error(y_test, y_pred) # Calculate MSE
    r2 = r2_score(y_test, y_pred)   # Calculate R2
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE


    # Print best parameters found by GridSearchCV
    print(f"Best Parameters for {algo_name}:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  {param_name}: {param_value}")

    print("MSE:", mse, "R2:", r2, "MAE:", mae)  # Print MSE, R2, MAE

    # Determine if the model is a good fit
    if mse < 10 and r2 > 0.8:
        print(f"The {algo_name} model is a good fit.")
    else:
        print(f"The {algo_name} model is not a good fit.")

    # Access the PCA step if it exists
    pca = None
    if 'pca' in best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps:
        pca = best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['pca']

    return y_pred, y_test, pca

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

def save_results_to_csv(y_test, final_predictions, categories, algo_name, save_re_path, normalized_scores):
    results_df = pd.DataFrame({
        'Actual Score': list(y_test),
        'Predicted Score': final_predictions,
        'Category': categories,
        'Normalized Score': normalized_scores
    })

    results_df.to_csv(os.path.join(save_re_path, f'model_predictions_{algo_name.replace(" ", "_").lower()}.csv'), index=False)
    print(f"Results have been exported to 'model_predictions_{algo_name.replace(' ', '_').lower()}.csv'.")

# Additional algorithms to try
algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    ('Random Forest', RandomForestRegressor(), {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]}),
    ('SVR', SVR(), {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}),
    ('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 5, 10, 20]}),
    ('Elastic Net', ElasticNet(), {'regressor__alpha': [0.1, 0.5, 1.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    ('Lasso', Lasso(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    ('Ridge', Ridge(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    ('Linear Regression', LinearRegression(), {})
]

final_df_path = "../data/preprocessed_data.csv"
model_save_path = '../models'
save_re_path = '../save_results'

for algo_name, regressor, param_grid in algorithms:
    print(f"\nTraining and evaluating using {algo_name}...")

    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, numerical_cols_pf = load_and_preprocess_data(final_df_path, 'Score')

    predictions_pf, y_test_pf, pca_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor, param_grid, algo_name, model_save_path)

    # Print dataset features with PCA analysis
    print("\nFeatures for", algo_name)
    print_dataset_features(X_train_pf, pca_pf, numerical_cols_pf)

    # Save predictions for each model
    joblib.dump(predictions_pf, os.path.join(model_save_path, f'predictions_{algo_name.replace(" ", "_").lower()}.joblib'))

    # Normalize the scores
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(predictions_pf.reshape(-1, 1)).flatten()

    final_predictions = np.maximum.reduce([predictions_pf])
    categories = categorize_scores(final_predictions)

    save_results_to_csv(y_test_pf, final_predictions, categories, algo_name, save_re_path, normalized_scores)