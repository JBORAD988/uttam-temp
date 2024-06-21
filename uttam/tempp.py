from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import os


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    ####### PCA CODE START #######
    n_components = min(len(numerical_cols), X.shape[0])

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
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols


def print_best_and_worst_features(model, feature_names, algo_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        max_index = np.argmax(importances)
        min_index = np.argmin(importances)
        print(
            f"\nHighest impact feature for {algo_name}: {feature_names[max_index]} (importance: {importances[max_index]:.4f})")
        print(
            f"Lowest impact feature for {algo_name}: {feature_names[min_index]} (importance: {importances[min_index]:.4f})")
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if hasattr(coefs, 'toarray'):
            coefs = coefs.toarray().flatten()
        max_index = np.argmax(np.abs(coefs))
        min_index = np.argmin(np.abs(coefs))
        print(
            f"\nHighest impact feature for {algo_name}: {feature_names[max_index]} (coefficient: {coefs[max_index]:.4f})")
        print(
            f"Lowest impact feature for {algo_name}: {feature_names[min_index]} (coefficient: {coefs[min_index]:.4f})")
    else:
        print(f"{algo_name} does not provide feature importances or coefficients.")


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name,
                       model_save_path):
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

    # Print best parameters found by GridSearchCV
    print(f"Best Parameters for {algo_name}: {grid_search.best_params_}")
    print("MSE:", mse, "R2:", r2)

    # Access the PCA step if it exists
    pca = None
    if 'pca' in best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps:
        pca = best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['pca']

    # Print best and worst features
    feature_names = list(preprocessor.named_transformers_['num'].get_feature_names_out()) + \
                    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())
    print_best_and_worst_features(best_model.named_steps['regressor'], feature_names, algo_name)

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


def save_results_to_csv(y_test, final_predictions, categories, algo_name, save_re_path):
    results_df = pd.DataFrame({
        'Actual Score': list(y_test),
        'Predicted Score': final_predictions,
        'Category': categories
    })

    results_df.to_csv(os.path.join(save_re_path, f'model_predictions_{algo_name.replace(" ", "_").lower()}.csv'),
                      index=False)
    print(f"Results have been exported to 'model_predictions_{algo_name.replace(' ', '_').lower()}.csv'.")


# Additional algorithms to try
algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    ('Random Forest', RandomForestRegressor(),
     {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]}),
    ('SVR', SVR(), {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}),
    ('Decision Tree', DecisionTreeRegressor(), {'regressor__max_depth': [None, 5, 10, 20]}),
    ('Elastic Net', ElasticNet(), {'regressor__alpha': [0.1, 0.5, 1.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    ('Lasso', Lasso(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    ('Ridge', Ridge(), {'regressor__alpha': [0.1, 0.5, 1.0]}),
    ('Linear Regression', LinearRegression(), {})
]

final_df_path = "../data/final.csv"
model_save_path = '../models'
save_re_path = '../save_results'

for algo_name, regressor, param_grid in algorithms:
    print(f"\nTraining and evaluating using {algo_name}...")

    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, numerical_cols_pf = load_and_preprocess_data(final_df_path, 'Score')

    predictions_pf, y_test_pf, pca_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor, param_grid, algo_name, model_save_path)



    # Print dataset features with PCA analysis
    print("\nFeatures for", algo_name)
    print_best_and_worst_features(X_train_pf, pca_pf, numerical_cols_pf)

    # Save predictions for each model
    joblib.dump(predictions_pf,
                os.path.join(model_save_path, f'predictions_{algo_name.replace(" ", "_").lower()}.joblib'))

    final_predictions = np.maximum.reduce([predictions_pf])
    categories = categorize_scores(final_predictions)

    save_results_to_csv(y_test_pf, final_predictions, categories, algo_name, save_re_path)
