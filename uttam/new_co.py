import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}, R2: {r2}")
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

def main():
    personal_factors_path = '../data/persoanl factor.csv'
    home_factors_path = '../data/home.csv'
    school_factors_path = '../data/school.csv'

    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf = load_and_preprocess_data(personal_factors_path, 'Score')
    X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf = load_and_preprocess_data(home_factors_path, 'Score')
    X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf = load_and_preprocess_data(school_factors_path, 'Score')

    # Choose additional regression algorithms
    regressor_gb = GradientBoostingRegressor(random_state=42)
    regressor_rf = RandomForestRegressor(random_state=42)
    regressor_svr = SVR()

    # Train and evaluate models
    predictions_pf_gb, y_test_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor_gb)
    predictions_hf_gb, y_test_hf = train_and_evaluate(X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf, regressor_gb)
    predictions_sf_gb, y_test_sf = train_and_evaluate(X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf, regressor_gb)

    predictions_pf_rf, y_test_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor_rf)
    predictions_hf_rf, y_test_hf = train_and_evaluate(X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf, regressor_rf)
    predictions_sf_rf, y_test_sf = train_and_evaluate(X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf, regressor_rf)

    predictions_pf_svr, y_test_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, regressor_svr)
    predictions_hf_svr, y_test_hf = train_and_evaluate(X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf, regressor_svr)
    predictions_sf_svr, y_test_sf = train_and_evaluate(X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf, regressor_svr)

    final_predictions_gb = np.maximum.reduce([predictions_pf_gb, predictions_hf_gb, predictions_sf_gb])
    final_predictions_rf = np.maximum.reduce([predictions_pf_rf, predictions_hf_rf, predictions_sf_rf])
    final_predictions_svr = np.maximum.reduce([predictions_pf_svr, predictions_hf_svr, predictions_sf_svr])

    categories_gb = categorize_scores(final_predictions_gb)
    categories_rf = categorize_scores(final_predictions_rf)
    categories_svr = categorize_scores(final_predictions_svr)

    for actual, predicted, category in zip(y_test_pf, final_predictions_gb, categories_gb):
        print(f"Actual: {actual}, Predicted: {predicted}, Category: {category}")

    # Create a DataFrame with the results
    results_df_gb = pd.DataFrame({
        'Actual Score': list(y_test_pf),
        'Predicted Score': final_predictions_gb,
        'Category': categories_gb
    })

    results_df_rf = pd.DataFrame({
        'Actual Score': list(y_test_pf),
        'Predicted Score': final_predictions_rf,
        'Category': categories_rf
    })

    results_df_svr = pd.DataFrame({
        'Actual Score': list(y_test_pf),
        'Predicted Score': final_predictions_svr,
        'Category': categories_svr
    })

    # Export the DataFrames to CSV files
    results_df_gb.to_csv('model_predictions_gb.csv', index=False)
    results_df_rf.to_csv('model_predictions_rf.csv', index=False)
    results_df_svr.to_csv('model_predictions_svr.csv', index=False)

    print("Results have been exported to 'model_predictions_gb.csv', 'model_predictions_rf.csv', and 'model_predictions_svr.csv'.")

if __name__ == "__main__":
    main()
