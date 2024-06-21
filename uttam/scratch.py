import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import os
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    n_samples = X.shape[0]
    n_features = len(numerical_cols)
    n_components = min(n_features, n_samples, 3)

    numerical_transformer_steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]

    if n_components > 1:
        numerical_transformer_steps.append(('pca', PCA(n_components=n_components)))

    numerical_transformer = Pipeline(steps=numerical_transformer_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols


def print_dataset_features(X, pca, numerical_cols):
    data = []
    if pca is not None:
        components = pca.components_
        for i, component in enumerate(components):
            feature_contributions = zip(numerical_cols, component)
            sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
            for feature, loading in sorted_features:
                data.append([f"Component {i + 1}", feature, loading])

    df = pd.DataFrame(data, columns=["PCA Component", "Feature", "Loading"])
    print(df)


def plot_pca_analysis(pca, numerical_cols, ax, title, color):
    explained_variance_ratio = pca.explained_variance_ratio_

    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color=color, alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(title)
    ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
    ax.grid(True)


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, regressor, param_grid, algo_name,
                       model_save_path):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)])

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    joblib.dump(best_model, os.path.join(model_save_path, f'best_model_{algo_name.replace(" ", "_").lower()}.joblib'))

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)

    pca = None
    if 'pca' in best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps:
        pca = best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['pca']

    return y_pred, y_test, pca, mse, r2, mae


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


def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


algorithms = [
    ('Gradient Boosting', GradientBoostingRegressor(), {'regressor__n_estimators': [40, 50, 100, 150]}),
    ('Random Forest', RandomForestRegressor(),
     {'regressor__n_estimators': [40, 50, 100, 150], 'regressor__max_depth': [None, 10, 20]})
]

final_df_path = "../data/preprocessed_data.csv"
model_save_path = '../models'
save_re_path = '../save_results'

metrics_data = []

# Prepare to plot PCA analysis for each algorithm and combined models
#fig, axes = plt.subplots(1, len(algorithms) + 1, figsize=(20, 5), sharey=True)
#colors = ['blue', 'green', 'red']

# Iterate over each model
for idx, (algo_name, regressor, param_grid) in enumerate(algorithms):
    print(f"\nTraining and evaluating using {algo_name}...")

    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf, numerical_cols_pf = load_and_preprocess_data(
        final_df_path, 'Score')

    predictions_pf, y_test_pf, pca_pf, mse, r2, mae = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf,
                                                                         preprocessor_pf, regressor, param_grid,
                                                                         algo_name, model_save_path)

    # Print dataset features with PCA analysis
    print("\nFeatures for", algo_name)
    print_dataset_features(X_train_pf, pca_pf, numerical_cols_pf)

    # Save predictions for each model
    joblib.dump(predictions_pf,
                os.path.join(model_save_path, f'predictions_{algo_name.replace(" ", "_").lower()}.joblib'))

    # Display results for each model
    final_predictions = np.maximum.reduce([predictions_pf])
    categories = categorize_scores(final_predictions)

    save_results_to_csv(y_test_pf, final_predictions, categories, algo_name, save_re_path)

    # Collect metrics for tabular display
    metrics_data.append([algo_name, mse, r2, mae])

    # Plot PCA analysis for the current algorithm
    #if pca_pf is not None:
      #  plot_pca_analysis(pca_pf, numerical_cols_pf, axes[idx], f"PCA Analysis for {algo_name}", colors[idx])

# Combine predictions from both models (simple averaging)
predictions_gb = joblib.load(os.path.join(model_save_path, 'predictions_gradient_boosting.joblib'))
predictions_rf = joblib.load(os.path.join(model_save_path, 'predictions_random_forest.joblib'))

final_predictions_combined = (predictions_gb + predictions_rf) / 2

# Categorize scores for combined predictions
categories_combined = categorize_scores(final_predictions_combined)

# Save results for combined predictions
save_results_to_csv(y_test_pf, final_predictions_combined, categories_combined, 'combined_models', save_re_path)

# Evaluate combined models MSE and R2
mse_combined = mean_squared_error(y_test_pf, final_predictions_combined)
mae_combined = mean_absolute_error(y_test_pf, final_predictions_combined)
actual_categories = categorize_scores(y_test_pf)
accuracy_combined = calculate_accuracy(actual_categories, categories_combined)

print("Combined Models MSE:", mse_combined, "R2:", accuracy_combined, "MAE:", mae_combined)

# Collect combined model metrics for tabular display
metrics_data.append(['Combined Model', mse_combined, accuracy_combined, mae_combined])

# Create a DataFrame for model evaluation metrics
metrics_df = pd.DataFrame(metrics_data, columns=["Model", "MSE", "R2/Accuracy", "MAE"])
print("\nModel Evaluation Metrics:")
print(metrics_df)

# Plot combined PCA analysis
#print("\nCombined Model PCA Analysis")
#plot_pca_analysis(pca_pf, numerical_cols_pf, axes[-1], "PCA Analysis for Combined Models", colors[-1])

# Create bar plots for MSE, R2, and MAE
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

metrics_df.plot(kind='bar', x='Model', y='MSE', ax=ax[0], color='blue', alpha=0.7, legend=False)
ax[0].set_title('Mean Squared Error (MSE)')
ax[0].set_ylabel('MSE')

metrics_df.plot(kind='bar', x='Model', y='R2/Accuracy', ax=ax[1], color='green', alpha=0.7, legend=False)
ax[1].set_title('R2 Score / Accuracy')
ax[1].set_ylabel('R2 / Accuracy')

metrics_df.plot(kind='bar', x='Model', y='MAE', ax=ax[2], color='red', alpha=0.7, legend=False)
ax[2].set_title('Mean Absolute Error (MAE)')
ax[2].set_ylabel('MAE')

plt.tight_layout()
plt.show()
