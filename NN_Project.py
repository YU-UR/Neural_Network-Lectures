import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# The dataset contains hourly energy prices and related features. Ensure the correct file path is provided.
file_path = "/content/SpotAllData_2015_2016.csv"  # Update this path as needed
data = pd.read_csv(file_path)

# Step 2: Convert 'Date' column to datetime and set it as the index
# This step helps in treating the data as time series, enabling time-based operations.
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')
data.set_index('Date', inplace=True)

# Step 3: Strip column names of trailing/leading whitespace
# Prevents issues caused by unexpected whitespace in column names.
data.columns = data.columns.str.strip()

# Step 4: Interpolate missing values
# Missing values are filled using time-based interpolation, ensuring continuity in the time series.
data_cleaned = data.interpolate(method='time', limit_direction='both')

# Step 5: Feature engineering - Lagged and rolling features
# Add lagged and rolling average features to capture temporal patterns in energy prices.
data_cleaned['Lagged_Price_1H'] = data_cleaned['EPEX SPOT DA_DE/AUT (?/MWh)'].shift(1)
data_cleaned['Lagged_Price_24H'] = data_cleaned['EPEX SPOT DA_DE/AUT (?/MWh)'].shift(24)
data_cleaned['Rolling_Avg_3H'] = data_cleaned['EPEX SPOT DA_DE/AUT (?/MWh)'].rolling(window=3).mean()
data_cleaned['Rolling_Avg_24H'] = data_cleaned['EPEX SPOT DA_DE/AUT (?/MWh)'].rolling(window=24).mean()

# Step 6: Drop rows with NaN values after feature engineering
# Dropping rows with NaN values is necessary after creating lagged and rolling features
# because these operations introduce NaN values in the initial rows of the dataset.
data_cleaned = data_cleaned.dropna()

# Step 7: Define target and predictors
# Define separate targets for 1-hour ahead and 24-hour ahead forecasts.
target_1H = 'EPEX SPOT DA_DE/AUT (?/MWh)'
target_24H = 'Lagged_Price_24H'
predictors = [
    'EXXA-Data',
    'PHELIX Futures Base FB01',
    'PHELIX Futures Peak FP01',
    'forecasted Load DE',
    'actual Load DE',
    'Load CZ FC',
    'Gen AUT FC',
    'Gen CZ FC',
    'Gen GER FC',
    'Gen FRA FC',
    'Wind Gen DE FC',
    'PV Gen DE FC',
    'Cos(Hour)',
    'Sin(Hours)',
    'Lagged_Price_1H',
    'Rolling_Avg_3H',
    'Rolling_Avg_24H'
]

# Step 8: Separate predictors and target variables
X = data_cleaned[predictors]
y_1H = data_cleaned[target_1H]
y_24H = data_cleaned[target_24H]

# Step 9: Train-test split
# Split data into training and testing sets (80/20 split) without shuffling to maintain temporal order.
X_train, X_test, y_1H_train, y_1H_test = train_test_split(X, y_1H, test_size=0.2, random_state=42, shuffle=False)
_, _, y_24H_train, y_24H_test = train_test_split(X, y_24H, test_size=0.2, random_state=42, shuffle=False)

# Step 10: Standardize features
# Scaling ensures all features contribute equally to models like MLP and Gradient Boosting.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 11: Handle non-numeric or invalid values
# Replace '#WERT!' and other invalid values with NaN, then impute missing values with column means.
data_cleaned = data_cleaned.replace("#WERT!", np.nan)
data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')
data_cleaned = data_cleaned.dropna()
data_cleaned = data_cleaned.fillna(data_cleaned.mean())

# Step 12: Visualize correlation matrix
# Correlation heatmap helps identify relationships between variables and potential multicollinearity issues.
corr_matrix = data_cleaned.corr()
plt.figure(figsize=(8, 4))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Step 13: Linear Regression Model
# Baseline model to evaluate performance compared to more complex models.
lr_model_1H = LinearRegression()
lr_model_1H.fit(X_train, y_1H_train)
lr_predictions_1H = lr_model_1H.predict(X_test)
lr_metrics_1H = {
    "MAE": mean_absolute_error(y_1H_test, lr_predictions_1H),
    "RMSE": np.sqrt(mean_squared_error(y_1H_test, lr_predictions_1H)),
    "R-squared": r2_score(y_1H_test, lr_predictions_1H)
}

lr_model_24H = LinearRegression()
lr_model_24H.fit(X_train, y_24H_train)
lr_predictions_24H = lr_model_24H.predict(X_test)
lr_metrics_24H = {
    "MAE": mean_absolute_error(y_24H_test, lr_predictions_24H),
    "RMSE": np.sqrt(mean_squared_error(y_24H_test, lr_predictions_24H)),
    "R-squared": r2_score(y_24H_test, lr_predictions_24H)
}

# Step 14: Neural Network (MLP)
# MLP captures non-linear relationships in the data for potentially improved accuracy.
mlp_model_1H = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                            alpha=0.001, batch_size=64, learning_rate='adaptive',
                            max_iter=200, random_state=42)
mlp_model_1H.fit(X_train_scaled, y_1H_train)
mlp_predictions_1H = mlp_model_1H.predict(X_test_scaled)
mlp_metrics_1H = {
    "MAE": mean_absolute_error(y_1H_test, mlp_predictions_1H),
    "RMSE": np.sqrt(mean_squared_error(y_1H_test, mlp_predictions_1H)),
    "R-squared": r2_score(y_1H_test, mlp_predictions_1H)
}

mlp_model_24H = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                             alpha=0.001, batch_size=64, learning_rate='adaptive',
                             max_iter=200, random_state=42)
mlp_model_24H.fit(X_train_scaled, y_24H_train)
mlp_predictions_24H = mlp_model_24H.predict(X_test_scaled)
mlp_metrics_24H = {
    "MAE": mean_absolute_error(y_24H_test, mlp_predictions_24H),
    "RMSE": np.sqrt(mean_squared_error(y_24H_test, mlp_predictions_24H)),
    "R-squared": r2_score(y_24H_test, mlp_predictions_24H)
}

# Step 15: Stacking Model
# Combines predictions from multiple base models using a meta-model for improved performance.
base_models = [
    ('lr', LinearRegression()),
    ('rf', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
]
meta_model = GradientBoostingRegressor(n_estimators=50, random_state=42)

stacking_model_1H = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model_1H.fit(X_train_scaled, y_1H_train)
stacking_predictions_1H = stacking_model_1H.predict(X_test_scaled)
stacking_metrics_1H = {
    "MAE": mean_absolute_error(y_1H_test, stacking_predictions_1H),
    "RMSE": np.sqrt(mean_squared_error(y_1H_test, stacking_predictions_1H)),
    "R-squared": r2_score(y_1H_test, stacking_predictions_1H)
}

stacking_model_24H = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model_24H.fit(X_train_scaled, y_24H_train)
stacking_predictions_24H = stacking_model_24H.predict(X_test_scaled)
stacking_metrics_24H = {
    "MAE": mean_absolute_error(y_24H_test, stacking_predictions_24H),
    "RMSE": np.sqrt(mean_squared_error(y_24H_test, stacking_predictions_24H)),
    "R-squared": r2_score(y_24H_test, stacking_predictions_24H)
}

# Step 16: Model Comparison
# Compare performance metrics of different models to determine the best approach.
model_comparison_1H = pd.DataFrame({
    "Model": ["Linear Regression", "Neural Network (MLP)", "Stacking Model"],
    "Mean Absolute Error (MAE)": [lr_metrics_1H["MAE"], mlp_metrics_1H["MAE"], stacking_metrics_1H["MAE"]],
    "Root Mean Squared Error (RMSE)": [lr_metrics_1H["RMSE"], mlp_metrics_1H["RMSE"], stacking_metrics_1H["RMSE"]],
    "R-squared": [lr_metrics_1H["R-squared"], mlp_metrics_1H["R-squared"], stacking_metrics_1H["R-squared"]]
})

model_comparison_24H = pd.DataFrame({
    "Model": ["Linear Regression", "Neural Network (MLP)", "Stacking Model"],
    "Mean Absolute Error (MAE)": [lr_metrics_24H["MAE"], mlp_metrics_24H["MAE"], stacking_metrics_24H["MAE"]],
    "Root Mean Squared Error (RMSE)": [lr_metrics_24H["RMSE"], mlp_metrics_24H["RMSE"], stacking_metrics_24H["RMSE"]],
    "R-squared": [lr_metrics_24H["R-squared"], mlp_metrics_24H["R-squared"], stacking_metrics_24H["R-squared"]]
})

print("Model Comparison for 1-hour ahead forecast:")
print(model_comparison_1H)
print("\nModel Comparison for 24-hour ahead forecast:")
print(model_comparison_24H)

# Step 17: Feature Importance (Stacking Model for 1H and 24H)
# Visualize feature importances from the meta-model to understand key predictors.
if hasattr(meta_model, 'feature_importances_'):
    feature_importances_1H = pd.DataFrame({
        'Feature': predictors,
        'Importance': meta_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances_1H, x='Importance', y='Feature')
    plt.title('Feature Importances (Stacking Model Meta-Model - 1 Hour)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Step 18: Save results
# Save the model comparison results to a file for future reference.
model_comparison_1H.to_csv("model_comparison_1H_results.csv", index=False)
model_comparison_24H.to_csv("model_comparison_24H_results.csv", index=False)
