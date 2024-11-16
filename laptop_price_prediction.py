#Muhammad Hamza Ashfaq
#Importing necessary Libraries
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Loading the dataset
data = pd.read_csv('laptop_data.csv')

# Initial data exploration
print("First 10 rows of the dataset:")
print(data.head(10))
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())


def extract_numeric(text):
    match = re.search(r'(\d+) GB', text)
    if match:
        return int(match.group(1))
    else:
        return 0

data['ram_gb'] = data['ram_gb'].apply(extract_numeric)
data['ssd'] = data['ssd'].apply(extract_numeric)
data['hdd'] = data['hdd'].apply(extract_numeric)

data['hdd'].fillna(0)
data['ssd'].fillna(0)
data['ram_gb'].fillna(0)

# Dropping unnecessary columns
data = data.drop(['brand', 'warranty', 'rating', 'Number of Ratings', 'Number of Reviews'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Displaying the modified dataset to show feature engineering
print("\nData after Feature Engineering:")
print(data.head(10))

# Splitting the data
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
# Defining parameter grids for both RandomForest and GradientBoosting
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None]
}
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Setting up GridSearch for RandomForest
rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_absolute_error')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Setting up GridSearch for GradientBoosting
gb = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_mean_absolute_error')
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_

# Evaluating both models on the test data
y_pred_rf = best_rf.predict(X_test)
y_pred_gb = best_gb.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))


print("\nRandom Forest Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")

print("\nGradient Boosting Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae_gb}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb}")

# Choosing the best model
if mae_gb < mae_rf:
    best_model = best_gb
    model_name = 'Gradient Boosting Regressor'
else:
    best_model = best_rf
    model_name = 'Random Forest Regressor'

print(f"\nBest Model: {model_name}")

# Cross-validation score for the selected best model
cross_val_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"\nCross-Validation Mean Absolute Error for {model_name}: {-cross_val_scores.mean()}")

# Saving the best model
joblib.dump(best_model, 'best_laptop_price_model.pkl')
print("\nBest model saved as 'best_laptop_price_model.pkl'")
