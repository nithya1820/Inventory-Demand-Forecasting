import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer
import streamlit as st
import matplotlib.pyplot as plt

# Step 1: Enhanced Dataset Generation
np.random.seed(42)
data = {
    'Product_ID': [f'P{i}' for i in range(1, 201)],  # Expanded dataset size
    'Warehouse': np.random.choice(['W1', 'W2', 'W3', 'W4'], size=200),
    'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], size=200),
    'Previous_Demand': np.random.randint(50, 800, size=200),  # Wider range for better variability
    'Supplier_Rating': np.random.uniform(1.5, 5.0, size=200),  # Lower minimum for more variability
    'Current_Stock': np.random.randint(0, 1500, size=200),  # Larger range for realistic inventory
    'Promotions': np.random.choice([0, 1], size=200, p=[0.6, 0.4]),
    'Season': np.random.choice(['Summer', 'Winter', 'Spring', 'Autumn'], size=200),
    'Demand': np.random.randint(80, 1200, size=200)  # Adjusted demand range
}

# Add correlated features for enhanced dataset
np.random.seed(42)
data['Stock_to_Demand_Ratio'] = data['Current_Stock'] / (data['Demand'] + 1)
data['Demand_Growth'] = np.random.uniform(-0.1, 0.3, size=200) * data['Previous_Demand']
data['Weekend_Flag'] = np.random.choice([0, 1], size=200, p=[0.8, 0.2])

# Convert to DataFrame
df_raw = pd.DataFrame(data)

# Introduce inconsistencies
for _ in range(20):  # Add missing values
    df_raw.loc[np.random.randint(0, 200), np.random.choice(df_raw.columns)] = np.nan

df_raw = pd.concat([df_raw, df_raw.iloc[:10]])  # Add duplicates

df_raw.loc[np.random.randint(0, len(df_raw)), 'Demand'] = 5000  # Add an extreme outlier

# Step 2: Train model on raw dataset
X_raw = df_raw.drop(columns=['Product_ID', 'Demand'])
y_raw = df_raw['Demand']

# Handle missing values in target variable
y_raw = y_raw.fillna(y_raw.mean())

# Encode categorical variables
X_raw = pd.get_dummies(X_raw, drop_first=True)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
model_raw = XGBRegressor(random_state=42, objective='reg:squarederror')
model_raw.fit(X_train_raw, y_train_raw)

y_pred_raw = model_raw.predict(X_test_raw)
rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
r2_raw = r2_score(y_test_raw, y_pred_raw)

# Step 3: Advanced Preprocessing
# Data Cleaning: Drop duplicates
df_clean = df_raw.drop_duplicates()

# Handle missing values
df_clean = df_clean.apply(lambda col: col.fillna(col.mean()) if col.dtypes != 'object' else col.fillna(col.mode()[0]))

# Handle outliers: Cap at 1st and 99th percentiles
for col in ['Previous_Demand', 'Current_Stock', 'Demand']:
    lower_limit = df_clean[col].quantile(0.01)
    upper_limit = df_clean[col].quantile(0.99)
    df_clean[col] = np.clip(df_clean[col], lower_limit, upper_limit)

# Data Transformation: Use robust scaling
numerical_cols = ['Previous_Demand', 'Supplier_Rating', 'Current_Stock', 'Stock_to_Demand_Ratio', 'Demand_Growth']
scaler = RobustScaler()
df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

# Data Discretization: Bin numerical columns into categories
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
binned_cols = ['Previous_Demand', 'Supplier_Rating']
df_clean[binned_cols] = discretizer.fit_transform(df_clean[binned_cols])

# Data Reduction: Drop less important columns
df_clean = df_clean.drop(columns=['Product_ID', 'Day_of_Week'])

# Save preprocessed dataset to CSV
df_clean.to_csv('preprocessed_inventory_data.csv', index=False)

# Step 4: Train model on preprocessed dataset
X_clean = df_clean.drop(columns=['Demand'])
y_clean = df_clean['Demand']

# Encode categorical variables
X_clean = pd.get_dummies(X_clean, drop_first=True)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Model Hyperparameter Tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(XGBRegressor(random_state=42, objective='reg:squarederror'), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_clean, y_train_clean)

best_model = grid_search.best_estimator_
y_pred_clean = best_model.predict(X_test_clean)
rmse_clean = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
r2_clean = r2_score(y_test_clean, y_pred_clean)

# Step 5: Visualization and Model Comparison
st.title("Inventory Demand Forecasting")

st.header("Raw Dataset")
st.write("Before Preprocessing")
st.write(df_raw.describe())
fig, ax = plt.subplots()
df_raw['Demand'].hist(bins=20, ax=ax)
ax.set_title("Demand Distribution")
st.pyplot(fig)
st.write(f"Raw Model Performance - RMSE: {rmse_raw}, R2: {r2_raw}")

st.header("Preprocessed Dataset")
st.write("After Preprocessing")
st.write(df_clean.describe())
fig, ax = plt.subplots()
df_clean['Demand'].hist(bins=20, ax=ax)
ax.set_title("Demand Distribution")
st.pyplot(fig)
st.write(f"Preprocessed Model Performance - RMSE: {rmse_clean}, R2: {r2_clean}")

improvement_rmse = rmse_raw - rmse_clean
improvement_r2 = r2_clean - r2_raw

st.header("Model Comparison")
st.write(f"Improvement in RMSE: {improvement_rmse}")
st.write(f"Improvement in R2: {improvement_r2}")

# Plot RMSE comparison
fig, ax = plt.subplots()
labels = ['Raw', 'Preprocessed']
rmse_values = [rmse_raw, rmse_clean]
ax.bar(labels, rmse_values, color='blue', alpha=0.7)
ax.set_title("RMSE Comparison")
ax.set_ylabel("RMSE")
st.pyplot(fig)

# Plot R2 comparison
fig, ax = plt.subplots()
r2_values = [r2_raw, r2_clean]
ax.bar(labels, r2_values, color='green', alpha=0.7)
ax.set_title("R2 Comparison")
ax.set_ylabel("R2")
st.pyplot(fig)

# Plot Feature Importance
st.header("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(best_model, ax=ax, importance_type='weight')
ax.set_title("Feature Importance")
st.pyplot(fig)
