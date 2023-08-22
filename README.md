# Medical_Cost_Analysis
Akbank Machine Learning Bootcamp Project
<p>Created by Burak Bayraktar</p>


<!-- Import required libraries -->
<code>
import pandas as pd  <!-- Pandas library for data manipulation. -->
import numpy as np  <!-- NumPy library for numerical operations. -->
import seaborn as sns  <!-- Seaborn library for data visualization. -->
import matplotlib.pyplot as plt  <!-- Matplotlib library for plotting. -->
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  <!-- Tools for model evaluation and tuning. -->
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  <!-- Data preprocessing tools. -->
from sklearn.linear_model import LinearRegression  <!-- Linear regression model. -->
from sklearn.tree import DecisionTreeRegressor  <!-- Decision tree regression model. -->
from sklearn.ensemble import RandomForestRegressor  <!-- Random forest regression model. -->
from sklearn.metrics import mean_squared_error, mean_absolute_error  <!-- Error metrics. -->
</code>

<!-- Load the dataset and analyze BMI distribution -->
<code>
# Load the dataset
data = pd.read_csv("health_insurance_dataset.csv")

# Examine the distribution of BMI
plt.figure(figsize=(8, 6))
sns.histplot(data['bmi'], bins=30, kde=True)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()
</code>

<!-- Perform Label Encoding for categorical variables -->
<code>
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
</code>

<!-- Perform One-Hot Encoding for 'region' variable -->
<code>
data = pd.get_dummies(data, columns=['region'], drop_first=True)
</code>

<!-- Split the dataset into features and target -->
<code>
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</code>

<!-- Scale the dataset using Standard Scaling -->
<code>
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
</code>

<!-- Initialize and train models -->
<code>
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    results[name] = np.sqrt(-scores.mean())

for name, score in results.items():
    print(f'{name}: RMSE = {score}')
</code>

<!-- Hyperparameter tuning using Grid Search -->
<code>
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
</code>

<!-- Evaluate the optimized model -->
<code>
y_pred = best_rf_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
</code>

mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
</code>

