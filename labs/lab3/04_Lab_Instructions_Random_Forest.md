
# Lab Instructions: Random Forests and Model Validation

## Objectives
In this lab, you'll learn how to:
- Implement a random forest regression model.
- Compare different models using mean absolute error (MAE).

## Step-by-Step Guide

### 1. Import Libraries
First, import the necessary libraries:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

### 2. Train the Random Forest Regressor
Create and train the random forest model:

```python
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=1693)
rf_regressor.fit(X_train, y_train)
```

### 3. Make Predictions
Predict for hypothetical students:

```python
prediction_a = rf_regressor.predict(new_student_scaled)
print(f"Predicted weekend alcohol consumption for Student A: {prediction_a[0]:.2f}")
```

### 4. Evaluate Model Performance
Calculate the mean absolute error for the random forest model:

```python
rf_mae = mean_absolute_error(y_test, rf_regressor.predict(X_test))
print(f"Random Forest Mean Absolute Error: {rf_mae:.2f}")
```

### Summary
You’ve successfully implemented a random forest model, predicted student behavior, and evaluated model performance using MAE.

