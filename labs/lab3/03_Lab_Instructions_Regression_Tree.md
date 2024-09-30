
# Lab Instructions: Decision Tree Regression

## Objectives
In this lab, you'll learn how to:
- Implement a decision tree regression model to predict student alcohol consumption on weekends.
- Visualize the decision tree.

## Step-by-Step Guide

### 1. Import Libraries
First, import the necessary libraries:

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
```

### 2. Train the Decision Tree Regressor
Create and train the decision tree model:

```python
regressor = DecisionTreeRegressor(max_depth=3, random_state=1693)
regressor.fit(X_train, y_train)
```

### 3. Visualize the Decision Tree
Export the tree structure:

```python
tree.export_graphviz(regressor, out_file='tree.dot', feature_names=selected_features)
# You can visualize this using Graphviz or similar tools
```

### 4. Make Predictions
Predict for a hypothetical student:

```python
prediction = regressor.predict(new_student_scaled)
print(f"Predicted weekend alcohol consumption: {prediction[0]:.2f}")
```

### Summary
You've successfully implemented a decision tree regression model, visualized its structure, and used it for predictions.

