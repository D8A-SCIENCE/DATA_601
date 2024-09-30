
# Lab Instructions: Support Vector Regression (SVR)

## Objectives
In this lab, you'll learn how to:
- Implement a support vector regression (SVR) model to predict student alcohol consumption on weekends.
- Prepare the data by selecting relevant features and encoding categorical variables.

## Step-by-Step Guide

### 1. Import Libraries
First, import the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
```

### 2. Data Preparation
#### Select Relevant Features
Select a subset of features from the dataset based on theoretical importance:

```python
selected_features = ['age', 'address', 'traveltime', 'failures', 'higher', 'internet', 
                     'romantic', 'famrel', 'freetime', 'goout', 'absences']
X = all_students[selected_features].values
```

#### Encode Categorical Variables
Use `LabelEncoder` to convert categorical features to numeric:

```python
discreteCoder_X = LabelEncoder()
categorical_columns = ['address', 'higher', 'internet', 'romantic']

for col in categorical_columns:
    X[:, selected_features.index(col)] = discreteCoder_X.fit_transform(X[:, selected_features.index(col)])
```

### 3. Train-Test Split
Split the data into training and test sets:

```python
y = all_students['Walc'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)
```

### 4. Data Scaling
Scale the features to ensure the SVR model works optimally:

```python
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)
```

### 5. Train the SVR Model
Create and train the SVR model:

```python
svr_regression = SVR(kernel='linear', epsilon=1.0)
svr_regression.fit(X_train, y_train)
```

### 6. Make Predictions
Predict for a hypothetical student:

```text
Age: 18
Address: Urban (label encoded as 1)
Travel Time: 3 (30 minutes to 1 hour)
Failures: 3
Desire for Higher Education: No (0)
Internet Access: No (0)
Romantic Relationship: Yes (1)
Relationship with Family: (4 out of 5)
Freetime: A little (3 out of 5)
Going Out: A little (3 out of 5)
Absences: 5
```
Convert this hypothetical student's data to a numpy array and scale it:
```python
new_student = np.array([18, 1, 3, 3, 0, 0, 1, 4, 3, 3, 5]).reshape(1, -1)
new_student_scaled = scale_X.transform(new_student)
prediction = svr_regression.predict(new_student_scaled)
print(f"Predicted weekend alcohol consumption: {prediction[0]:.2f}")
```

### Summary
You've successfully implemented an SVR model to predict alcohol consumption, and you learned how to encode data, scale features, and make predictions.

