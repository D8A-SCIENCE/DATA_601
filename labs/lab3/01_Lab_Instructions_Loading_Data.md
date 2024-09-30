
# Lab Instructions: Loading and Preparing Data

## Objectives
In this lab, you'll learn how to:
- Load two datasets containing student alcohol consumption information.
- Merge them into a single dataframe for analysis.

## Prerequisites
You should be familiar with the following Python libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualization.

## Step-by-Step Guide

### 1. Import Libraries
First, import the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

These libraries are essential for data handling, numerical computations, and plotting.

### 2. Load the Data
The datasets you need are named `studentmat.csv` (for Math students) and `studentpor.csv` (for Portuguese students). They are available on Kaggle or through the provided files.

Load the datasets using Pandas:

```python
students_math = pd.read_csv('./data/studentmat.csv')
students_port = pd.read_csv('./data/studentpor.csv')
```

### 3. Inspect the Data
To get an initial look at the Math dataset, use:

```python
students_math.head()
```

This will display the first few rows, allowing you to get a sense of the data structure.

### 4. Check Column Names and Metadata
If you're unsure what each column represents, you can refer to the metadata provided on Kaggle, where all column definitions are available.

You can also list all column names in Python:

```python
list(students_math.columns)
```

This command provides a list of all the variables (column names) in the Math dataset.

### 5. Determine Dataset Size
To determine the number of observations (students) and features (variables) in each dataset, use the `.shape` attribute:

```python
students_math.shape
students_port.shape
```

For example, if `students_math.shape` returns `(395, 33)`, it means there are 395 observations and 33 features in the dataset.

### 6. Merge the Dataframes
Since there are two datasets (Math and Portuguese), we need to merge them to perform analysis on all students.

Use the Pandas `concat` function to merge the datasets:

```python
all_students = pd.concat([students_math, students_port], ignore_index=True)
```

The parameter `ignore_index=True` ensures that the index is reset to avoid redundancy, since both datasets start indexing at 0.

### 7. Verify the Merged Dataset
Check if the merge was successful by looking at the new shape of the merged dataframe:

```python
all_students.shape
```

The expected output should show 1044 observations and 33 columns, confirming that the two datasets have been combined.

### 8. Summary
You have now successfully loaded and merged the data for the Math and Portuguese classes.

You’re ready to proceed to the next steps, which involve building machine learning models to predict student alcohol consumption.

## Next Steps
Move to 02_Lab_Instructions_SVR.md