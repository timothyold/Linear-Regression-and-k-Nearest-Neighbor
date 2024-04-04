# Linear-Regression-and-k-Nearest-Neighbor

This README provides an overview of two Python scripts for data analysis and machine learning modeling. Below are the details for each file.

## Linear Regression Analysis

### Overview
This script performs a linear regression analysis to predict unemployment rates. It includes data visualization, data preprocessing, model training, prediction, and evaluation.

### Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

### Data
- The dataset is loaded from `dataset_1.csv`.
- It uses a heatmap to visualize correlations between features.

### Processing Steps
1. **Data Loading**: Load the dataset using pandas.
2. **Visualization**: Generate a heatmap of the correlations between features.
3. **Data Preprocessing**: Scale the features using Min-Max scaling.
4. **Model Training**: Split the dataset into training and test sets, then train a Linear Regression model.
5. **Prediction**: Make predictions on the test set.
6. **Evaluation**: Calculate and print the regression score, coefficients, intercept, and mean squared error.
7. **Output Visualization**: Plot a scatter plot comparing true and predicted unemployment rates.

## K-Nearest Neighbors Classification

### Overview
This script uses the K-Nearest Neighbors (KNN) algorithm to classify data into categories. It evaluates the model's performance with different values of K and distance metrics.

### Dependencies
- pandas
- numpy
- scikit-learn

### Data
- The dataset is loaded from `dataset_2.csv`.
- The data is split into training and test sets with a 80-20 split.

### Processing Steps
1. **Data Loading**: Load the dataset using pandas.
2. **Data Splitting**: Split the dataset into training and test sets.
3. **Data Preprocessing**: Scale the features using Min-Max scaling.
4. **Model Training and Evaluation**:
    - Define a function `evaluate_knn` to train KNN models with different values of K and distance metrics, and calculate misclassification rates.
    - Evaluate the model using Euclidean and Manhattan distances for K values from 2 to 5.
5. **Output**: Print the misclassification rates for different K values and distance metrics.

### How to Run
1. Ensure all dependencies are installed.
2. Run each script in a Python environment, adjusting file paths as necessary.

### Notes
- These scripts are examples of basic machine learning tasks: regression and classification.
- Customization of parameters and further tuning of models is recommended for optimal performance.
