import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('dataset_2.csv')

# Splitting the dataset into 20% test and 80% training set
test_data = df.iloc[:int(0.2*len(df))]
train_data = df.iloc[int(0.2*len(df)):]

# Separating features and variables
X_train = train_data.drop(['CASE_ID', 'FRAUD'], axis=1)
y_train = train_data['FRAUD']
X_test = test_data.drop(['CASE_ID', 'FRAUD'], axis=1)
y_test = test_data['FRAUD']

# Scaling the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate KNN
def evaluate_knn(k_values, distance, X_train, y_train, X_test, y_test):
    misclass_rates = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance, weights='uniform', algorithm='auto')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        misclass_rate = np.mean(y_pred != y_test)
        misclass_rates[k] = misclass_rate
    return misclass_rates

# Evaluating KNN for k=2 to k=5 using Euclidean distance
k_values = [2, 3, 4, 5]
euclidean_misclass_rates = evaluate_knn(k_values, 'euclidean', X_train_scaled, y_train, X_test_scaled, y_test)

# Using Manhattan distance
manhattan_misclass_rates = evaluate_knn(k_values, 'manhattan', X_train_scaled, y_train, X_test_scaled, y_test)

print("Euclidean Distance Misclassification Rates:", euclidean_misclass_rates)
print("Manhattan Distance Misclassification Rates:", manhattan_misclass_rates)
