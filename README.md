# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data
2. Split Dataset into Training and Testing Sets
3. Train the Model Using Stochastic Gradient Descent (SGD)
4. Make Predictions and Evaluate Accuracy
5. Generate Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Jaya Abirami S
RegisterNumber:212223220038
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```


## Output:
![prediction of iris species using SGD Classifier](sam.png)
![{68A8F214-7397-4D9B-AD8A-8AC216907F50}](https://github.com/user-attachments/assets/02d60bdb-53f3-4f8f-beb6-b46ad0b3c492)

![{A5C568B0-1C81-4612-9CFD-6493EB6D927F}](https://github.com/user-attachments/assets/08f40ab8-ec6f-4a0f-ba89-a8c10c96fc2b)

![{33FD8B28-4337-4AD2-9156-14760E831725}](https://github.com/user-attachments/assets/769404c4-fb23-4c96-9bb7-421bd1a6636f)

![{C26DD5BC-ADC0-4D6D-88B7-8F18EF7847D6}](https://github.com/user-attachments/assets/d124966f-0fa6-4e11-ada7-16c7aa112185)






## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
