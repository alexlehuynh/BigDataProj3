import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Question 1: Read the CSV file
df = pd.read_csv('breast-cancer.csv')

# Question 2: Remove rows with missing values
df = df.dropna()

# Split the dataset into features (X) and target (y)
X = df.drop('diagnosis', axis=1)  # assuming 'diagnosis' is the target column
y = df['diagnosis']

# Split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Question 3: Create a Random Forest Classifier
rf = RandomForestClassifier()

# Train the classifier
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Print feature importances
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature}, Importance: {importance}")

# Get the top two features
top_two_features = X.columns[importances.argsort()[::-1][:2]]

# Plot the top two features
sns.scatterplot(data=df, x=top_two_features[0], y=top_two_features[1], hue='diagnosis')
plt.show()

# Question 6: Remove the feature with the lowest importance
X_train_reduced = X_train.drop(X.columns[importances.argmin()], axis=1)
X_test_reduced = X_test.drop(X.columns[importances.argmin()], axis=1)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Track the training time
start = time.time()

# Train the classifier
clf.fit(X_train_reduced, y_train)

# Print the training time
print("Training time: ", time.time() - start)

# Predict the response for test dataset
y_pred = clf.predict(X_test_reduced)

# Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Model Precision, Recall
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Question 4: Create a SVM Classifier with RBF kernel
clf_svm = svm.SVC(kernel='rbf')

# Track the training time
start_svm = time.time()

# Train the classifier
clf_svm.fit(X_train, y_train)

# Print the training time
print("SVM Training time: ", time.time() - start_svm)

# Predict the response for test dataset
y_pred_svm = clf_svm.predict(X_test)

# Model Accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Model Precision, Recall
print("SVM Metrics:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrix
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))