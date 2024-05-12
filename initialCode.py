import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

#pip install with command pip install --user scikit-learn matplotlib

# Read the CSV file
df = pd.read_csv('breast-cancer.csv')

# Remove rows with missing values
df = df.dropna()

# Split the dataset into features (X) and target (y)
X = df.drop('diagnosis', axis=1)  # assuming 'diagnosis' is the target column
y = df['diagnosis']

# Split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Track the training time
start = time.time()

# Train the classifier
clf.fit(X_train, y_train)

# Print the training time
print("Training time: ", time.time() - start)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Model Precision, Recall
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Draw the decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf, filled=True)
plt.show()

#ended at q3