import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

# Question 1. Reading the file into a pandas dataframe
df = pd.read_csv('breast-cancer.csv')

# can remove this later just using for testing purposes
print("Dataframe shape after reading the CSV file:", df.shape)


# Question 2. Remove any row which contain empty cell(s) "Bad Data".  Split into 80% training set and 20% testing

df = df.dropna() #removes rows with empty cells

print("Dataframe shape after removing rows with empty cells:", df.shape) #there are no empty cells

# Split the dataset into features (X) and target (y)
X = df.drop('diagnosis', axis=1)  # assuming 'diagnosis' is the target column
y = df['diagnosis']

# Split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Question 3. Modeling and Evaluations - Decision Tree Classifier
# Train the dataset on the Decision Tree Classifier using the training set
start_time = time.time()
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)
end_time = time.time()
print("Training time (Decision Tree):", end_time - start_time)

# Draw the decision tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_tree, filled=True)
plt.show()

# Evaluate the model using the testing data
Y_pred_tree = clf_tree.predict(X_test)
print("Classification report (Decision Tree):\n", classification_report(y_test, Y_pred_tree))

# Calculate the confusion matrix
cm_tree = confusion_matrix(y_test, Y_pred_tree)
print("Confusion Matrix:\n", cm_tree)

# Extract the values from the confusion matrix
tn, fp, fn, tp = cm_tree.ravel()

# Calculate the specificity for each class
specificity_0 = tn / (tn + fp)
specificity_1 = tp / (tp + fn)

# Print the specificity values
print(f"Specificity for class 0: {specificity_0:.2f}")
print(f"Specificity for class 1: {specificity_1:.2f}")


# Question 4. Modeling and Evaluations - Support Vector Machine (RBF) Classifier

# Train the dataset on the Support Vector Machine (RBF) Classifier
start_time = time.time()
clf_svm_rbf = SVC(kernel='rbf')
clf_svm_rbf.fit(X_train, y_train)
end_time = time.time()
print("Training time (SVM RBF):", end_time - start_time)

# Evaluate the model using the testing data
Y_pred_svm_rbf = clf_svm_rbf.predict(X_test)

# Classification report
print("Classification report (SVM RBF):\n", classification_report(y_test, Y_pred_svm_rbf))

# Calculate the confusion matrix
cm_svm_rbf = confusion_matrix(y_test, Y_pred_svm_rbf)
print("Confusion Matrix:\n", cm_svm_rbf)

# Extract the values from the confusion matrix
tn, fp, fn, tp = cm_svm_rbf.ravel()

# Calculate the specificity for each class
specificity_0 = tn / (tn + fp)
specificity_1 = tp / (tp + fn)

# Print the specificity values
print(f"Specificity for class 0: {specificity_0:.2f}")
print(f"Specificity for class 1: {specificity_1:.2f}")

# Question 6
# Find feature importance using Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
feature_importance = rf_clf.feature_importances_

# Visualize the feature importance for all columns
plt.figure(figsize=(12, 12))
plt.bar(X_train.columns, feature_importance)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# Remove the feature with the lowest importance and retrain Decision Tree
least_important_feature = X_train.columns[np.argmin(feature_importance)]
X_train_reduced = X_train.drop(least_important_feature, axis=1)
X_test_reduced = X_test.drop(least_important_feature, axis=1)

start_time = time.time()
clf_tree_reduced = DecisionTreeClassifier(random_state=42)
clf_tree_reduced.fit(X_train_reduced, y_train)
end_time = time.time()
print(f"Training time (Decision Tree without {least_important_feature}):", end_time - start_time)

Y_pred_tree_reduced = clf_tree_reduced.predict(X_test_reduced)
print("Classification report (Decision Tree without least important feature):\n", classification_report(y_test, Y_pred_tree_reduced))

plt.figure(figsize=(15, 7.5))
plot_tree(clf_tree_reduced, filled=True)
plt.show()

# Remove the four features with the lowest importances and retrain Decision Tree
four_least_important_features = X_train.columns[np.argsort(feature_importance)[:4]]
X_train_reduced_4 = X_train.drop(four_least_important_features, axis=1)
X_test_reduced_4 = X_test.drop(four_least_important_features, axis=1)

start_time = time.time()
clf_tree_reduced_4 = DecisionTreeClassifier(random_state=42)
clf_tree_reduced_4.fit(X_train_reduced_4, y_train)
end_time = time.time()
print(f"Training time (Decision Tree without {four_least_important_features}):", end_time - start_time)

Y_pred_tree_reduced_4 = clf_tree_reduced_4.predict(X_test_reduced_4)
print("Classification report (Decision Tree without four least important features):\n", classification_report(y_test, Y_pred_tree_reduced_4))

plt.figure(figsize=(15, 7.5))
plot_tree(clf_tree_reduced_4, filled=True)
plt.show()

# Remove the ten features with the lowest importances and retrain Decision Tree
ten_least_important_features = X_train.columns[np.argsort(feature_importance)[:10]]
X_train_reduced_10 = X_train.drop(ten_least_important_features, axis=1)
X_test_reduced_10 = X_test.drop(ten_least_important_features, axis=1)

start_time = time.time()
clf_tree_reduced_10 = DecisionTreeClassifier(random_state=42)
clf_tree_reduced_10.fit(X_train_reduced_10, y_train)
end_time = time.time()
print(f"Training time (Decision Tree without {ten_least_important_features}):", end_time - start_time)

Y_pred_tree_reduced_10 = clf_tree_reduced_10.predict(X_test_reduced_10)
print("Classification report (Decision Tree without ten least important features):\n", classification_report(y_test, Y_pred_tree_reduced_10))

plt.figure(figsize=(15, 7.5))
plot_tree(clf_tree_reduced_10, filled=True)
plt.show()





#-------------------------------------------------------------------------------------------------------


#Question 7

# Create a dictionary to store the performance metrics and training times of the models
model_performance = {}

# After training and evaluating the Decision Tree Classifier
dt_time = end_time - start_time
dt_accuracy = accuracy_score(y_test, Y_pred_tree)
dt_report = classification_report(y_test, Y_pred_tree)
dt_conf_matrix = confusion_matrix(y_test, Y_pred_tree)
model_performance['Decision Tree'] = {
    'Training Time': dt_time,
    'Accuracy': dt_accuracy,
    'Report': dt_report,
    'Confusion Matrix': dt_conf_matrix
}

# After training and evaluating the SVM Classifier with RBF kernel
svm_time = end_time - start_time
svm_accuracy = accuracy_score(y_test, Y_pred_svm_rbf)
svm_report = classification_report(y_test, Y_pred_svm_rbf)
svm_conf_matrix = confusion_matrix(y_test, Y_pred_svm_rbf)
model_performance['SVM'] = {
    'Training Time': svm_time,
    'Accuracy': svm_accuracy,
    'Report': svm_report,
    'Confusion Matrix': svm_conf_matrix
}

# After retraining the Decision Tree Classifier with reduced features
train_time = end_time - start_time
acc = accuracy_score(y_test, Y_pred_tree_reduced_10)
report = classification_report(y_test, Y_pred_tree_reduced_10)
conf_matrix = confusion_matrix(y_test, Y_pred_tree_reduced_10)
model_performance['Decision Tree (Reduced Features)'] = {
    'Training Time': train_time,
    'Accuracy': acc,
    'Report': report,
    'Confusion Matrix': conf_matrix
}

# Print the performance metrics and training times of the models
for model, performance in model_performance.items():
    print(f"{model} Performance:")
    print(f"Training Time: {performance['Training Time']}")
    print(f"Accuracy: {performance['Accuracy']}")
    print(f"Classification Report:\n{performance['Report']}")
    print(f"Confusion Matrix:\n{performance['Confusion Matrix']}")
    print("\n")

# Answer to Q1: The model with the highest accuracy in the model_performance dictionary performed the best.
best_model = max(model_performance, key=lambda x: model_performance[x]['Accuracy'])
print(f"The best performing model is: {best_model}")

# Answer to Q2: If the training time of 'Decision Tree (Reduced Features)' is less than the training time of 'Decision Tree', then removing least important features speeds up training times.
if model_performance['Decision Tree (Reduced Features)']['Training Time'] < model_performance['Decision Tree']['Training Time']:
    print("Removing least important features speeds up training times.")
else:
    print("Removing least important features does not speed up training times.")

# Answer to Q3: If the accuracy of 'Decision Tree (Reduced Features)' is less than the accuracy of 'Decision Tree', then removing least important features lowers the performance of the model.
if model_performance['Decision Tree (Reduced Features)']['Accuracy'] < model_performance['Decision Tree']['Accuracy']:
    print("Removing least important features lowers the performance of the model.")
else:
    print("Removing least important features does not lower the performance of the model.")

# Answer to Q4: Removing less important features can be beneficial when dealing with extremely large datasets (Big Data) as it can reduce the dimensionality of the data, speed up training times, and make the model less complex and easier to interpret, without significantly impacting the performance of the model.
print("Removing less important features can be beneficial when dealing with extremely large datasets (Big Data) as it can reduce the dimensionality of the data, speed up training times, and make the model less complex and easier to interpret, without significantly impacting the performance of the model.")