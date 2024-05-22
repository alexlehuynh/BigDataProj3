import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns

# Question 1: Read the CSV file
df = pd.read_csv('breast-cancer.csv')

# Question 2: Remove rows with missing values
df = df.dropna()

# Remove 'id' column as it's not a feature
df = df.drop(columns=['id'])

# Split the dataset into features (X) and target (y)
X = df.drop('diagnosis', axis=1)  # assuming 'diagnosis' is the target column
y = df['diagnosis']

# Split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate Decision Tree Classifier
def train_decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    start = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return training_time, accuracy, report, conf_matrix, clf

# Question 3: Train and evaluate Decision Tree Classifier
dt_time, dt_accuracy, dt_report, dt_conf_matrix, dt_clf = train_decision_tree(X_train, X_test, y_train, y_test)

print("Decision Tree Training time: ", dt_time)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:\n", dt_report)
print("Decision Tree Confusion Matrix:\n", dt_conf_matrix)

plt.figure(figsize=(20, 10))
plot_tree(dt_clf, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
plt.show()

# Question 4: Train and evaluate SVM Classifier with RBF kernel
clf_svm = svm.SVC(kernel='rbf')
start_svm = time.time()
clf_svm.fit(X_train, y_train)
svm_time = time.time() - start_svm
y_pred_svm = clf_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)

print("SVM Training time: ", svm_time)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_report)
print("SVM Confusion Matrix:\n", svm_conf_matrix)

# Question 6: Feature Importance using Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Plot the top two features
top_two_features = feature_importance_df.head(2)['Feature'].values
sns.scatterplot(data=df, x=top_two_features[0], y=top_two_features[1], hue='diagnosis')
plt.xlabel(top_two_features[0])
plt.ylabel(top_two_features[1])
plt.show()

# Function to retrain decision tree with reduced features
def retrain_decision_tree_with_reduced_features(X_train, X_test, y_train, y_test, drop_features):
    X_train_reduced = X_train.drop(columns=drop_features)
    X_test_reduced = X_test.drop(columns=drop_features)
    return train_decision_tree(X_train_reduced, X_test_reduced, y_train, y_test)

# Remove the feature with the lowest importance
low_importance_feature = feature_importance_df.tail(1)['Feature'].values
train_time, acc, report, conf_matrix, dtc_reduced = retrain_decision_tree_with_reduced_features(X_train, X_test, y_train, y_test, low_importance_feature)

print(f"Reduced Decision Tree Training Time (1 feature removed): {train_time}")
print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
plt.figure(figsize=(20, 10))
plot_tree(dtc_reduced, filled=True, feature_names=X_train.drop(columns=low_importance_feature).columns, class_names=['Benign', 'Malignant'])
plt.show()

# Remove the four features with the lowest importances
low_importance_features = feature_importance_df.tail(4)['Feature'].values
train_time, acc, report, conf_matrix, dtc_reduced = retrain_decision_tree_with_reduced_features(X_train, X_test, y_train, y_test, low_importance_features)

print(f"Reduced Decision Tree Training Time (4 features removed): {train_time}")
print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
plt.figure(figsize=(20, 10))
plot_tree(dtc_reduced, filled=True, feature_names=X_train.drop(columns=low_importance_features).columns, class_names=['Benign', 'Malignant'])
plt.show()

# Remove the ten features with the lowest importances
low_importance_features = feature_importance_df.tail(10)['Feature'].values
train_time, acc, report, conf_matrix, dtc_reduced = retrain_decision_tree_with_reduced_features(X_train, X_test, y_train, y_test, low_importance_features)

print(f"Reduced Decision Tree Training Time (10 features removed): {train_time}")
print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
plt.figure(figsize=(20, 10))
plot_tree(dtc_reduced, filled=True, feature_names=X_train.drop(columns=low_importance_features).columns, class_names=['Benign', 'Malignant'])
plt.show()
