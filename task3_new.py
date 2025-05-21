import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "D:\\internship_work_for_own\\Prodigy\\bank.csv"  # update this if the file is in a different folder
data = pd.read_csv(file_path, sep=';')  # delimiter is semicolon in this dataset

print("First 5 rows of dataset:")
print(data.head())

label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    if column != 'y':  # Target will be encoded separately
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

target_encoder = LabelEncoder()
data['y'] = target_encoder.fit_transform(data['y'])  # 'yes' becomes 1, 'no' becomes 0

X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=target_encoder.classes_, filled=True)
plt.title("Decision Tree for Bank Marketing Data")
plt.show()
