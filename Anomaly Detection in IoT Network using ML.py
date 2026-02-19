import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

file_path = r"D:\RT_IOT2022"   
if not os.path.exists(file_path):
    print("File not found! Check path:", file_path)
    exit()

data = pd.read_csv(file_path)
print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

print("Dataset Shape:", data.shape)

data.dropna(inplace=True)

print("\nChecking column types\n")
print(data.dtypes.value_counts())
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object', 'string']).columns
print("\nCategorical Columns:", list(categorical_cols))

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

X = X.apply(pd.to_numeric)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Preprocessing Complete")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train-Test Split done")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}
results = []

best_f1 = 0
best_model_obj = None
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if f1 > best_f1:
        best_f1 = f1
        best_model_obj = model
        best_model_name = name
        best_y_pred = y_pred

    results.append([name, acc, prec, rec, f1, end - start])

results_df = pd.DataFrame(results, columns=[
    "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)"
])

print("\n Model Comparison")
print(results_df)

script_dir = os.path.dirname(os.path.abspath(__file__))

results_folder = os.path.join(script_dir, "results")
os.makedirs(results_folder, exist_ok=True)

save_path = os.path.join(os.getcwd(), "model_comparison_results.csv")
results_df.to_csv(save_path, index=False)

print("Results saved successfully at:", save_path)
sns.set_style("whitegrid")
plt.figure()
sns.barplot(x="Algorithm", y="Accuracy", data=results_df)
plt.title("Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure()
sns.barplot(x="Algorithm", y="F1 Score", data=results_df)
plt.title("F1 Score Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure()
sns.barplot(x="Algorithm", y="Training Time (s)", data=results_df)
plt.title("Training Time Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

best_model = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
print("\nBest Model Based on F1 Score:")
print(best_model)

print(f"\nGenerating Confusion Matrix for {best_model_name}")

cm = confusion_matrix(y_test, best_y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

