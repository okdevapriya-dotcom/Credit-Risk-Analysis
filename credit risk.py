import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\sunil\OneDrive\Desktop\Credit_Risk_Project\credit_data\customer_data.csv")
df["fea_2"] = df["fea_2"].fillna(df["fea_2"].mean())
plt.figure()
sns.countplot(x="label", data=df)
plt.title("Default vs Non-Default Count")
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
X = df.drop(["label", "id"], axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=3000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
coefficients = pd.DataFrame(
    model.coef_[0],
    index=X.columns,
    columns=["Coefficient"]
).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(8,6))
coefficients.plot(kind="bar")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
