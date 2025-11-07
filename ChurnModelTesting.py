# Install dependencies if running on a fresh environment
# !pip install pandas numpy scikit-learn matplotlib seaborn shap joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import shap
import joblib
# You can download directly from IBM’s open dataset URL:
df = pd.read_csv("/Users/tachouyou/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1/WA_Fn-UseC_-Telco-Customer-Churn.csv")


print(df.shape)
df.head()

# Replace empty strings and convert "TotalCharges" to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
df = df.dropna(subset=['TotalCharges'])

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID (not useful)
df = df.drop(columns=['customerID'])

# Convert categorical variables to dummy/one-hot
cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

joblib.dump(model, "telco_churn_model.pkl")
joblib.dump(scaler, "telco_scaler.pkl")
print("✅ Model and scaler saved.")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("F1:", round(f1_score(y_test, y_pred), 3))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(9,6))
sns.barplot(x=importances[:10], y=importances.index[:10])
plt.title("Top 10 Feature Importances")
plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ✅ Use X_test here, not X
shap.summary_plot(shap_values[1], X_test, max_display=10)
