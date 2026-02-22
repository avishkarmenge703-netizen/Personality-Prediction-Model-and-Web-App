import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Load the dataset (assuming the file is in the same directory)
# ------------------------------------------------------------
data = pd.read_csv('personality_synthetic_dataset (1).csv')

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------
X = data.drop('personality_type', axis=1)
y = data['personality_type']

# ------------------------------------------------------------
# 3. Encode the target labels (Extrovert, Introvert, Ambivert)
# ------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ------------------------------------------------------------
# 4. Train / test split (80% train, 20% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------------------------------------------
# 5. Feature scaling (optional, but can help some models)
#    Tree‑based models don't require scaling, but we keep it for completeness.
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6. Baseline Random Forest model
# ------------------------------------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
print("Baseline Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------------------------------------
# 7. Hyperparameter tuning for enhancement
# ------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned, target_names=label_encoder.classes_))

# ------------------------------------------------------------
# 8. Save the best model, scaler, and label encoder for later use
# ------------------------------------------------------------
joblib.dump(best_model, 'personality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nModel, scaler, and label encoder saved successfully.")
