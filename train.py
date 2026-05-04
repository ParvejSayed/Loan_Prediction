import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

os.makedirs("artifacts", exist_ok=True)

train = pd.read_csv("Train.csv")
data = train.drop(columns=["Loan_ID"])

X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

trf1 = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler()),
])

trf3 = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("scaler", MinMaxScaler()),
])

trf4 = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["0", "1", "2", "3+"], ["Rural", "Semiurban", "Urban"], ["Not Graduate", "Graduate"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

trf5 = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")),
])

final_preprocessor = ColumnTransformer(transformers=[
    ("num_scale", trf1, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]),
    ("knn",       trf3, ["Credit_History"]),
    ("ordinal",   trf4, ["Dependents", "Property_Area", "Education"]),
    ("onehot",    trf5, ["Gender", "Married", "Self_Employed"]),
])

X_train_t = final_preprocessor.fit_transform(X_train)
X_test_t = final_preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)

# Best hyperparameters from notebook GridSearchCV
model = GradientBoostingClassifier(learning_rate=0.2, max_depth=4, n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

with open("artifacts/final_preprocessor.pkl", "wb") as f:
    pickle.dump(final_preprocessor, f)
print("Saved: artifacts/final_preprocessor.pkl")

with open("artifacts/gradient_boosting_best_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: artifacts/gradient_boosting_best_model.pkl")

y_pred = model.predict(X_test_t)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
