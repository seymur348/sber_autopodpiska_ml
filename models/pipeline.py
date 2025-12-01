
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# =========================================
# 1. Load prepared dataset
# =========================================
df = pd.read_pickle("../data/sessions_prepared.pkl")

y = df["target"]
X = df.drop(columns=["target"])


# =========================================
# 2. Feature groups
# =========================================
onehot_cols = [
    "device_category",
    "utm_medium",
    "geo_country"
]

freq_cols = [
    "utm_source",
    "utm_campaign",
    "utm_adcontent",
    "device_browser",
    "geo_city"
]

numeric_cols = [
    "visit_hour",
    "visit_weekday",
    "visit_month",
    "is_weekend",
    "hits",
    "pageviews",
    "events",
    "screen_w",
    "screen_h"
]


# =========================================
# 3. Frequency encoding (fit on FULL dataset)
# =========================================
freq_maps = {}

for col in freq_cols:
    mapping = X[col].value_counts(normalize=True)
    freq_maps[col] = mapping
    X[col] = X[col].map(mapping)


# =========================================
# 4. Train/Test split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# onehot → only str
for col in onehot_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)


# =========================================
# 5. ColumnTransformer
# =========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
        ("num", "passthrough", numeric_cols + freq_cols)
    ],
    remainder="drop"
)


# =========================================
# 6. Fast Logistic Regression
# =========================================
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        solver="liblinear",
        max_iter=1200,
        class_weight="balanced"
    ))
])


# =========================================
# 7. Train
# =========================================
model.fit(X_train, y_train)
print("Модель обучена!")


# =========================================
# 8. Metrics
# =========================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# =========================================
# 9. Save model and freq maps
# =========================================
with open("../models/model.pkl", "wb") as f:
    dill.dump(model, f)

with open("../models/freq_maps.pkl", "wb") as f:
    dill.dump(freq_maps, f)

print("model.pkl и freq_maps.pkl сохранены!")


#