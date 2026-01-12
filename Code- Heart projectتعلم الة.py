#project التنبؤ بأمراض القلب
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
#  نحمل ونعرف الداتا بيس
df = pd.read_csv("heart.csv")
print(" Dataset information:")
print("Shape:", df.shape)#نطبع حجم الداتا (الصفوف والاعمد )
print("\n first 5 rows:\n", df.head())#بيطبع اول  خمس صفوف كتجربة
print("\n missing values:\n", df.isnull().sum())# اذا فيه قيم ناقصة

target_col = "target"
print("\nTarget distribution:\n", df[target_col].value_counts())# الحالات ،العمود (0,1)

# فصل الهدف عن الميزات

X = df.drop(columns=[target_col])
y = df[target_col]

# نحدد الأعمدة الرقمية والفئوية
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]#اعمدة فئوية
numeric_cols = [c for c in X.columns if c not in categorical_cols]# وهما الباقي رقمي

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

#   البيانات
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())#نطبع القيم الرقمية
])

categorical_transformer = Pipeline(steps=[# القيم الفئوية
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

#  ننشاء نموذجين داخل Pipelines
log_pipe = Pipeline(steps=[  #LogisticRegression
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

rf_pipe = Pipeline(steps=[  #Random Forest
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

#   train,testنقسمه الى
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  تدريب النموذجين train
log_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

#  نقيم test
def evaluate(model, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n {name} ")
    print(f"accuracy : {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    return f1, y_pred

# نقيم النموذجين
f1_log, y_pred_log = evaluate(log_pipe, "Logistic Regression")
f1_rf,  y_pred_rf  = evaluate(rf_pipe,  "Random Forest")

# يختار النموذج الاحسن على (حسب F1)
if f1_rf >= f1_log:
    best_name = "Random Forest"
    best_model = rf_pipe
    best_pred = y_pred_rf
else:
    best_name = "Logistic Regression"
    best_model = log_pipe
    best_pred = y_pred_log

print(f"\nbest model (by F1): {best_name}")

# Cross Validation
print("\n Fold Cross Validation:")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="f1")
print("F1 per fold:", cv_scores)
print("Mean F1:", round(cv_scores.mean(), 3))

#  Confusion Matrix
print("\n Confusion Matrix ")
cm = confusion_matrix(y_test, best_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Disease", "Disease"])
disp.plot(values_format="d")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
#يطبع اهم الميزات اذا
if best_name == "Random Forest":
    importances = best_model.named_steps["model"].feature_importances_
    features = best_model.named_steps["prep"].get_feature_names_out()
    print("\nTop Features:\n", sorted(zip(importances, features), reverse=True)[:10])
print("\nProject is successful ")