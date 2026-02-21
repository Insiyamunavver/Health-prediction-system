import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_loader import load_data

# 📥 Load data
df = load_data()
df.columns = df.columns.str.strip().str.lower()

# 🎯 Features / Target
X = df[["age", "gender", "symptoms", "symptom_count"]]
y = df["disease"]

# ✂ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🧹 Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "symptom_count"]),
        (
            "gender",
            OneHotEncoder(handle_unknown="ignore"),  # ✅ FIX ADDED
            ["gender"]
        ),
        ("symptoms", TfidfVectorizer(), "symptoms"),
    ]
)

# 🤖 Models to Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

best_model_name = None
best_f1 = 0
best_pipeline = None

print("\n🚀 Training Multiple Models...\n")

# 🔁 Train & Evaluate Each Model
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"📊 {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = pipeline

# 🏆 Save BEST pipeline (✅ FIXED BUG)
joblib.dump(best_pipeline, "model.joblib")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")
print("✅ Best pipeline saved as model.joblib")
