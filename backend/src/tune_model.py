import joblib
import wandb

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_loader import load_data


# 🚀 Start ONE W&B run
wandb.init(project="healthcare-disease-classification")

# 📥 Load data
df = load_data()
df.columns = df.columns.str.strip().str.lower()

# 🎯 Features / Target
X = df[["age", "gender", "symptoms", "symptom_count"]]
y = df["disease"]

# ✂ Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🧹 Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "symptom_count"]),
        ("gender", OneHotEncoder(), ["gender"]),
        ("symptoms", TfidfVectorizer(), "symptoms"),
    ]
)

# 🤖 Models + Hyperparameter Space
models = {
    "LogisticRegression": (
        LogisticRegression(),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__max_iter": [200, 500],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(),
        {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1],
        },
    ),
}

best_f1 = 0
best_pipeline = None
best_model_name = None

print("\n🔍 Hyperparameter Tuning Started...\n")

# 🔁 Tune Each Model
for model_name, (model, param_grid) in models.items():
    print(f"🚀 Tuning {model_name}...")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=4,
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    # 📊 Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")

    # 🎯 ROC-AUC (multi-class)
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    probs = best_model.predict_proba(X_test)

    roc_auc = roc_auc_score(
        y_test_bin,
        probs,
        multi_class="ovr",
    )

    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)

    # 📡 Log to W&B
    wandb.log(
        {
            f"{model_name}_accuracy": acc,
            f"{model_name}_f1": f1,
            f"{model_name}_precision": precision,
            f"{model_name}_recall": recall,
            f"{model_name}_roc_auc": roc_auc,
            f"{model_name}_best_params": search.best_params_,
        }
    )

    # 🏆 Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_pipeline = best_model
        best_model_name = model_name

# 🏆 Save Best Tuned Pipeline
joblib.dump(best_pipeline, "best_tuned_model.joblib")

print(f"\n🏆 Best Tuned Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")

# 📦 Log Best Model Artifact
artifact = wandb.Artifact(
    name="best-tuned-model",
    type="model",
)

artifact.add_file("best_tuned_model.joblib")
wandb.log_artifact(artifact)

print("\n✅ Best tuned model saved & logged to W&B!")

# 🏆 Save BEST pipeline + components
joblib.dump(best_pipeline, "best_pipeline.joblib")

joblib.dump(
    best_pipeline.named_steps["preprocessor"],
    "preprocessor.joblib",
)

joblib.dump(
    best_pipeline.named_steps["model"],
    "model.joblib",
)

print("\n✅ Best pipeline, preprocessor, and model saved!")

# 📦 Create W&B Artifact
artifact = wandb.Artifact(
    name="best-model",
    type="model",
    description=f"Best tuned model: {best_model_name}",
)

artifact.add_file("best_pipeline.joblib")
artifact.add_file("preprocessor.joblib")
artifact.add_file("model.joblib")

# 📡 Log Artifact
wandb.log_artifact(artifact)

# 🔚 End W&B run
wandb.finish()

print(f"\n🏆 Best Model ({best_model_name}) Registered in W&B!")
