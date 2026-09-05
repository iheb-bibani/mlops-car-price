import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from src.data import load_data
from src.model import train_model
from src.evaluate import (
    evaluate_model,
    check_quality_gate,
)
from src.registry import (
    register_and_promote_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "car-price-prediction"
MODEL_NAME = "car-price-model"
MODEL_PATH = "models/car_price_model.pkl"


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print()
print("Loading data...")

X, y = load_data()

print(
    f"Dataset loaded: {len(X)} rows"
)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print()
print("Training model...")

best_model, best_params, mae_cv = train_model(
    X_train,
    y_train,
)

print()
print(
    "Best params:",
    best_params,
)


# ============================================================
# 4. EVALUATION
# ============================================================

metrics = evaluate_model(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    mae_cv=mae_cv,
)


print()
print("MODEL EVALUATION")
print("----------------")

print(
    "MAE CV      :",
    metrics["mae_cv"],
)

print(
    "MAE Test    :",
    metrics["mae_test"],
)

print(
    "Latency     :",
    metrics["latency_ms"],
    "ms",
)


# ============================================================
# 5. QUALITY GATE
# ============================================================

quality_gate_passed = check_quality_gate(
    metrics
)


# ============================================================
# 6. MLFLOW RUN
# ============================================================

with mlflow.start_run():

    # --------------------------------------------------------
    # Log hyperparameters
    # --------------------------------------------------------

    mlflow.log_params(
        best_params
    )


    # --------------------------------------------------------
    # Log metrics
    # --------------------------------------------------------

    mlflow.log_metrics(
        metrics
    )


    # --------------------------------------------------------
    # Quality Gate + Registry
    # --------------------------------------------------------

    if quality_gate_passed:

        print()
        print(
            "Quality gate PASSED"
        )

        print(
            "Registering challenger..."
        )

        register_and_promote_model(
            model=best_model,
            metrics=metrics,
            model_name=MODEL_NAME,
        )

    else:

        print()
        print(
            "Quality gate FAILED"
        )

        print(
            "Model NOT registered"
        )

        # On garde quand même le modèle
        # comme artifact MLflow.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
        )


# ============================================================
# 7. LOCAL MODEL SAVE
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH,
)


print()
print(
    f"Model saved locally: {MODEL_PATH}"
)


# ============================================================
# END
# ============================================================

print()
print(
    "Training pipeline finished."
)