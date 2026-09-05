import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient


# 1. Charger les données
df = pd.read_csv("data/cars_synthetic.csv")


# 2. Définir X et y
X = df[
    [
        "brand",
        "model",
        "year",
        "mileage",
        "horsepower"
    ]
]

y = df["price"]


# 3. Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Preprocessing
categorical_features = ["brand", "model"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 5. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42
            )
        )
    ]
)


# 6. Hyperparamètres
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_leaf": [1, 2, 5]
}


# 7. Cross-validation + recherche
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae_test = mean_absolute_error(
    y_test,
    y_pred
)

mae_cv = -grid_search.best_score_

MAX_MAE = 4000

mlflow.set_experiment("car-price-prediction")

with mlflow.start_run():

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("mae_cv", mae_cv)
    mlflow.log_metric("mae_test", mae_test)

    if mae_test <= MAX_MAE:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name="car-price-model"
        )

        print("Quality gate PASSED")
        print("Model registered")

    else:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model"
        )

        print("Quality gate FAILED")
        print("Model NOT registered")

client = MlflowClient()

client.set_registered_model_alias(
    name="car-price-model",
    alias="champion",
    version="2"
)

# 8. Meilleur modèle
best_model = grid_search.best_estimator_


# 9. Évaluation finale
y_pred = best_model.predict(X_test)

mae_test = mean_absolute_error(
    y_test,
    y_pred
)

print("Best params :", grid_search.best_params_)
print("MAE CV      :", -grid_search.best_score_)
print("MAE Test    :", mae_test)


# 10. Sauvegarde
joblib.dump(
    best_model,
    "models/car_price_model.pkl"
)

print("Model saved.")