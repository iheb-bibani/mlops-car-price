from sklearn.model_selection import GridSearchCV

from src.preprocessing import build_pipeline


PARAM_GRID = {
    "model__n_estimators": [
        100,
        200,
    ],
    "model__max_depth": [
        None,
        8,
        12,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        5,
    ],
}


def train_model(
    X_train,
    y_train,
):
    pipeline = build_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_model = (
        grid_search.best_estimator_
    )

    best_params = (
        grid_search.best_params_
    )

    mae_cv = (
        -grid_search.best_score_
    )

    return (
        best_model,
        best_params,
        mae_cv,
    )