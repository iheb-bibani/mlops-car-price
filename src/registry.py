import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def register_and_promote_model(
    model,
    metrics,
    model_name="car-price-model",
):
    client = MlflowClient()

    # ========================================================
    # 1. Enregistrer le challenger
    # ========================================================

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=model_name,
    )

    challenger_version = (
        model_info.registered_model_version
    )

    print(
        f"Challenger registered as V{challenger_version}"
    )

    client.set_model_version_tag(
        name=model_name,
        version=challenger_version,
        key="role",
        value="challenger",
    )

    # ========================================================
    # 2. Chercher le champion actuel
    # ========================================================

    try:
        champion = client.get_model_version_by_alias(
            name=model_name,
            alias="champion",
        )

    except MlflowException:
        print(
            "No champion exists yet."
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=challenger_version,
        )

        client.set_model_version_tag(
            name=model_name,
            version=challenger_version,
            key="role",
            value="champion",
        )

        print(
            f"V{challenger_version} becomes the first champion."
        )

        return challenger_version

    # ========================================================
    # 3. Lire les métriques du champion
    # ========================================================

    champion_run = client.get_run(
        champion.run_id
    )

    champion_mae_cv = (
        champion_run
        .data
        .metrics
        .get("mae_cv")
    )

    challenger_mae_cv = metrics["mae_cv"]

    print()
    print("CHAMPION VS CHALLENGER")
    print("-----------------------")

    print(
        f"Champion V{champion.version}"
        f" MAE CV: {champion_mae_cv}"
    )

    print(
        f"Challenger V{challenger_version}"
        f" MAE CV: {challenger_mae_cv}"
    )

    # ========================================================
    # 4. Comparaison
    # ========================================================

    if challenger_mae_cv < champion_mae_cv:

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=challenger_version,
        )

        client.set_model_version_tag(
            name=model_name,
            version=champion.version,
            key="role",
            value="previous_champion",
        )

        client.set_model_version_tag(
            name=model_name,
            version=challenger_version,
            key="role",
            value="champion",
        )

        print()
        print(
            f"V{challenger_version} becomes the new champion."
        )

    else:

        print()
        print(
            f"Champion remains V{champion.version}."
        )

    return challenger_version