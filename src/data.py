import pandas as pd


FEATURES = [
    "brand",
    "model",
    "year",
    "mileage",
    "horsepower",
]

TARGET = "price"


def load_data(
    filepath="data/cars_synthetic.csv"
):
    df = pd.read_csv(filepath)

    X = df[FEATURES]
    y = df[TARGET]

    return X, y