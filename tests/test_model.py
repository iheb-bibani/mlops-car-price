import joblib
import pandas as pd


def test_model_prediction():

    model = joblib.load(
        "models/car_price_model.pkl"
    )

    sample = pd.DataFrame(
        [
            {
                "brand": "Toyota",
                "model": "Yaris",
                "year": 2021,
                "mileage": 55000,
                "horsepower": 129
            }
        ]
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] > 0