import joblib


def test_model_can_be_loaded():

    model = joblib.load(
        'models/titanic_gradient_boosting_pipeline.pkl'
    )

    assert model is not None




# test the metadata file

def test_metadata_can_be_loaded():

    metadata = joblib.load(
        'models/titanic_metadata.pkl'
    )

    assert 'ticket_group_sizes' in metadata

    assert 'fare_group_bins' in metadata




# test prediction consistency

import numpy as np
import joblib

from src.features import (
    create_passenger_features
)


def test_loaded_model_prediction():

    model = joblib.load(
        'models/titanic_gradient_boosting_pipeline.pkl'
    )

    metadata = joblib.load(
        'models/titanic_metadata.pkl'
    )

    passenger = create_passenger_features(
        pclass=3,
        sex='male',
        age=25,
        sibsp=0,
        parch=0,
        fare=7.25,
        embarked='S',
        name='Smith, Mr. John',
        cabin='',
        ticket='UNKNOWN',
        ticket_group_sizes=metadata['ticket_group_sizes'],
        fare_group_bins=metadata['fare_group_bins']
    )

    prediction = model.predict(
        passenger
    )

    probability = model.predict_proba(
        passenger
    )

    assert prediction.shape == (1,)

    assert probability.shape == (1, 2)

    assert 0 <= probability[0, 1] <= 1

    assert prediction[0] in [0, 1]


