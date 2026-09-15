from src.features import (
    create_passenger_features
)


def test_create_passenger_features():

    ticket_group_sizes = {
        'TEST123': 2
    }

    fare_group_bins = [
        -float('inf'),
        10,
        20,
        50,
        float('inf')
    ]

    passenger = create_passenger_features(
        pclass=1,
        sex='female',
        age=30,
        sibsp=1,
        parch=1,
        fare=40,
        embarked='S',
        name='Smith, Mrs. Jane',
        cabin='C85',
        ticket='TEST123',
        ticket_group_sizes=ticket_group_sizes,
        fare_group_bins=fare_group_bins
    )

    assert passenger.shape == (1, 14)

    assert passenger['FamilySize'].iloc[0] == 3

    assert passenger['IsAlone'].iloc[0] == 0

    assert passenger['Title'].iloc[0] == 'Mrs'

    assert passenger['Deck'].iloc[0] == 'C'

    assert passenger['TicketGroupSize'].iloc[0] == 2

    assert passenger['FarePerPerson'].iloc[0] == 20





from src.features import (
    validate_passenger_input
)


def test_valid_passenger_input():

    errors = validate_passenger_input(
        pclass=1,
        sex='female',
        age=30,
        sibsp=0,
        parch=0,
        fare=50,
        embarked='S',
        name='Smith, Mrs. Jane',
        cabin='C85',
        ticket='TEST123'
    )

    assert errors == []


# test an invalid passenger

def test_invalid_passenger_input():

    errors = validate_passenger_input(
        pclass=5,
        sex='unknown',
        age=-10,
        sibsp=-1,
        parch=0,
        fare=-50,
        embarked='X',
        name='John Smith',
        cabin='123',
        ticket='TEST123'
    )

    assert len(errors) > 0




