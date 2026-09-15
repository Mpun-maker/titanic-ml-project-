import pandas as pd


def load_training_data(
    file_path='data/train.csv'
):

    train_data = pd.read_csv(
        file_path
    )

    return train_data


def create_ticket_group_sizes(
    train_data
):

    ticket_group_sizes = (
        train_data['Ticket']
        .value_counts()
        .to_dict()
    )

    return ticket_group_sizes


def create_fare_group_bins(
    train_data
):

    train_data = train_data.copy()

    train_data['TicketGroupSize'] = (
        train_data
        .groupby('Ticket')['Ticket']
        .transform('count')
    )

    train_data['FarePerPerson'] = (
        train_data['Fare'] /
        train_data['TicketGroupSize']
    )

    fare_group_bins = [
        -float('inf'),

        train_data[
            'FarePerPerson'
        ].quantile(0.25),

        train_data[
            'FarePerPerson'
        ].quantile(0.50),

        train_data[
            'FarePerPerson'
        ].quantile(0.75),

        float('inf')
    ]

    return fare_group_bins


def assign_fare_group(
    fare_per_person,
    fare_group_bins
):

    if fare_per_person <= fare_group_bins[1]:

        return 'Low'

    elif fare_per_person <= fare_group_bins[2]:

        return 'Medium-Low'

    elif fare_per_person <= fare_group_bins[3]:

        return 'Medium-High'

    else:

        return 'High'


def create_passenger_features(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    name,
    cabin,
    ticket,
    ticket_group_sizes,
    fare_group_bins
):

    # -----------------------------------
    # Family size
    # -----------------------------------

    family_size = (
        sibsp +
        parch +
        1
    )


    # -----------------------------------
    # Is alone
    # -----------------------------------

    is_alone = int(
        family_size == 1
    )


    # -----------------------------------
    # Extract title
    # -----------------------------------

    if ',' in name and '.' in name:

        title = (
            name
            .split(',')[1]
            .split('.')[0]
            .strip()
        )

    else:

        title = 'Mr'


    # -----------------------------------
    # Normalize titles
    # -----------------------------------

    title_mapping = {
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs'
    }

    title = title_mapping.get(
        title,
        title
    )


    rare_titles = [
        'Lady',
        'Countess',
        'Capt',
        'Col',
        'Don',
        'Dr',
        'Major',
        'Rev',
        'Sir',
        'Jonkheer',
        'Dona'
    ]

    if title in rare_titles:

        title = 'Rare'


    # -----------------------------------
    # Extract deck
    # -----------------------------------

    if cabin:

        deck = cabin[0]

    else:

        deck = 'Unknown'


    # -----------------------------------
    # Ticket group size
    # -----------------------------------

    ticket_group_size = (
        ticket_group_sizes.get(
            ticket,
            1
        )
    )


    # -----------------------------------
    # Fare per person
    # -----------------------------------

    fare_per_person = (
        fare /
        ticket_group_size
    )


    # -----------------------------------
    # Fare group
    # -----------------------------------

    fare_group = assign_fare_group(
        fare_per_person,
        fare_group_bins
    )


    # -----------------------------------
    # Create dataframe
    # -----------------------------------

    passenger = pd.DataFrame({

        'Age': [age],

        'SibSp': [sibsp],

        'Parch': [parch],

        'Fare': [fare],

        'FamilySize': [family_size],

        'IsAlone': [is_alone],

        'TicketGroupSize': [
            ticket_group_size
        ],

        'FarePerPerson': [
            fare_per_person
        ],

        'Pclass': [pclass],

        'Sex': [sex],

        'Embarked': [embarked],

        'Title': [title],

        'Deck': [deck],

        'FareGroup': [fare_group]
    })


    return passenger


def validate_passenger_input(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    name,
    cabin,
    ticket
):

    errors = []


    # -----------------------------------
    # Passenger class
    # -----------------------------------

    if pclass not in [1, 2, 3]:

        errors.append(
            'Passenger class must be 1, 2, or 3.'
        )


    # -----------------------------------
    # Sex
    # -----------------------------------

    if sex not in ['male', 'female']:

        errors.append(
            'Sex must be male or female.'
        )


    # -----------------------------------
    # Age
    # -----------------------------------

    if age < 0 or age > 100:

        errors.append(
            'Age must be between 0 and 100.'
        )


    # -----------------------------------
    # Siblings / spouses
    # -----------------------------------

    if sibsp < 0 or sibsp > 10:

        errors.append(
            'Number of siblings/spouses must '
            'be between 0 and 10.'
        )


    # -----------------------------------
    # Parents / children
    # -----------------------------------

    if parch < 0 or parch > 10:

        errors.append(
            'Number of parents/children must '
            'be between 0 and 10.'
        )


    # -----------------------------------
    # Fare
    # -----------------------------------

    if fare < 0:

        errors.append(
            'Fare cannot be negative.'
        )


    # -----------------------------------
    # Embarked
    # -----------------------------------

    if embarked not in ['C', 'Q', 'S']:

        errors.append(
            'Embarkation port must be C, Q, or S.'
        )


    # -----------------------------------
    # Name
    # -----------------------------------

    if name.strip():

        if ',' not in name or '.' not in name:

            errors.append(
                'Name should follow the format '
                '"Surname, Title. FirstName".'
            )


    # -----------------------------------
    # Cabin
    # -----------------------------------

    if cabin.strip():

        if not cabin[0].isalpha():

            errors.append(
                'Cabin should start with a letter, '
                'for example C85.'
            )


    # -----------------------------------
    # Return validation result
    # -----------------------------------

    return errors