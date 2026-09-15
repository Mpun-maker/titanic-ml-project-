import streamlit as st
import joblib

from src.features import (
    create_passenger_features,
    validate_passenger_input
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title='Titanic Survival Prediction',
    page_icon='🚢',
    layout='centered'
)


# --------------------------------------------------
# Load Model and Metadata
# --------------------------------------------------

model = joblib.load(
    'models/titanic_gradient_boosting_pipeline.pkl'
)

metadata = joblib.load(
    'models/titanic_metadata.pkl'
)

ticket_group_sizes = (
    metadata['ticket_group_sizes']
)

fare_group_bins = (
    metadata['fare_group_bins']
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header('🚢 Titanic ML Project')

    st.write(
        'An end-to-end machine learning project '
        'that predicts passenger survival.'
    )

    st.divider()

    st.subheader('Model Information')

    st.write(
        '**Model:** Gradient Boosting'
    )

    st.write(
        '**Problem:** Binary Classification'
    )

    st.write(
        '**Target:** Passenger Survival'
    )

    st.write(
        '**Repeated CV ROC-AUC:** 0.896'
    )

    st.divider()

    st.caption(
        'Built with Python, Scikit-learn '
        'and Streamlit.'
    )


# --------------------------------------------------
# Main Title
# --------------------------------------------------

st.title('🚢 Titanic Survival Prediction')

st.write(
    'Enter passenger information below to '
    'estimate the probability of survival.'
)

st.info(
    'This application uses a Gradient Boosting '
    'model trained on the Kaggle Titanic dataset.'
)


# --------------------------------------------------
# Passenger Input Form
# --------------------------------------------------

with st.form('passenger_form'):

    st.subheader('👤 Passenger Information')

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            'Passenger Class',
            [1, 2, 3],
            help='1 = First Class, 2 = Second Class, 3 = Third Class'
        )

        sex = st.selectbox(
            'Sex',
            ['male', 'female']
        )

        age = st.number_input(
            'Age',
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        fare = st.number_input(
            'Fare',
            min_value=0.0,
            value=30.0,
            step=0.01
        )

    with col2:

        sibsp = st.number_input(
            'Siblings / Spouses',
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        parch = st.number_input(
            'Parents / Children',
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        embarked = st.selectbox(
            'Embarked',
            ['C', 'Q', 'S'],
            help='C = Cherbourg, Q = Queenstown, S = Southampton'
        )

    # --------------------------------------------------
    # Additional Information
    # --------------------------------------------------

    with st.expander(
        '➕ Additional Passenger Information'
    ):

        name = st.text_input(
            'Name',
            value='Smith, Mr. John',
            help='Example: Smith, Mr. John'
        )

        cabin = st.text_input(
            'Cabin',
            value='',
            help='Example: C85. Leave blank if unknown.'
        )

        ticket = st.text_input(
            'Ticket',
            value='',
            help='Enter the passenger ticket number if known.'
        )

    st.divider()

    submitted = st.form_submit_button(
        '🔮 Predict Survival',
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    errors = validate_passenger_input(
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
    )

    # --------------------------------------------------
    # Validation Errors
    # --------------------------------------------------

    if errors:

        st.error(
            'Please correct the following problems:'
        )

        for error in errors:

            st.warning(error)

    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    else:

        passenger = create_passenger_features(
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
        )

        prediction = model.predict(
            passenger
        )[0]

        probability = model.predict_proba(
            passenger
        )[0, 1]

        # --------------------------------------------------
        # Result Section
        # --------------------------------------------------

        st.divider()

        st.subheader('📊 Prediction Result')

        result_col, probability_col = st.columns(2)

        with result_col:

            if prediction == 1:

                st.success(
                    '🎉 Predicted to Survive'
                )

            else:

                st.error(
                    'Predicted Not to Survive'
                )

        with probability_col:

            st.metric(
                label='Estimated Survival Probability',
                value=f'{probability:.2%}'
            )

        st.progress(
            float(probability)
        )

        # --------------------------------------------------
        # Interpretation
        # --------------------------------------------------

        if probability >= 0.75:

            st.success(
                'The model estimates a relatively high '
                'probability of survival.'
            )

        elif probability >= 0.50:

            st.info(
                'The model estimates a moderate '
                'probability of survival.'
            )

        else:

            st.warning(
                'The model estimates a relatively low '
                'probability of survival.'
            )

        # --------------------------------------------------
        # Disclaimer
        # --------------------------------------------------

        st.caption(
            'Note: This prediction is based on historical '
            'Titanic passenger data and is intended for '
            'educational and portfolio purposes.'
        )