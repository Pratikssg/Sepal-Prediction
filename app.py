import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
# We trained this model in the model_builder.py script
model = joblib.load('iris_model.joblib')

# Set the title of the Streamlit app
st.title('Iris Flower Species Prediction')

# Add some text to the app
st.write('This app predicts the species of an Iris flower based on its measurements.')

# Create sliders in the sidebar for user input
st.sidebar.header('Input Features')


def user_input_features():
    """
    Creates sliders for user to input flower measurements.
    Returns a pandas DataFrame with the input.
    """
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
df_input = user_input_features()

# Display the user input
st.subheader('User Input')
st.write(df_input)

# Make a prediction when the button is clicked
if st.button('Predict'):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader('Prediction')
    st.write(f'The predicted species is: **{prediction[0]}**')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)