import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import pickle

# Load trained model & scaler
with open('regressor.pkl', 'wb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'wb') as file:
    scaler = pickle.load(file)

with open('xgboost_model.pkl', 'wb') as file:
    xgboost_model = pickle.load(file)

# HTML Design
html_temp = """
<div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Song Popularity Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Predict the popularity of a song using audio features</h4>
</div>
"""

desc_temp = """ 
### About This App
This app uses machine learning to predict the **popularity score** of a song (0-100) based on its audio features.

#### Data Source
Kaggle: [Song Popularity Dataset](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset)
"""

def main():
    stc.html(html_temp)
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediction":
        run_prediction_app()

def run_prediction_app():
    st.subheader("Fill in Song Features Below")

    col1, col2 = st.columns(2)
    danceability = col1.slider('Danceability', 0.0, 1.0, 0.5)
    energy = col2.slider('Energy', 0.0, 1.0, 0.5)
    loudness = col1.slider('Loudness (dB)', -60.0, 0.0, -10.0)
    speechiness = col2.slider('Speechiness', 0.0, 1.0, 0.1)
    acousticness = col1.slider('Acousticness', 0.0, 1.0, 0.3)
    instrumentalness = col2.slider('Instrumentalness', 0.0, 1.0, 0.0)
    liveness = col1.slider('Liveness', 0.0, 1.0, 0.1)
    valence = col2.slider('Valence', 0.0, 1.0, 0.5)
    tempo = st.slider('Tempo (BPM)', 40.0, 250.0, 120.0)

    button = st.button("Predict Popularity")

    if button:
        result = predict_popularity([danceability, energy, loudness, speechiness,
                                     acousticness, instrumentalness, liveness,
                                     valence, tempo])
        st.success(f"Predicted Song Popularity: **{result:.2f} / 100**")

def predict_popularity(features):
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)
    log_prediction = model.predict(input_scaled)[0]
    prediction = np.exp(log_prediction)  # Undo log transform
    return prediction

if __name__ == "__main__":
    main()