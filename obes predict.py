import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from streamlit_option_menu import option_menu

# Dummy users (username: password)
USERS = {
    "admin": "admin123",
    "user1": "pass1",
    "user2": "pass2",
}

# Load model & scaler
rf_model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Page config
st.set_page_config(layout="centered")

def login():
    st.title("🔐 Login ke Aplikasi")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in USERS and USERS[username] == password:
                st.session_state.user = username
                st.success(f"Selamat datang, {username}!")
                st.rerun()
            else:
                st.error("Username atau password salah.")

# Require login
if "user" not in st.session_state:
    login()
    st.stop()

# Sidebar
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.user}**")
    if st.button("Logout"):
        del st.session_state.user
        st.rerun()

# Menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Predict", "Riwayat"],
    icons=["house", "person", "clock-history"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "black"},
        "nav-link": {"font-size": "16px", "text-align": "center"},
        "nav-link-selected": {"background-color": "#ff4848", "color": "white"},
    },
)

def get_recommendation(category):
    recommendations = {
        'Underweight': """- Porsi makan tinggi kalori sehat\n- Makan 5-6x sehari\n- Latihan kekuatan\n- Konsultasi medis""",
        'Normal': """- Pola makan seimbang\n- Olahraga rutin\n- Hindari gula/lemak jenuh""",
        'Overweight': """- Porsi rendah kalori tinggi nutrisi\n- Jalan kaki tiap hari\n- Konsultasi gizi""",
        'Obesity': """- Makan kecil sering\n- Mulai aktivitas ringan\n- Konsultasi medis terarah"""
    }
    return recommendations.get(category, "Tidak ada rekomendasi.")

def save_prediction(username, input_data, pred_class):
    log = input_data.copy()
    log["User"] = username
    log["Predicted_Class"] = pred_class
    log["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        old = pd.read_csv("history.csv")
        new = pd.concat([old, log], ignore_index=True)
    except FileNotFoundError:
        new = log
    new.to_csv("history.csv", index=False)

if selected == "Home":
    st.title("Obesity Class Predictor")
    st.markdown("""
        Aplikasi ini memprediksi status obesitas seseorang berdasarkan input gaya hidup.
        Gunakan menu Predict untuk memasukkan data Anda.
    """)

elif selected == "Predict":
    st.title("User Input for Obesity Prediction")

    if 'step' not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        sex = st.radio("Sex", ['Male', 'Female'])
        age = st.slider("Age", 10, 80, 25)
        height = st.slider("Height (cm)", 100, 220, 170)

        if st.button("Next"):
            st.session_state.sex = sex
            st.session_state.age = age
            st.session_state.height = height
            st.session_state.step = 2

    elif st.session_state.step == 2:
        overweight_families = st.radio("Overweight/Obese Families", ['Yes', 'No'])
        fast_food = st.radio("Fast Food Consumption", ['Yes', 'No'])
        vegetables_frequency = st.selectbox("Vegetable Frequency", ['Rarely', 'Sometimes', 'Always'])
        main_meals = st.selectbox("Main Meals Daily", ['1-2', '3', '3+'])
        food_intake = st.selectbox("Food Intake Between Meals", ['Rarely', 'Sometimes', 'Usually', 'Always'])
        smoking = st.radio("Smoking", ['Yes', 'No'])
        liquid_intake = st.selectbox("Liquid Intake", ['<1L', '1-2L', '>2L'])
        calorie_count = st.radio("Count Calories?", ['Yes', 'No'])
        exercise = st.selectbox("Exercise Days", ['None', '1-2', '3-4', '5-6', '6+'])
        tech_hours = st.selectbox("Tech Time", ['0-2h', '3-5h', '>5h'])
        transport = st.selectbox("Transportation", ['Automobile', 'Motorbike', 'Bike', 'Public transport', 'Walking'])

        if st.button("Predict"):
            with st.spinner("Memproses prediksi..."):
                sex = st.session_state.sex
                age = st.session_state.age
                height = st.session_state.height

                input_data = pd.DataFrame({
                    'Sex': [1 if sex == 'Male' else 2],
                    'Age': [age],
                    'Height': [height],
                    'Overweight_Obese_Family': [1 if overweight_families == 'Yes' else 2],
                    'Consumption_of_Fast_Food': [1 if fast_food == 'Yes' else 2],
                    'Frequency_of_Consuming_Vegetables': [1.0 if vegetables_frequency == 'Rarely' else 2.0 if vegetables_frequency == 'Sometimes' else 3.0],
                    'Number_of_Main_Meals_Daily': [1.0 if main_meals == '1-2' else 2.0 if main_meals == '3' else 3.0],
                    'Food_Intake_Between_Meals': [1 if food_intake == 'Rarely' else 2 if food_intake == 'Sometimes' else 3 if food_intake == 'Usually' else 4],
                    'Smoking': [1 if smoking == 'Yes' else 2],
                    'Liquid_Intake_Daily': [1.0 if liquid_intake == '<1L' else 2.0 if liquid_intake == '1-2L' else 3.0],
                    'Calculation_of_Calorie_Intake': [1 if calorie_count == 'Yes' else 2],
                    'Physical_Excercise': [0.0 if exercise == 'None' else 1.0 if exercise == '1-2' else 2.0 if exercise == '3-4' else 3.0 if exercise == '5-6' else 3.5],
                    'Schedule_Dedicated_to_Technology': [0.0 if tech_hours == '0-2h' else 1.0 if tech_hours == '3-5h' else 2.0],
                    'Type_of_Transportation_Used': [1 if transport == 'Automobile' else 2 if transport == 'Motorbike' else 3 if transport == 'Bike' else 4 if transport == 'Public transport' else 5],
                })

                input_scaled = scaler.transform(input_data)
                pred = rf_model.predict(input_scaled)
                pred_class = {1: 'Underweight', 2: 'Normal', 3: 'Overweight', 4: 'Obesity'}[pred[0]]

                st.success(f"Predicted Class: **{pred_class}**")
                st.subheader("Rekomendasi")
                st.markdown(get_recommendation(pred_class))

                save_prediction(st.session_state.user, input_data, pred_class)

            st.button("Ulangi", on_click=lambda: st.session_state.update(step=1))

elif selected == "Riwayat":
    st.title("📋 Riwayat Prediksi Anda")

    try:
        df = pd.read_csv("history.csv")
        user_data = df[df["User"] == st.session_state.user]

        if user_data.empty:
            st.info("Belum ada riwayat prediksi.")
        else:
            # Reorder columns: Timestamp, User, Predicted_Class, then the rest
            cols = user_data.columns.tolist()
            reordered = ["Timestamp", "User", "Predicted_Class"] + [col for col in cols if col not in ["Timestamp", "User", "Predicted_Class"]]
            st.dataframe(user_data[reordered].sort_values("Timestamp", ascending=False))
    except FileNotFoundError:
        st.warning("Belum ada data riwayat disimpan.")
