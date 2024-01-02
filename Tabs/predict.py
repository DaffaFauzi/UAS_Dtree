import streamlit as st

from web_functions import predict
from sklearn.tree import export_graphviz


def app(df, x, y):

    st.title("Halaman Prediksi")


col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("input nilai Age")
with col1:
    Sex = st.number_input("input nilai Sex")
with col2:
    BP = st.text_input("input nilai BP")
with col2:
    Cholesterol = st.text_input("input nilai Cholesterol")
with col2:
    Na_to_K = st.text_input("input nilai Na_to_K")

features = [Age, Sex, BP, Cholesterol, Na_to_K]

# tombol prediksi
if st.button("Prediksi"):
    prediction, score = predict(x, y, features)
    score = score
    st.info("Prediksi Sukses...")

    if (prediction == 1):
        st.warning("Accuracy Sebelumnya")
    else:
        st.success("Accuracy Sukses")

    st.write("Model yang digunakan memiliki tingkat akurasi",
             (score*100), "%")
