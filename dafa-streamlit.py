import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv('drug200.csv')

st.title('Klasifikasi Jenis Obat')
st.caption('Daffa Fauzi | 211351038')
st.markdown("""---""")

#visualisasi
for col in ["Sex","BP", "Cholesterol"]:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop(['Drug'], axis=1)
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

enc = OrdinalEncoder()
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

st.header("Visualisasi Model")
if st.button('Tampilkan Gambar'):
    plt.figure(figsize=(12,8))
    tree.plot_tree(clf_gini.fit(X_train, y_train))
    st.pyplot()
    plt.show()
    st.button("Hide", type="primary")

#input klasifikasi
st.header("Input Niai Untuk Klasifikasi")
Age = st.number_input('Masukkan Umur :')
Sex = st.number_input('Masukkan Jenis Kelamin:')
BP = st.number_input('Masukkan nilai BP :')
Cholesterol = st.number_input('Masukkan Nilai Cholesterol :')
Na_to_K = st.number_input('Input Nilai Na_to_K :')


if st.button('Klasifikasikan Obat') :
    sue = clf_gini.predict([[Age,Sex,BP,Cholesterol,Na_to_K]])

    st.success(f"Hasil Klasifikasi Obat Adalah: {sue}")
