```bash
Laporan Proyek Machine Learning
```
```bash
Nama: Daffa Fauzi
```
```bash
Nim:211351038
```
```bash
Kelas: Pagi A
```
```bash
Domain Proyek

Dataset yang Anda berikan merupakan dataset yang berisi informasi tentang berbagai jenis obat (A, B, C, X, Y) dengan beberapa fitur terkait. Proyek pada dataset ini dapat melibatkan berbagai tugas, tergantung pada tujuan analisis atau model yang ingin dikembangkan
```
```bash
Business Understanding

Pahami deskripsi dataset dan variabel-variabelnya yang diberikan pada halaman dataset Kaggle. Identifikasi kolom-kolom yang ada, tipe data, dan deskripsi masing-masing variabel.
```
```bash
Problem Statements

1. Prediksi Jenis Obat:
2. Analisis Sentimen terhadap Obat
3. Identifikasi Faktor Pengaruh Preskripsi Obat
```
```bash 
Goals

Tujuan dari dataset ini, drugs-a-b-c-x-y-for-decision-trees, dapat bervariasi tergantung pada kebutuhan analisis dan tujuan bisnis spesifik. Beberapa tujuan umum yang dapat dicapai melalui analisis datase
```
```bash
Variabel-variabel pada Dataset Groceries adalah sebagai berikut

1.Age (Umur): Variabel ini mungkin menyimpan informasi tentang usia pasien.
2.Sex (Jenis Kelamin): Variabel yang mungkin menyimpan informasi tentang jenis kelamin pasien, seperti 'Male' atau 'Female'.
3.BP (Tekanan Darah): Variabel yang mungkin menyimpan informasi tentang tekanan darah pasien, seperti 'HIGH', 'NORMAL', atau 'LOW'.
4.Cholesterol (Kolesterol): Variabel yang mungkin menyimpan informasi tentang tingkat kolesterol pasien, seperti 'HIGH' atau 'NORMAL'.
5.Na_to_K (Rasio Natrium ke Kalium): Variabel yang mungkin menyimpan informasi tentang rasio natrium ke kalium dalam tubuh pasien.
6.Drug (Obat): Variabel target yang mungkin menyimpan informasi tentang jenis obat yang diresepkan kepada pasien (A, B, C, X, Y)
```
```bash
Data Understanding

Untuk mendapatkan pemahaman yang baik tentang dataset drugs-a-b-c-x-y-for-decision-trees, kita perlu melakukan serangkaian langkah untuk menganalisis struktur dan karakteristik datanya (https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)
```
```bash
Data Collection

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset (drugs-a-b-c-x-y-for-decision-trees), jika anda tertarik dengan datasetnya, anda bisa click link diatas.
```
```bash
Data Discovery And Profiling

Untuk bagian ini, kita akan menggunakan teknik EDA
```
```bash
Import Modul yang di butuhkan

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import tree
import pickle
```

```bash
Karena kita menggunakan google colab untuk mengerjakannya maka kita akan import files juga

from google.colab import files
files.upload()
```
```bash
Setelah mengupload filenya, maka kita akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

```bash
Done, lalu mari kita download datasetsnya

!kaggle datasets download -d pablomgomez21/drugs-a-b-c-x-y-for-decision-trees
```
```bash
Selanjutnya kita harus extract file yang tadi telah didownload

!unzip drugs-a-b-c-x-y-for-decision-trees.zip -d dises
!ls dises
```
```bash
Memasukkan file csv yang telah diextract pada sebuah variable

data = pd.read_csv("/content/dises/drug200.csv")
data.head()
```
```bash
data.shape
```
```bash
Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info,

data.info()
```
```bash
data.columns
```
```bash
Untuk melihat mengenai Nilai count, Nilai mean, Nilai std, Nilai min, berapa 25% dari data, berapa 50% dari data, berapa 75% dari data, Nilai max

data.describe()
```
```bash
Eda
```
```bash
Untuk melihat gambar korelasi columns

X_train.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(12,8))
plt.show()
```
```bash

Untuk melihat gambar colums 

scatter_matrix(X_train, figsize=(12,8))
plt.show()
```
```bash
Untuk Melihat Age,Sex,BP,Colesterol

X_train.hist(figsize=(15,10))
plt.show()
```
```bash
Untuk melihat Plasma

sns.heatmap(X_train.isnull(), yticklabels = False, cmap = "plasma")
plt.show()
```
```bash
Untuk melihat gambar apakah ada data yang null

sns.heatmap(X_train.isnull(), yticklabels = False, cmap = "plasma")
plt.show()
```
```bash
Data Propretion
```
```bash
Menghapus columns Drug

X = data.drop(['Drug'], axis=1)
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
```
```bash
Mengecek train.shape dan test.shape

X_train.shape, X_test.shape
```
```bash
Mengecek x.train dtypes

X_train.dtypes
```
```bash
Membuat kolom

encoder = ce.OrdinalEncoder(cols=['Sex', 'BP', 'Cholesterol'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```
```bash
Membuat kolom

encoder = ce.OrdinalEncoder(cols=['Sex', 'BP', 'Cholesterol'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```
```bash
Menampilan colums X_train.head

X_train.head()
```
```bash
Menampilkan colums X_test.head

X_test.head()
```
```bash
Modeling
```
```bash
mari kita import library yang nanti akan digunakan

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
```
```bash
Hasilnya dibawah ini

Model accuracy score with criterion gini index: 0.8485
Training-set accuracy score: 0.8358
```
```bash
Membuat model DecisionTreeClassifier dan memasukkan modul dari sklearn(memasukkan library) dan kita bisa melihat score dari model kita

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```
```bash
Tahap berikutnya adalah kita coba model dengan inputan berikut

if accuracy_score(y_test, y_pred_gini) > accuracy_score(y_train, y_pred_train_gini):
    print("Gini Index Criterion is Better and it has accuracy equal to ", accuracy_score(y_test, y_pred_gini) * 100)
else:
    print("Entropy Criterion is Better and it has accuracy equal to ", accuracy_score(y_train, y_pred_train_gini) * 100)
```
```bash
kita tampilkan visualisasi hasil prediksi model DecisionTree

plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.show()
```
```bash
Jika sudah berhasil jangan lupa untuk simpan model menggunakan pickle

filename = 'drug.sav'
pickle.dump(model.predict,open(filename,'wb'))
```
```bash
Evaluation

Decision Tree Classifier adalah model yang digunakan dalam proyek untuk memprediksi terkena atau tidak terkena penyakit paru-paru. Decision Tree merupakan model yang dapat memetakan keputusan berdasarkan serangkaian aturan dan pemilihan fitur.

Evaluasi model dilakukan dengan beberapa metrik, seperti akurasi, confusion matrix, dan classification report

Classification report memberikan informasi terperinci tentang performa model pada setiap kelas, termasuk precision, recall, dan f1-score.

Visualisasi model Decision Tree menggunakan plot_tree memberikan representasi grafis dari struktur pohon keputusan. Setiap node pada pohon mewakili suatu keputusan atau prediksi berdasarkan nilai fitur tertentu.
```
```bash
Penyebaran
```
```bash
Penyebaran

https://dtreeuas.streamlit.app/
```