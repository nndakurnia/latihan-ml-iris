import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Mengonversi data ke DataFrame untuk memudahkan manipulasi
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

# Menampilkan beberapa baris pertama dari dataset
print(df.head())

# Menampilkan statistik ringkasan
print(df.describe())

# Menampilkan informasi umum tentang DataFrame
print(df.info())

# Membuat pairplot untuk melihat distribusi data dan hubungan antar fitur
sns.pairplot(df, hue='species')
plt.show()

# Membuat boxplot untuk setiap fitur
plt.figure(figsize=(12, 6))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Memisahkan data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model klasifikasi SVM dengan kernel linear
svm = SVC(kernel='linear', random_state=42)

# Melatih model
svm.fit(X_train, y_train)

# Membuat prediksi
y_pred = svm.predict(X_test)

# Mengevaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Menampilkan laporan klasifikasi
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Membuat dan menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Menyimpan model yang telah dilatih
joblib.dump(svm, 'iris_model.pkl')
