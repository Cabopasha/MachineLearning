# Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Veri Yükleme
df = pd.read_csv(r"C:\Users\ENES\OneDrive\Masaüstü\Cancer_Data.csv")

# 2. Veri Ön İşleme
# Gereksiz sütunları kaldırma
df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors='ignore')

# "diagnosis" sütununu binary hedef sütununa dönüştürme: B=0, M=1.
df["target"] = df["diagnosis"].map({"B": 0, "M": 1})
df.drop(columns=["diagnosis"], inplace=True)

# Veri setinin ilk 5 satırını görüntüle
print("Veri setinin ilk 5 satırı:")
print(df.head())

# Veri seti bilgisi
print("\nVeri seti bilgisi:")
print(df.info())

print("\nTemel istatistiksel bilgiler:")
print(df.describe())

# Eksik verilerin kontrolü
print("\nEksik veri sayıları:")
print(df.isnull().sum())

# Eksik veriler varsa düzeltme
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Sınıf dağılımı
print("\nSınıf dağılımı (target):")
print(df['target'].value_counts())

# 3. Veri Setinin Train-Test Bölünmesi
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3.1 Verilerin Normalizasyonu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Scikit-learn ile Logistic Regression Modeli
print("\n--- Scikit-learn Logistic Regression ---")
start_train = time.time()
model_sk = LogisticRegression(max_iter=5000)
model_sk.fit(X_train, y_train)
end_train = time.time()
train_time_sk = end_train - start_train

start_test = time.time()
y_pred_sk = model_sk.predict(X_test)
end_test = time.time()
test_time_sk = end_test - start_test

cm_sk = confusion_matrix(y_test, y_pred_sk)
acc_sk = accuracy_score(y_test, y_pred_sk)

print("Eğitim süresi (Scikit-learn):", train_time_sk)
print("Test süresi (Scikit-learn):", test_time_sk)
print("Scikit-learn Model Accuracy:", acc_sk)
print("Confusion Matrix (Scikit-learn):\n", cm_sk)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_sk, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Scikit-learn Logistic Regression - Confusion Matrix")
plt.show()


# 5. Custom Logistic Regression Modeli (Gradient Descent)
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _sigmoid(self, z):
        # z değerini güvenli aralığa çekerek overflow sorunlarını önleme
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
        self.costs = []

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.theta) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Binary cross-entropy loss
            cost = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.costs.append(cost)

            # Gradient hesaplamaları
            d_theta = np.dot(X.T, (y_predicted - y)) / len(y)
            d_bias = np.mean(y_predicted - y)

            # Parametre güncellemesi
            self.theta -= self.learning_rate * d_theta
            self.bias -= self.learning_rate * d_bias

    def predict_prob(self, X):
        linear_model = np.dot(X, self.theta) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_prob(X)
        return (y_pred_prob >= threshold).astype(int)


print("\n--- Custom Logistic Regression (Gradient Descent) ---")
# Verileri numpy array'e dönüştürme (zaten scaler'dan geçti)
X_train_np = X_train
X_test_np = X_test
y_train_np = y_train.values
y_test_np = y_test.values

start_train = time.time()
custom_model = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
custom_model.fit(X_train_np, y_train_np)
end_train = time.time()
train_time_custom = end_train - start_train

start_test = time.time()
y_pred_custom = custom_model.predict(X_test_np)
end_test = time.time()
test_time_custom = end_test - start_test

cm_custom = confusion_matrix(y_test_np, y_pred_custom)
acc_custom = accuracy_score(y_test_np, y_pred_custom)

print("Eğitim süresi (Custom Model):", train_time_custom)
print("Test süresi (Custom Model):", test_time_custom)
print("Custom Model Accuracy:", acc_custom)
print("Confusion Matrix (Custom Model):\n", cm_custom)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Custom Logistic Regression - Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(custom_model.n_iterations), custom_model.costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Custom Logistic Regression Cost Convergence")
plt.show()
