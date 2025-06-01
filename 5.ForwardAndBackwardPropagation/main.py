import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Veri setini yükle (yolunu kendi bilgisayarına göre güncelle)
df = pd.read_csv(r"C:\Users\ENES\OneDrive\Masaüstü\Cancer_Data.csv")

# Gereksiz sütunları kaldır
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Etiket sütununu sayısallaştır (M: 1, B: 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Giriş (X) ve hedef (y) değişkenlerini ayır
X = df.drop("diagnosis", axis=1).values  # shape: (569, 30)
y = df["diagnosis"].values.reshape(-1, 1)  # shape: (569, 1)

# Özellikleri min-max normalizasyon ile [0,1] aralığına ölçekle
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Transpoz al: (features, samples) => (30, 569), y: (1, 569)
X = X.T
y = y.T

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate):
        np.random.seed(42)  # Sabit rasgelelik
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01  # (10, 30)
        self.b1 = np.zeros((hidden_size, 1))  # (10, 1)
        self.W2 = np.random.randn(1, hidden_size) * 0.01  # (1, 10)
        self.b2 = np.zeros((1, 1))  # (1, 1)
        self.lr = learning_rate

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1       # (10, 569)
        A1 = self.relu(Z1)                  # (10, 569)
        Z2 = self.W2.dot(A1) + self.b2      # (1, 569)
        A2 = self.sigmoid(Z2)               # (1, 569)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_cost(self, A2, Y):
        m = Y.shape[1]
        eps = 1e-8
        return -np.sum(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)) / m

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        dZ2 = A2 - Y                         # (1, 569)
        dW2 = (1/m) * dZ2.dot(A1.T)         # (1, 10)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * self.relu_derivative(Z1)  # (10, 569)
        dW1 = (1/m) * dZ1.dot(X.T)                           # (10, 30)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        A2, _ = self.forward(X)
        return (A2 > 0.5).astype(int)

# Parametreler
input_size = X.shape[0]  # 30
hidden_size = 10
learning_rate = 0.01
epochs = 1000

# Modeli başlat
nn = NeuralNetwork(input_size, hidden_size, learning_rate)

# Eğitim döngüsü
losses = []
for i in range(epochs):
    A2, cache = nn.forward(X)
    cost = nn.compute_cost(A2, y)
    nn.backward(cache, y)
    losses.append(cost)

    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {cost:.4f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Kaybı (Loss) Grafiği")
plt.grid(True)
plt.show()

# Tahmin
y_pred = nn.predict(X)

# Skor
accuracy = accuracy_score(y.flatten(), y_pred.flatten())
print(f"Model doğruluğu: {accuracy * 100:.2f}%")

# Karışıklık matrisi
cm = confusion_matrix(y.flatten(), y_pred.flatten())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Karışıklık Matrisi")
plt.show()
