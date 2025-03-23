# Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Veri setinin yüklenmesi
df = pd.read_csv(r"C:\Users\ENES\OneDrive\Masaüstü\Cancer_Data.csv")

# Gereksiz sütunların kaldırılması (örn. 'id' ve 'Unnamed: 32')
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop(['Unnamed: 32'], axis=1, inplace=True)

# diagnosis sütununun sayısallaştırılması: M -> 1, B -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Özellikler (X) ve hedef (y) ayrımı
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Eğitim ve test setlerine ayırma (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#############################################
# 1. Scikit-learn Gaussian Naive Bayes Modeli #
#############################################
gnb = GaussianNB()

# Eğitim süresi ölçümü
start_time = time.time()
gnb.fit(X_train, y_train)
sklearn_fit_time = time.time() - start_time

# Tahmin süresi ölçümü
start_time = time.time()
y_pred_sklearn = gnb.predict(X_test)
sklearn_predict_time = time.time() - start_time

# Karmaşıklık matrisi hesaplama
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)


#############################################
# 2. Custom Gaussian Naive Bayes Modeli       #
#############################################
class CustomGaussianNB:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            # Varyansa çok küçük değerlerin sıfır bölünmesini önlemek için 1e-9 ekliyoruz.
            self.var[c] = np.var(X_c, axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / float(X.shape[0])

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Log öncelik (prior) hesaplanıyor
                log_prior = np.log(self.priors[c])
                # Tüm özellikler için log-likelihood hesaplanıyor
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
                log_likelihood -= 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
                posterior = log_prior + log_likelihood
                posteriors.append(posterior)
            # En yüksek log posterior değere sahip sınıf tahmin ediliyor
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)


# Custom modelin eğitimi ve tahmini
custom_model = CustomGaussianNB()

start_time = time.time()
custom_model.fit(X_train, y_train)
custom_fit_time = time.time() - start_time

start_time = time.time()
y_pred_custom = custom_model.predict(X_test)
custom_predict_time = time.time() - start_time

cm_custom = confusion_matrix(y_test, y_pred_custom)

#############################################
# 3. Sonuçların Görselleştirilmesi ve Karşılaştırılması
#############################################

# Eğitim ve tahmin sürelerini yazdırma
print("Scikit-learn Modeli - Eğitim Süresi: {:.6f} saniye".format(sklearn_fit_time))
print("Scikit-learn Modeli - Tahmin Süresi: {:.6f} saniye".format(sklearn_predict_time))
print("Custom Model - Eğitim Süresi: {:.6f} saniye".format(custom_fit_time))
print("Custom Model - Tahmin Süresi: {:.6f} saniye".format(custom_predict_time))

# Karmaşıklık matrislerinin yazdırılması
print("\nScikit-learn Modeli için Karmaşıklık Matrisi:")
print(cm_sklearn)

print("\nCustom Model için Karmaşıklık Matrisi:")
print(cm_custom)

# Karmaşıklık matrislerini görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn, display_labels=gnb.classes_)
disp1.plot(ax=ax[0], cmap=plt.cm.Blues)
ax[0].set_title("Scikit-learn GaussianNB")

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=custom_model.classes)
disp2.plot(ax=ax[1], cmap=plt.cm.Greens)
ax[1].set_title("Custom GaussianNB")

plt.tight_layout()
plt.show()

#yzm212
