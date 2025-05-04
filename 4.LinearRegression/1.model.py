import pandas as pd
import numpy as np

# Veri setini yükle
df = pd.read_csv(""C:\Users\ENES\OneDrive\Masaüstü\antalya_kiralik_ev.csv"")

# Gerekli sütunları seç
df_model = df[['fiyat', 'brut_alan_m2', 'net_alan_m2', 'oda_sayisi',
               'bina_kat_sayisi', 'banyo_sayisi', 'balkon', 'asansor',
               'esya_durumu', 'site_icinde', 'aidat']].copy()

# "odasayisi'nı sayıya çevir
df_model['oda_sayisi'] = (
    df_model['oda_sayisi']
    .str.extract(r'(\d+)\+(\d+)')
    .astype(float)
    .sum(axis=1)
)

# Eksik verileri sil
df_model.dropna(inplace=True)

# Özellikler ve hedef değişkeni ayır
X = df_model.drop('fiyat', axis=1).values
y = df_model['fiyat'].values

# Bias için sabit 1'ler ekle
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# En küçük kareler çözümü
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Tahmin ve MSE hesapla
y_pred = X_b @ theta
mse = np.mean((y - y_pred) ** 2)

print("NumPy ile Linear Regression")
print("Katsayılar (theta):", theta)
print("MSE:", mse)

