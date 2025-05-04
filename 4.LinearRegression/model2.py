import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veri setini yükle
df = pd.read_csv("C:\Users\ENES\OneDrive\Masaüstü\antalya_kiralik_ev.csv")

# Gerekli sütunları seç
df_model = df[['fiyat', 'brut_alan_m2', 'net_alan_m2', 'oda_sayisi',
               'bina_kat_sayisi', 'banyo_sayisi', 'balkon', 'asansor',
               'esya_durumu', 'site_icinde', 'aidat']].copy()

# "oda_sayisi"nı sayıya çevir
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

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Tahmin ve MSE hesapla
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

print("Scikit-learn ile Linear Regression")
print("Katsayılar:", np.r_[model.intercept_, model.coef_])
print("MSE:", mse)
