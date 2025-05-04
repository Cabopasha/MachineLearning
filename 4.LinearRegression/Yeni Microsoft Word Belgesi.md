Lineer Regresyon

Bu Projede Ne Yapıldı?

Bu projede, Antalya'daki kiralık ev verilerini kullanarak ev fiyatlarını tahmin etmeye yönelik iki farklı doğrusal regresyon modeli geliştirildi:

1\. \*\*NumPy ile En Küçük Kareler Yöntemi (OLS - Ordinary Least Squares)\*\*

2\. \*\*Scikit-learn kütüphanesinin `LinearRegression` sınıfı ile model\*\*

Her iki modelde de aynı veri ön işleme adımları uygulandı ve aynı özellikler kullanıldı.

\---

Model Sonuçları ve Karşılaştırma

| Yöntem                    | Ortalama Kare Hatası (MSE)     |

\|---------------------------|-------------------------------|

| \*\*NumPy (Least Squares)\*\* | ≈ 159,889,572 TL²             |

| \*\*Scikit-learn\*\*          | ≈ 159,889,572 TL²             |

\*\*Gözlem:\*\* Her iki yöntem de neredeyse birebir aynı sonucu vermiştir.

\---

Neden Aynı Sonuçlar Elde Edildi?

\- Scikit-learn kütüphanesindeki `LinearRegression` sınıfı da arka planda \*\*en küçük kareler yöntemini (OLS)\*\* uygular.

\- NumPy ile elle hesaplanan katsayılar, Scikit-learn modelinin otomatik olarak bulduğu katsayılarla aynıdır.

\- Verilerin aynı olması ve modelin doğrusallığı nedeniyle iki yaklaşım da aynı tahminleri üretmiştir.




` `Kullanılan Özellikler

\- Brüt Alan (m²)

\- Net Alan (m²)

\- Oda Sayısı (örneğin "2+1" → 3)

\- Bina Kat Sayısı

\- Banyo Sayısı

\- Balkon (0/1)

\- Asansör (0/1)

\- Eşya Durumu (0/1)

\- Site İçinde (0/1)

\- Aidat (TL)

\---

` `Sonuç

\- Lineer regresyon yöntemiyle ev fiyatları başarılı bir şekilde tahmin edilmiştir.

\- NumPy ile model kurulumu, algoritmanın temellerini anlamak açısından önemlidir.

\- Scikit-learn ise daha pratik ve projelerde yaygın olarak kullanılır.


