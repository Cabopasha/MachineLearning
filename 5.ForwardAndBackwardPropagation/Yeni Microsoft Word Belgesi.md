\# YZM212 - Sinir Ağı ile Kanser Tanısı (İleri-Geri Yayılım)

Bu projede Amaç, NumPy kullanarak ileri ve geri yayılım algoritmalarını sıfırdan implemente ederek, bir yapay sinir ağı ile \*\*kanser tanı sınıflandırması\*\* yapmaktır.

\##  Problem Tanımı

Kullanılan veri seti: \*\*Breast Cancer Wisconsin (Diagnostic) Dataset\*\*

Hedef: `diagnosis` sütununa göre tümörün \*\*malign (M)\*\* ya da \*\*benign (B)\*\* olduğunu sınıflandırmak.

\##  Kullanılan Kütüphaneler

\- NumPy

\- Pandas

\- Matplotlib

\- Seaborn

\- Scikit-learn (sadece `confusion\_matrix` ve `accuracy\_score` için)

**Model Özellikleri**

- Giriş nöronları: 30
- Gizli katman: 10 nöron, ReLU aktivasyon
- Çıkış katmanı: 1 nöron, Sigmoid aktivasyon
- Eğitim: 1000 epoch, gradyan iniş
- Kayıp fonksiyonu: Binary Cross-Entropy

` `**Çıktılar**

` `**Doğruluk (Accuracy)**

Model doğruluğu: %96-98 aralığında (rastgele ağırlıklara bağlı)

` `**Kayıp Grafiği**

Grafik, eğitim sırasında modelin ne kadar iyi öğrendiğini gösterir.

` `**Karışıklık Matrisi**

Sınıflandırma başarılarını ve hataları görselleştirir.

` `**Teorik Bilgi**

Bu projede aşağıdaki yapay sinir ağı bileşenleri manuel olarak kodlanmıştır:

- İleri Yayılım (Forward Propagation)
- Geri Yayılım (Backward Propagation)
- Aktivasyon fonksiyonları: ReLU, Sigmoid
- Gradyan İniş (Gradient Descent)
- Kaybın (Loss) hesaplanması ve görselleştirilmesi

` `**Kaynaklar**

- <https://numpy.org/>
- https://pandas.pydata.org/
- <https://scikit-learn.org/>
- <https://en.wikipedia.org/wiki/Backpropagation>


