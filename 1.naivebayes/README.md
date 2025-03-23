# machine-learning
# yzm212
naive Bayes
# Naive Bayes İkili Sınıflandırma Uygulaması

## Proje Tanımı

Bu projede, Naive Bayes yöntemi kullanılarak ikili sınıflandırma gerçekleştirilmiştir. İki farklı yaklaşım uygulanmıştır:
- **Scikit-learn Modeli:** `GaussianNB` kullanılarak model eğitimi.
- **Custom Model:** Python ile sıfırdan oluşturulmuş (custom) Gaussian Naive Bayes modeli.

Amaç, iki modelin eğitim ve tahmin süreçlerini karşılaştırmak, ayrıca performanslarını (confusion matrix ve işlem süreleri) analiz etmektir.

## Veri Seti

Proje kapsamında kullanılan veri seti **Cancer_Data.csv** dosyasıdır. Veri setinin özellikleri:
- **Örnek Sayısı:** 569 örnek (satır)
- **Özellik Sayısı:** 33 özellik (sütun)
- **Hedef Değişken:** `diagnosis` sütunu (M: Malignant, B: Benign)

Veri ön işleme adımlarında:
- Gereksiz sütunlar (örn. `id` ve `Unnamed: 32`) kaldırılmış,
- `diagnosis` sütunu sayısallaştırılarak (M → 1, B → 0) ikili sınıflandırma için uygun hale getirilmiştir.

## Kullanılan Yöntem ve Kütüphaneler

### Kullanılan Kütüphaneler
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **time** (eğitim ve tahmin sürelerini ölçmek için)

### Yöntem
1. **Veri Ön İşleme:**
   - Veri seti okunmuş, gereksiz sütunlar kaldırılmış.
   - `diagnosis` sütunu sayısallaştırılmış.
   - Eğitim (%70) ve test (%30) setleri oluşturulmuştur.

2. **Model Eğitimi:**
   - **Scikit-learn Modeli:** `GaussianNB` kullanılarak model eğitilmiş ve test edilmiştir.
   - **Custom Model:** Python ile sıfırdan oluşturulan `CustomGaussianNB` sınıfı, her sınıf için ortalama, varyans ve öncelik değerlerini hesaplayarak model oluşturmuştur.

3. **Performans Ölçümü:**
   - Her iki model için eğitim (fit) ve tahmin (predict) süreleri `time` modülü kullanılarak ölçülmüştür.
   - Modellerin performansı, confusion matrix kullanılarak analiz edilip görselleştirilmiştir.

## Sonuçlar ve Tartışma

- **Performans Karşılaştırması:**  
  Her iki modelin oluşturduğu confusion matrix'ler karşılaştırılmıştır. Bu karşılaştırma, modellerin sınıflandırma başarısını gözler önüne sermektedir.

- **İşlem Süreleri:**  
  Eğitim ve tahmin süreleri ölçülmüş; bu sayede modellerin hesaplama verimliliği incelenmiştir.

- **Değerlendirme:**  
  Model performansının değerlendirilmesinde veri setinin sınıf dağılımı, özelliklerin yapısı ve yapılan veri ön işleme adımları önemli rol oynamaktadır. Her iki yaklaşım benzer doğruluk sonuçları verebilir; ancak custom modelin geliştirilmesi, temel algoritmanın anlaşılmasına katkı sağlamaktadır.

## Proje Dosya Yapısı

Kaynaklar :
https://www.youtube.com/watch?v=pMHZ3MHPPLo
https://chatgpt.com/
https://gemini.google.com/app?hl=tr
https://stackoverflow.com/questions/14254203/mixing-categorial-and-continuous-data-in-naive-bayes-classifier-using-scikit-lea
https://www.quora.com/What-is-the-best-way-to-use-continuous-variables-for-a-naive-bayes-classifier-Do-we-need-to-cluster-them-or-leave-for-self-learning-Pls-help
https://www.kaggle.com
https://www.kaggle.com/datasets/erdemtaha/cancer-data
