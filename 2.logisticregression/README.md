# **Logistic Regression ile İkili Sınıflandırma Projesi**

# **Logistic Regression ile İkili Sınıflandırma Projesi**

## **Proje Tanımı**
Bu projede, meme kanseri veri seti kullanılarak \*\*logistic regression\*\* yöntemi ile ikili sınıflandırma yapılmıştır. İki farklı model uygulanmıştır:

- - \*\*Scikit-learn\*\* kütüphanesi kullanılarak hazır Logistic Regression modeli.
- - \*\*Custom (Gradient Descent Tabanlı)\*\* Logistic Regression modeli: Bu model, maximum likelihood estimation tabanlı cost function ve gradient descent algoritması kullanılarak sıfırdan yazılmıştır.

Projenin amacı, iki modelin eğitim ve test süreleri, performans (accuracy, confusion matrix) gibi metrikler üzerinden karşılaştırılmasını ve model optimizasyonu açısından farkların teorik olarak tartışılmasını sağlamaktır.

## **Veri Seti**
Veri seti [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) üzerinden alınmış olup, aşağıdaki özelliklere sahiptir:

- - \*\*Örnek Sayısı:\*\* 569
- - \*\*Özellik Sayısı:\*\* 30 (farklı ölçümler ve istatistiksel değerler)
- - \*\*Hedef Sütunu:\*\* `diagnosis` (B: benign, M: malignant)  

`  `Bu sütun, proje kapsamında B=0, M=1 olacak şekilde \*\*binary target\*\* sütununa dönüştürülmüştür.

- - Gereksiz sütunlar ("id" ve "Unnamed: 32") projeden çıkarılmıştır.

## **Kullanılan Yöntemler ve Araçlar**
- - \*\*Python:\*\* Temel programlama dili.
- - \*\*Pandas ve NumPy:\*\* Veri yükleme, ön işleme ve hesaplamalar için.
- - \*\*Matplotlib & Seaborn:\*\* Veri görselleştirme (confusion matrix, cost convergence grafikleri).
- - \*\*Scikit-learn:\*\* Hazır Logistic Regression modelinin uygulanması ve veri setinin train-test bölünmesi, model değerlendirme metrikleri.
- - \*\*StandardScaler:\*\* Model eğitimi öncesinde verinin normalize edilmesi.
- - \*\*Time modülü:\*\* Model eğitim ve test sürelerinin ölçülmesi.
- - \*\*Custom Logistic Regression:\*\* Kendi implementasyonumuz, gradient descent algoritması ile parametre güncellemesi ve cost convergence analizi yapılmaktadır.

## **Model Eğitimi ve Karşılaştırma**
### **Scikit-learn Logistic Regression**
- - \*\*Veri Normalize Edilmiştir:\*\* StandardScaler ile eğitim ve test verileri ölçeklendirilmiştir.
- - \*\*Eğitim Parametreleri:\*\* `max\_iter=5000` olarak belirlenmiş, convergence uyarısının azaltılması sağlanmıştır.
- - \*\*Performans:\*\* Eğitim ve test süreleri ölçülmüş, confusion matrix ve accuracy hesaplanmıştır.

### **Custom Logistic Regression (Gradient Descent)**
- - \*\*Kendi İmplementasyonumuz:\*\* Sıfırdan yazılmış modelde gradient descent algoritması kullanılmıştır.
- - \*\*Sigmoid Fonksiyonu:\*\* `np.clip` kullanılarak aşırı değerlerden kaynaklanabilecek overflow hataları önlenmiştir.
- - \*\*Performans:\*\* Eğitim ve test süreleri, cost convergence grafiği, confusion matrix ve accuracy karşılaştırması yapılmıştır.

## **Sonuçlar**
- - \*\*Scikit-learn Modeli:\*\* Yüksek accuracy (%97 civarında) elde edilmiştir.
- - \*\*Custom Model:\*\* Accuracy %93 civarında gözlemlenmiş, ancak modelin cost fonksiyonunun ilerleyişi görselleştirilmiştir.

## **Tartışma**
- - \*\*Veri Ön İşleme:\*\* Verinin normalize edilmesi, gradient descent algoritmasının daha stabil çalışmasını sağlamıştır.
- - \*\*Optimizasyon Yöntemleri:\*\* Scikit-learn, optimize edilmiş algoritmalar (lbfgs, liblinear vb.) kullanırken, custom modelde gradient descent ile manuel optimizasyon yapılmıştır.
- - \*\*Performans Metrikleri:\*\* Confusion matrix, accuracy gibi metrikler üzerinden yapılan karşılaştırma, modelin veri setine uygunluğunu göstermiştir.

## **Sonuç**
Bu proje, logistic regression algoritmasının iki farklı yöntemle uygulanmasını ve karşılaştırılmasını içermektedir.

