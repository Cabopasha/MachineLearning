# Özdeğer Hesaplama - LucasBN Yöntemi

Bu proje, bir kare matrisin özdeğerlerini LucasBN olarak adlandırılan bir yöntemle (karakteristik denklem üzerinden) hesaplamayı amaçlar. Ayrıca sonuçları NumPy kütüphanesinin `eig` fonksiyonu ile karşılaştırır.

## İçerik

- `find\_eigenvalues(matrix)`: Verilen bir matrisin karakteristik denklemini kullanarak özdeğerlerini hesaplar.

- NumPy `eig` fonksiyonu kullanılarak hesaplanan özdeğerler ve özvektörlerle karşılaştırma yapılır.

- Kod, hem 2x2 hem de daha büyük boyutlu kare matrisler ile çalışabilir.

## Gereksinimler

- Python 3.x

- NumPy

Aşağıdaki komut ile gerekli bağımlılığı yükleyebilirsiniz:

```bash

pip install numpy

Python dosyasını çalıştırarak örnek bir matrisin özdeğerlerini hesaplayabilirsiniz.

python eigenvalue\_lucasbn.py

LucasBN yöntemiyle hesaplanan özdeğerler:

[7. 4. 4.]

NumPy'ın eig fonksiyonu ile hesaplanan özdeğerler:

[7. 4. 4.]

NumPy'ın eig fonksiyonu ile hesaplanan özvektörler:

[[ 0.11704115 -0.81649658 0.81649658]

[ 0.97563473 0.40824829 -0.40824829]

[-0.18526492 0.40824829 -0.40824829]]

**------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

\* get\_dimensions(matrix): Matrisin boyutlarını döner.

\* list\_multiply(list1, list2): İki listeyi polinom olarak çarpan fonksiyon.

\* list\_add(list1, list2, sub=1): İki listeyi (polinomları) toplayan veya çıkaran fonksiyon.

\* identity\_matrix(dimensions): Kimlik matrisini üretir.

\* characteristic\_equation(matrix): Karakteristik matrisin polinom halini üretir.

\* determinant\_equation(matrix): Karakteristik denklem için determinant polinomunu hesaplar.

\* find\_eigenvalues(matrix): Kökleri bulmak için karakteristik denklemi çözer.
