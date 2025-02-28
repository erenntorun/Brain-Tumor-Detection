# Brain Tumor Image Classification

Bu proje, beyin tümörü tespit etmek için bir derin öğrenme modeli geliştirmeyi amaçlamaktadır. Model, Kaggle'dan alınan beyin tümörü veri seti ile eğitilmiş ve test edilmiştir.

## Veri Seti
Veri setine aşağıdaki bağlantıdan ulaşabilirsiniz:
[Kaggle - Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

## Proje İçeriği
Bu proje, beyin tümörü olup olmadığını belirleyen bir CNN (Convolutional Neural Network) modeli içermektedir. Model, eğitim ve test aşamalarından oluşmaktadır.

### Kullanılan Kütüphaneler
- OpenCV
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- TensorFlow/Keras
- Seaborn
- natsort

## Çalıştırma Adımları

### 1. Modeli Eğitme
```bash
python train.py
```
Bu komut, veri setini okuyarak bir CNN modeli eğitir ve modeli `BrainTumor150EpochsCategorical.keras` dosyasına kaydeder.

### 2. Modeli Test Etme
```bash
python test.py
```
Bu komut, eğitilmiş modeli yükleyerek yeni görüntüler üzerinde tahmin yapar ve sonuçları görselleştirir.
Not: Test için kendim oluşturduğum "pred" dosyasını projeye yükledim.

## Dosya Açıklamaları
- `train.py`: Modelin eğitimini gerçekleştiren Python kodu.
- `test.py`: Eğitilmiş modeli kullanarak yeni görüntüler üzerinde tahmin yapan Python kodu.

## Model Detayları
Model, aşağıdaki mimariye sahiptir:
- 3 adet evrişim (Convolutional) katmanı
- ReLU aktivasyon fonksiyonları
- Maksimum havuzlama (MaxPooling) katmanları
- Flatten katmanı
- 2 Tam bağlı (Dense) katman
- Softmax aktivasyon fonksiyonu
- Categorical Crossentropy kayıp fonksiyonu
- Adam optimizasyon algoritması

## Sonuçların Görselleştirilmesi
Eğitim sürecinde loss ve accuracy değerleri grafiksel olarak görselleştirilmiştir. Ayrıca, test aşamasında confusion matrix ve sınıflandırma raporu oluşturulmuştur.

## Lisans
Bu proje eğitim amaçlıdır ve herkese açık olarak paylaşılmaktadır.

