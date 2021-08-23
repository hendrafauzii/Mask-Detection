# Mask Detection

## Deskripsi
Program ini dibuat untuk melakukan pendeteksian masker pada muka seseorang secara _real time_ dengan menggunakan teknik _Deep Learning_. Algoritma yang digunakan pada program ini yaitu _Caffe Model SSD ResNet10_ yang digunakan untuk mendeteksi wajah, kemudian algoritma CNN _(Convolution Neural Network)_ yang dibangun menggunakan _framework_ **Keras** dan **TensorFlow**. Algoritma CNN digunakan untuk mengklasifikasikan wajah yang terdeteksi ke dua macam kelas yaitu _With Mask_ atau _Without Mask_.

## Dataset
Sebelum melakukan pelatihan model pastikan anda sudah mendownload _dataset_ dari https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset. _Dataset_ ini digunakan untuk klasifikasi deteksi masker wajah dengan gambar. _Dataset_ terdiri dari hampir 12K gambar dengan ukuran hampir 328,92 MB.

## Instalasi
1. _Clone repository_

       git clone https://github.com/hendrafauzii/Mask-Detection.git
       
2. Ubah _directory_ anda ke repository yang sudah di _clone_

       cd Mask-Detection 
       
3. Instal _libraries_ yang dibutuhkan

       pip install -r requirements.txt


## Pelatihan Model dan Uji Coba
1. Buka file "train_mask_model.ipynb" dengan Jupyter
2. Ubah directory dataset pada baris code "path_dataset" ke directory dataset anda 
3. Pilih menu "Run ALL"
4. Tunggu proses pelatihan model hingga selesai (proses pelatihan ini memakan waktu cukup lama, tergantung spesifikasi PC anda)
5. Setelah proses pelatihan selesai anda akan mendapatkan file "mask_model.tflite"
6. Untuk mendeteksi masker wajah secara _real-time_ ketik perintah berikut:

       python main.py
