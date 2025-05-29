# Laporan Proyek Machine Learning - Diana Mulhimah

## Klasifikasi Tingkat Kemiskinan di Indonesia

## Domain Proyek

Tingkat kemiskinan merupakan salah satu indikator utama dalam mengukur kesejahteraan dan keberhasilan pembangunan suatu daerah. Di Indonesia, meskipun angka kemiskinan cenderung menurun secara nasional, masih terdapat disparitas signifikan antar daerah. Berdasarkan laporan [^1]. Ketimpangan sosial dan akses terhadap layanan dasar seperti pendidikan, sanitasi, dan pekerjaan formal masih menjadi masalah struktural yang menyebabkan ketimpangan tingkat kemiskinan di setiap kabupaten/kota.<br/>

Seiring dengan berkembangnya teknologi, analisis prediktif menggunakan algoritma machine learning menjadi pendekatan yang sangat potensial untuk memetakan serta mengklasifikasikan tingkat kemiskinan secara objektif dan efisien. Dengan mengandalkan data sekunder seperti Indeks Pembangunan Manusia (IPM), rata-rata lama sekolah, pengeluaran per kapita, akses sanitasi, air minum layak, dan tingkat pengangguran terbuka, kita dapat membangun model prediksi yang akurat. Studi [^2] menunjukkan bahwa pendekatan klasifikasi dapat membantu dalam memahami tingkat keparahan kemiskinan di berbagai wilayah secara lebih objektif. Hal ini diperkuat oleh temuan [^3] yang mengidentifikasi bahwa Indeks Pembangunan Manusia (IPM) dan Tingkat Pengangguran Terbuka (TPT) memiliki pengaruh signifikan terhadap tingkat kemiskinan di Indonesia.<br/>

Menurut [^4], model Decision Tree terbukti efektif dalam mengklasifikasikan penerima bantuan sosial. Selain itu, [^5] juga menunjukkan bahwa teknik klasifikasi seperti ini dapat meningkatkan ketepatan pengambilan kebijakan berbasis data. Kedua studi tersebut menekankan pentingnya analisis prediktif dalam program-program intervensi sosial.<br/>
Oleh karena itu, pendekatan analisis prediktif berbasis data sangat penting untuk mendukung kebijakan tepat sasaran, hasil klasifikasi prediktif dapat menjadi landasan kuat bagi pemerintah dalam menentukan wilayah yang membutuhkan prioritas bantuan sosial dan program pembangunan. Dengan pendekatan ini, proses klasifikasi dan pemetaan kemiskinan dapat dilakukan secara cepat dan menyeluruh terhadap ratusan wilayah tanpa perlu survei manual yang mahal dan memakan waktu.<br/>

## Business Understanding

### Problem Statements
- Algoritma klasifikasi mana yang paling akurat dalam memprediksi tingkat kemiskinan?
- Bagaimana menangani Ketidakseimbangan distribusi Kelas?<br/>

### Goals
- Mengidentifikasi algoritma klasifikasi yang memberikan performa terbaik dalam memprediksi tingkat kemiskinan. 
- Mengatasi Ketidakseimbangan Data dengan SMOTE untuk menyeimbangkan distribusi kelas dalam dataset.

### Solution Statements
- Menerapkan dan membandingkan kinerja beberapa model klasifikasi: Decision Tree, Random Forest, Support Vector Machine (SVM), NB, dan K-Nearest Neighbors (KNN).
- Evaluasi performa model menggunakan metrik: **accuracy, precision, recall, F1-score**, dan **confusion matrix** untuk melihat performa tiap kelas.
- Menerapkan teknik SMOTE (*Synthetic Minority Over-sampling Technique*) untuk menghasilkan data sintetis pada kelas minoritas sehingga distribusi kelas menjadi lebih seimbang.

## Data Understanding
Dataset yang digunakan dalam proyek ini berjudul **Klasifikasi Tingkat Kemiskinan di Indonesia**, yang bersumber dari [Kaggle](https://https://www.kaggle.com/). [Dataset Klasifikasi Tingkat Kemiskinan di Indonesia: ](https://www.kaggle.com/datasets/ermila/klasifikasi-tingkat-kemiskinan-di-indonesia/).
Dataset ini berisi data indikator sosial-ekonomi dari berbagai Kabupaten/Kota di Indonesia yang digunakan untuk memprediksi dan mengklasifikasikan tingkat kemiskinan suatu wilayah dan dataset ini memiliki 13 kolom yang terdiri dari fitur numerik dan kategorikal.<br/> 

Variabel-variabel pada Klasifikasi Tingkat Kemiskinan di Indonesia Kaggle dataset adalah sebagai berikut:
- `Provinsi`: Nama provinsi tempat wilayah berada.
- `Kab/Kota`: Nama kabupaten/kota.
- `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (%)`: Persentase penduduk yang berada di bawah garis kemiskinan.
- `Rata-rata Lama Sekolah Penduduk 15+ (Tahun)`: Rerata tahun pendidikan formal yang diselesaikan oleh penduduk usia 15 tahun ke atas.
- `Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)`: Rata-rata pengeluaran per kapita dalam ribuan rupiah per tahun.
- `ndeks Pembangunan Manusia (IPM)`: Indeks gabungan yang mengukur pembangunan manusia dari aspek pendidikan, kesehatan, dan pengeluaran.
- `Umur Harapan Hidup (Tahun)`: Usia harapan hidup saat lahir di wilayah tersebut.
_ `Persentase Rumah Tangga yang Memiliki Akses terhadap Sanitasi Layak (%)`: Proporsi rumah tangga dengan fasilitas sanitasi yang layak.
- `Persentase Rumah Tangga yang Memiliki Akses terhadap Air Minum Layak (%)`: Proporsi rumah tangga dengan akses terhadap sumber air minum yang layak.
- `Tingkat Pengangguran Terbuka (%)`: Persentase penduduk usia kerja yang sedang tidak bekerja dan sedang mencari pekerjaan.
- `Tingkat Partisipasi Angkatan Kerja (%)`: Persentase penduduk usia kerja yang masuk dalam angkatan kerja.
- `PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)`: Produk Domestik Regional Bruto wilayah berdasarkan pengeluaran.
- `Klasifikasi Kemiskinan`: (**Target Label**) Kategori tingkat kemiskinan: `0` Tidak Miskin `1` Miskin


Rubrik/Kriteria Tambahan (Opsional):
Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
- Jumlah total baris (entri): 999 baris (baris ke-0 sampai ke-998).
- Jumlah kolom: 13 kolom. 10 kolom bertipe object/string dan 3 kolom bertipe float64.
- Namun banyak kolom hanya memiliki 514 nilai non-null, artinya hanya 514 baris yang memiliki data, sisanya (999 - 514 = 485 baris) kosong/null.

| No | Kolom                                         | Non-Null | Tipe Data | Catatan                   |
| -- | --------------------------------------------- | -------- | --------- | ------------------------- |
| 0  | Provinsi                                      | 514      | object    | Nama provinsi             |
| 1  | Kab/Kota                                      | 514      | object    | Nama kabupaten/kota       |
| 2  | Persentase Penduduk Miskin (P0)               | 514      | object    | Perlu dikonversi ke float |
| 3  | Rata-rata Lama Sekolah Penduduk 15+           | 514      | object    | Perlu dikonversi ke float |
| 4  | Pengeluaran per Kapita Disesuaikan            | 514      | float64   | Sudah numerik             |
| 5  | Indeks Pembangunan Manusia                    | 514      | object    | Perlu dikonversi ke float |
| 6  | Umur Harapan Hidup                            | 514      | object    | Perlu dikonversi ke float |
| 7  | Persentase rumah tangga akses sanitasi layak  | 514      | object    | Perlu dikonversi ke float |
| 8  | Persentase rumah tangga akses air minum layak | 514      | object    | Perlu dikonversi ke float |
| 9  | Tingkat Pengangguran Terbuka                  | 514      | object    | Perlu dikonversi ke float |
| 10 | Tingkat Partisipasi Angkatan Kerja            | 514      | object    | Perlu dikonversi ke float |
| 11 | PDRB Konstan menurut Pengeluaran              | 514      | float64   | Sudah benar               |
| 12 | Klasifikasi Kemiskinan                        | 514      | float64   | Sudah benar               |

- Data Tidak Lengkap: Hanya 514 dari 999 baris yang memiliki data, sisanya mungkin baris kosong atau placeholder. Ini perlu dibersihkan.
- Banyak Kolom Bertipe object Padahal Seharusnya Numerik: Kolom seperti `Persentase Penduduk Miskin`, `Lama Sekolah`, `IPM`, `dll`. masih berupa string (object) dan mungkin mengandung simbol (%, koma, atau spasi).


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Rubrik/Kriteria Tambahan (Opsional):

Menjelaskan proses data preparation yang dilakukan
Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Rubrik/Kriteria Tambahan (Opsional):

Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. Jelaskan proses improvement yang dilakukan.
Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik akurasi, precision, recall, dan F1 score. Jelaskan mengenai beberapa hal berikut:

Penjelasan mengenai metrik yang digunakan
Menjelaskan hasil proyek berdasarkan metrik evaluasi
Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

Rubrik/Kriteria Tambahan (Opsional):

Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.


## Referensi
[^1]: Badan Pusat Statistik Indonesia, “Profil Kemiskinan di Indonesia Maret 2023,” Badan Pusat statistik, no. 57, 2023.
[^2]:	N. P. N. Hendayanti and M. Nurhidayati, “KLASIFIKASI TINGKAT KEPARAHAN KEMISKINAN PROVINSI DI INDONESIA DENGAN ANALISIS DISKRIMINAN,” Math Educa Journal, vol. 5, no. 1, 2021, doi: 10.15548/mej.v5i1.2510.
[^3]:	R. F. Saragih, P. R. Silalahi, and K. Tambunan, “Pengaruh Indeks Pembangunan Manusia, Tingkat Pengangguran Terbuka Terhadap Tingkat Kemiskinan di Indonesia Tahun 2007 – 2021,” PESHUM : Jurnal Pendidikan, Sosial dan Humaniora, vol. 1, no. 2, 2022, doi: 10.56799/peshum.v1i2.36.
[^4]:	L. Qadrini, A. Sepperwali, and A. Aina, “Decision Tree Dan Adaboost Pada Klasifikasi Penerima Program Bantuan Sosial,” Jurnal Inovasi Penelitian, vol. 2, no. 7, 2021.
[^5]:	A. Anna, “PENGUJIAN TEKNIK ALGORITMA KLASIFIKASI TERHADAP TINGKAT KEMISKINAN PENDUDUK,” JTIK (Jurnal Teknik Informatika Kaputama), vol. 7, no. 1, 2023, doi: 10.59697/jtik.v7i1.35.
 

