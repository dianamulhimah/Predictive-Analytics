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

Informasi Awal Dataset 
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

- Jumlah total baris (entri): 999 baris (baris ke-0 sampai ke-998).
- Jumlah kolom: 13 kolom. 10 kolom bertipe object/string dan 3 kolom bertipe float64.
- Data Tidak Lengkap: Hanya 514 dari 999 baris yang memiliki data, sisanya mungkin baris kosong atau placeholder. Ini perlu dibersihkan.
- Banyak Kolom Bertipe object Padahal Seharusnya Numerik: Kolom seperti `Persentase Penduduk Miskin`, `Lama Sekolah`, `IPM`, `dll`. masih berupa string (object) dan mungkin mengandung simbol (%, koma, atau spasi). perlu diKonversi Data Teks ke Numerik.
- Dataset ini belum siap untuk dianalisis secara langsung karena banyak kolom numerik masih tersimpan sebagai teks/string, dan sekitar 48,5% baris memiliki nilai kosong.

Data Cleaning
| No. | Kolom                   | Non-Null Count | Tipe Data |
|-----|-------------------------|----------------|-----------|
| 1   | `provinsi`              | 514            | object    |
| 2   | `kota`                  | 514            | object    |
| 3   | `persen_miskin`         | 514            | float64   |
| 4   | `lama_sekolah`          | 514            | float64   |
| 5   | `pengeluaran_kapita`   | 514            | float64   |
| 6   | `ipm`                   | 514            | float64   |
| 7   | `umur_harapan`          | 514            | float64   |
| 8   | `akses_sanitasi`        | 514            | float64   |
| 9   | `akses_air`             | 514            | float64   |
| 10  | `pengangguran`          | 514            | float64   |
| 11  | `partisipasi_kerja`     | 514            | float64   |
| 12  | `pdrb`                  | 514            | float64   |
| 13  | `klasifikasi_kemiskinan`| 514            | float64   |

- Semua kolom telah diubah ke format snake_case yang lebih pendek dan konsisten, mempermudah pemanggilan dan analisis.
- Kolom yang seharusnya numerik tapi bertipe object (karena simbol koma ,, persen %, atau spasi) dibersihkan:
  - Mengganti `,` menjadi `.` sebagai desimal (sesuai format numerik Python)
  - Menghapus `%`
  - Menghapus spasi
- Konversi ke float64 menggunakan pd.to_numeric()
- Terdapat 484 baris duplikat berasal dari 485 baris kosong, hanya satu baris kosong yang unik, sisanya duplikat dari baris kosong tersebut.
- Pembersihan Dilakukan dengan `dropna()` menghapus 485 baris kosong dan tersisa 514 baris lengkap dan bersih

Data Statistik
| Statistik        | persen\_miskin | lama\_sekolah | pengeluaran\_kapita | ipm     | umur\_harapan | akses\_sanitasi | akses\_air | pengangguran | partisipasi\_kerja | pdrb         | klasifikasi\_kemiskinan |
| ---------------- | -------------- | ------------- | ------------------- | ------- | ------------- | --------------- | ---------- | ------------ | ------------------ | ------------ | ----------------------- |
| **count**        | 514.000        | 514.000       | 514.000             | 514.000 | 514.000       | 514.000         | 514.000    | 514.000      | 514.000            | 514.000      | 514.000                 |
| **mean**         | 12.273         | 8.437         | 10324.788           | 69.927  | 69.657        | 77.202          | 85.137     | 5.059        | 69.464             | 21964077.06  | 0.121                   |
| **std**          | 7.459          | 1.631         | 2717.144            | 6.497   | 3.447         | 18.584          | 15.702     | 2.637        | 6.396              | 47904922.45  | 0.326                   |
| **min**          | 2.380          | 1.420         | 3976.000            | 32.840  | 55.430        | 0.000           | 0.000      | 0.000        | 56.390             | 147485.00    | 0.000                   |
| **25% (Q1)**     | 7.150          | 7.510         | 8574.000            | 66.643  | 67.388        | 70.218          | 79.043     | 3.180        | 65.070             | 3654292.00   | 0.000                   |
| **50% (Median)** | 10.455         | 8.305         | 10196.500           | 69.610  | 69.975        | 81.800          | 89.795     | 4.565        | 68.955             | 8814926.00   | 0.000                   |
| **75% (Q3)**     | 14.888         | 9.338         | 11719.000           | 73.113  | 72.043        | 89.883          | 96.400     | 6.530        | 72.343             | 19735100.00  | 0.000                   |
| **max**          | 41.660         | 12.830        | 23888.000           | 87.180  | 77.730        | 99.970          | 100.000    | 13.370       | 97.930             | 460081000.00 | 1.000                   |

- Hampir semua indikator menunjukkan ketimpangan yang cukup ekstrem antar daerah — terutama pada: Kemiskinan (maks 41.66% vs min 2.38%), IPM (maks 87.18 vs min 32.84), PDRB (sangat besar rentangnya)
- Daerah dengan IPM rendah cenderung juga memiliki lama sekolah rendah, pengeluaran rendah, dan akses sanitasi buruk.
- PDRB tinggi biasanya berasosiasi dengan akses air/sanitasi yang baik, lama sekolah tinggi, dan IPM tinggi.
- Nilai 0 pada akses sanitasi/air atau pengangguran perlu ditelusuri lebih lanjut: apakah itu memang kenyataan, data hilang yang tidak ditandai dengan NaN, atau error input
- Nilai 0 pada kolom `akses_sanitasi`, `akses_air`, `pengangguran` tersebut dianggap outlier atau kesalahan pengisian data, karena secara logika sangat tidak mungkin suatu wilayah memiliki akses air, sanitasi, atau pengangguran 0% secara absolut.
- Kolom klasifikasi_kemiskinan sangat tidak seimbang, dengan mayoritas bernilai 0. 

Pembersihan dan imputasi data
| Statistik | Akses Sanitasi (%) | Akses Air (%) | Pengangguran (%) |
| --------: | -----------------: | ------------: | ---------------: |
|     Count |                514 |           514 |              514 |
|      Mean |              77.36 |         85.31 |             5.08 |
|       Std |              18.27 |         15.25 |             2.62 |
|       Min |               0.26 |          0.87 |             0.41 |
|       25% |              70.27 |         79.18 |             3.20 |
|       50% |              81.84 |         89.82 |             4.59 |
|       75% |              89.88 |         96.40 |             6.53 |
|       Max |              99.97 |        100.00 |            13.37 |

- Pada tahap ini, dilakukan pembersihan dan imputasi data terhadap atribut `akses_sanitasi`, `akses_air`, dan `pengangguran`. Ditemukan nilai 0 yang secara logika tidak mungkin terjadi dan dianggap sebagai missing value. Nilai 0 digantikan dengan NaN, lalu di-imputasi menggunakan nilai median, agar tidak terlalu terpengaruh oleh outlier.

![Bloxplot](https://
![Bloxplot](https://github.com/user-attachments/assets/ae677a9c-3342-40a9-a4a3-0a5c07a38e7f)

- `persen_miskin` Terlihat banyak outlier di atas (daerah dengan persentase kemiskinan sangat tinggi). Median kemiskinan sekitar 10–12%.
- `lama_sekolah` Terdapat beberapa outlier di bawah (daerah dengan rata-rata lama sekolah < 4 tahun). Median sekitar 8–9 tahun.
- `pengeluaran_kapita` Distribusi cukup miring ke kanan (right-skewed) dengan banyak outlier di atas. Artinya, sebagian besar daerah berpengeluaran rendah, dan hanya sedikit daerah dengan pengeluaran tinggi.
- `ipm` (Indeks Pembangunan Manusia) Distribusi relatif simetris, namun ada outlier rendah. Median sekitar 70.
- `umur_harapan` Tidak terlalu banyak outlier, relatif normal. Median sekitar 70 tahun.
- `akses_sanitasi` dan `akses_air` Keduanya memiliki banyak outlier di bawah. Ini mengindikasikan ketimpangan besar dalam akses layanan dasar (banyak daerah masih tertinggal).
- `pengangguran` Memiliki beberapa outlier tinggi, yang menandakan daerah dengan pengangguran ekstrem. Median sekitar 4–5%.
- `partisipasi_kerja` Hampir simetris, namun ada beberapa outlier tinggi (di atas 85%). Median sekitar 70%.
- `pdrb` Distribusi sangat skewed ke kanan. Banyak outlier sangat tinggi: daerah kaya seperti DKI Jakarta, Papua, dll.
- `klasifikasi_kemiskinan` Mayoritas daerah masuk kategori `0` (tidak miskin), hanya sedikit yang termasuk `1` (miskin)`. Ini menunjukkan imbalance class (perlu penanganan khusus saat modeling, seperti oversampling SMOTE.
- **Banyak outlier signifikan di hampir semua fitur Dibiarkan karena mencerminkan realitas daerah tertinggal/kaya**.



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
 

