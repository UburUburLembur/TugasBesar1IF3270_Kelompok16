# Tugas Besar 1 IF3270 Pembelajaran Mesin Feedforward Neural Network

Repository ini berisi implementasi jaringan saraf tiruan (Artificial Neural Network) dari awal menggunakan Python dan pustaka seperti NumPy, Matplotlib, dan Scikit-learn.

## Fitur
- Implementasi forward propagation dan backward propagation
- Berbagai fungsi aktivasi: ReLU, Sigmoid, Softmax, dll.
- Berbagai fungsi loss: MSE, Binary Cross Entropy, Categorical Cross Entropy
- Metode inisialisasi bobot seperti Xavier, He, dan Random
- Visualisasi bobot dan distribusi gradien
- Training dengan validasi dan plotting loss
- Kemampuan menyimpan dan memuat model

## Setup dan Instalasi
Pastikan Anda memiliki Python 3.x dan menginstal dependensi yang diperlukan:

```bash
pip install numpy matplotlib scikit-learn tqdm viznet
```

## Cara Menjalankan Program

### 1. Import Modul dan Load Dataset
```python
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
```

### 2. Preprocessing Data
```python
# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.expand_dims(y, axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Inisialisasi dan Training Model
```python
nn = NeuralNetwork(
    input_size=X.shape[1],
    hidden_layers=[16, 8],
    output_size=1,
    activations=["relu", "sigmoid"],
    loss_function="binary_cross_entropy",
    weight_init_methods=["xavier", "xavier"],
    weight_init_params=[{}, {}]
)

nn.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
```

### Parameter Inisialisasi Model
Saat membuat objek `NeuralNetwork`, berikut adalah parameter yang dapat digunakan:
- `input_size` (int): Jumlah fitur input.
- `hidden_layers` (list[int]): Daftar jumlah neuron pada setiap lapisan tersembunyi.
- `output_size` (int): Jumlah neuron pada lapisan output.
- `activations` (list[str]): Daftar fungsi aktivasi untuk setiap lapisan.
- `loss_function` (str): Jenis fungsi loss yang digunakan (misalnya, "mse", "binary_cross_entropy").
- `weight_init_methods` (list[str]): Metode inisialisasi bobot untuk setiap lapisan (misalnya, "xavier", "he").
- `weight_init_params` (list[dict]): Parameter tambahan untuk metode inisialisasi bobot.

### Parameter Pelatihan Model
Saat memanggil metode `train()`, berikut adalah parameter yang dapat digunakan:
- `X_train` (numpy.ndarray): Data fitur untuk pelatihan.
- `y_train` (numpy.ndarray): Label target untuk pelatihan.
- `epochs` (int): Jumlah iterasi training.
- `learning_rate` (float): Kecepatan pembelajaran model.
- `batch_size` (int): Jumlah sampel dalam setiap batch training.

### 4. Evaluasi dan Prediksi
```python
y_pred = nn.predict(X_test)
print("Prediksi:", y_pred)
```

### 5. Menyimpan dan Memuat Model
```python
nn.save_model("model.npz")
nn.load_model("model.npz")
```

## Visualisasi
Untuk melihat distribusi bobot dan loss selama training:
```python
nn.plot_loss()
nn.plot_weight_distribution([0, 1])
```

## Pembagian Kerja
### Muhammad Zakkiy (10122074)
- Mengerjakan syntax bagian forward propagation, backward propagation, train, predict, plot_loss, initialize_weights, load dataset mnist_784 dan menggunakan method fetch_openml.
- Melakukan pengujian untuk variasi fungsi aktivasi, learning rate, dan perbandingan dengan library sklearn MLP.
- Mengerjakan laporan bagian penjelasan implementasi, penjelasan forward propagation, penjelasan backward propagation dan weight update, pengaruh fungsi aktivasi, pengaruh learning rate, perbandingan dengan library sklearn, dan kesimpulan serta saran.
- Melakukan fiksasi syntax.
- Membuat repository github.

### Ghaisan Zaki Pratama (10122078)
- Mengerjakan syntax bagian fungsi aktivasi, turunan fungsi aktivasi, loss function, turunan loss function, plot bobot, plot distribusi bobot, representasi graf.
- Melakukan pengujian untuk variasi depth dan width dan inisialisasi bobot.
- Mengerjakan laporan bagian deskripsi persoalan, penjelasan implementasi, deskripsi kelas beserta atribut dan method, pengaruh depth dan width, dan pengaruh inisialisasi bobot.
- Membuat template laporan.
