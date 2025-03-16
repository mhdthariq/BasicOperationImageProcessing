import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Dapatkan path absolut ke gambar
base_dir = os.path.dirname(os.path.abspath(__file__))  # Direktori skrip saat ini
image_path = os.path.join(base_dir, "..", "Assets", "sunflower.jpg")  # Path ke gambar

# Coba baca gambar
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Gambar Asli')

# 2. Tampilkan sebagian kecil nilai RGB untuk referensi
rgb_values = img[:10, :10, :]
print("Nilai RGB (10x10 pertama):")
print(rgb_values)

# 3. Ubah ke grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 4. Resize gambar ke ukuran yang lebih kecil jika terlalu besar
max_size = 512
if max(img_gray.shape) > max_size:
    scale = max_size / max(img_gray.shape)
    new_size = (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale))
    img_resized = cv2.resize(img_gray, new_size)
else:
    img_resized = img_gray.copy()

# 5. Tampilkan nilai grayscale dari sebagian kecil area
gray_values = img_resized[:10, :10]
print("Nilai Grayscale (10x10 pertama):")
print(gray_values)

# 6. Terapkan filter Sobel
sobel_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)

# 7. Hitung magnitudo gradien
grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# 8. Tampilkan nilai M (10x10 pertama)
print("Nilai M (Magnitudo Gradien - 10x10 pertama):")
print(grad_magnitude[:10, :10])

# 9. Thresholding
thresholded_image = (grad_magnitude >= np.mean(grad_magnitude)).astype(np.uint8)

# 10. Tampilkan hasil
plt.subplot(2, 3, 2)
plt.imshow(img_resized, cmap='gray')
plt.title('Gambar Grayscale (Resized)')

plt.subplot(2, 3, 3)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel Horizontal')

plt.subplot(2, 3, 4)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Vertikal')

plt.subplot(2, 3, 5)
plt.imshow(grad_magnitude, cmap='gray')
plt.title('Magnitudo Gradien')

plt.subplot(2, 3, 6)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholding (1/0)')

plt.tight_layout()
plt.show()
