import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar
img = cv2.imread('../Assets/sunflower.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Gambar Asli')

# 2. Tampilkan nilai RGB asli (5x5 pertama)
rgb_values = img[:5, :5, :]
print("Nilai RGB (5x5 pertama):")
print(rgb_values)

# 3. Ubah ke grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 4. Resize gambar menjadi 32x32
img_resized = cv2.resize(img_gray, (32, 32))

# 5. Tampilkan nilai grayscale dari 5x5 pertama
gray_values = img_resized[:5, :5]
print("Nilai Grayscale (5x5 pertama):")
print(gray_values)

# 6. Terapkan filter Sobel
sobel_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)

# 7. Hitung magnitudo gradien
grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# 8. Tampilkan nilai M (5x5 pertama)
print("Nilai M (Magnitudo Gradien - 5x5 pertama):")
print(grad_magnitude[:5, :5])

# 9. Thresholding
thresholded_image = (grad_magnitude >= 1).astype(np.uint8)  # Semua M selalu >= 0

# 10. Tampilkan hasil
plt.subplot(2, 3, 2)
plt.imshow(img_resized, cmap='gray')
plt.title('Gambar 32x32 (Grayscale)')

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
