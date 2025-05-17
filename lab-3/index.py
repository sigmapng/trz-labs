import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.ndimage import rank_filter
from adaptive_median_filter import adaptive_median_filter
import os

output_folder = os.path.join("lab-3", "output")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_path = 'C:/Users/riabk/Documents/TRZ_labs/lab-3/images/image.jpg'
if not os.path.exists(image_path):
    print(f"Помилка: Файл {image_path} не існує!")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Помилка: Не вдалося відкрити або знайти зображення.")
    exit()

plt.gray()
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title('Оригінальне зображення')
plt.savefig(os.path.join(output_folder, 'original.jpg'))
plt.show()

img_gaussian = random_noise(img, mode='gaussian', var=0.001)
img_gaussian = (255*img_gaussian).astype(np.uint8)

plt.figure(figsize=(8, 6))
plt.imshow(img_gaussian)
plt.title('Зображення з гаусівським шумом')
plt.savefig(os.path.join(output_folder, 'gaussian_noise.jpg'))
plt.show()

img_median = cv2.medianBlur(img_gaussian, 3)

print("Починаємо адаптивну медіанну фільтрацію...")
Smax = 7
img_adaptive_median = adaptive_median_filter(img_gaussian, Smax)
print("Адаптивну медіанну фільтрацію завершено!")

plt.figure(figsize=(12, 6))

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Оригінал')

plt.subplot(142)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Гаусівський шум')

plt.subplot(143)
plt.imshow(img_median, cmap='gray')
plt.title('Медіанний фільтр')

plt.subplot(144)
plt.imshow(img_adaptive_median, cmap='gray')
plt.title('Адаптивний медіанний фільтр')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'median_adaptive_comparison.jpg'))
plt.show()

size = 3
img_min = rank_filter(img_gaussian, rank=0, size=size)
img_max = rank_filter(img_gaussian, rank=8, size=size)

plt.figure(figsize=(12, 6))

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Оригінал')

plt.subplot(142)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Гаусівський шум')

plt.subplot(143)
plt.imshow(img_min, cmap='gray')
plt.title('Мінімальний фільтр')

plt.subplot(144)
plt.imshow(img_max, cmap='gray')
plt.title('Максимальний фільтр')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'min_max_comparison.jpg'))
plt.show()

img_blur_3 = cv2.blur(img_gaussian, (3, 3))
img_blur_7 = cv2.blur(img_gaussian, (7, 7))

plt.figure(figsize=(12, 6))

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Оригінал')

plt.subplot(142)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Гаусівський шум')

plt.subplot(143)
plt.imshow(img_blur_3, cmap='gray')
plt.title('Розмиття 3x3')

plt.subplot(144)
plt.imshow(img_blur_7, cmap='gray')
plt.title('Розмиття 7x7')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'blur_comparison.jpg'))
plt.show()
