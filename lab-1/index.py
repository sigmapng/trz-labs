import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

images_dir = os.path.join(current_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print(f"Створено папку: {images_dir}")
    print("Будь ласка, помістіть зображення 'sample.png' у цю папку і запустіть скрипт знову.")
    exit()

image_path = os.path.join(images_dir, "sample.png")
if not os.path.exists(image_path):
    print(f"Помилка: зображення не знайдено за шляхом {image_path}")
    print("Будь ласка, помістіть файл 'sample.png' у папку 'images'.")
    exit()

output_dir = os.path.join(current_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Створено папку для результатів: {output_dir}")

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"Помилка: не вдалося зчитати зображення з {image_path}")
    exit()

print(f"Розміри зображення: {img.shape}")
height, width = img.shape[:2]
channels = img.shape[2] if len(img.shape) > 2 else 1
print(f"Ширина: {width}, Висота: {height}, Канали: {channels}")

image_size_1 = width * height * channels
print(f"Об'єм зображення (спосіб 1): {image_size_1} байт")

image_size_2 = img.size * img.itemsize
print(f"Об'єм зображення (спосіб 2): {image_size_2} байт")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Оригінальне зображення")
plt.axis('off')
plt.show()

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title("Звичайне сіре зображення")
axes[0].axis('off')

min_val = np.min(gray_image)
max_val = np.max(gray_image)
mid_val = (min_val + max_val) // 2

vmin_val = mid_val - 30
vmax_val = mid_val + 30
axes[1].imshow(gray_image, cmap='gray', vmin=vmin_val, vmax=vmax_val)
axes[1].set_title(f"Сіре з масштабуванням (vmin={vmin_val}, vmax={vmax_val})")
axes[1].axis('off')

plt.tight_layout()
plt.show()

quality_levels = [75, 30, 0]
for quality in quality_levels:
    output_path = os.path.join(output_dir, f"image_quality_{quality}.jpg")
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Збережено зображення з якістю {quality}: {output_path}")
    print(f"Розмір файлу: {os.path.getsize(output_path)} байт")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, quality in enumerate(quality_levels):
    jpeg_path = os.path.join(output_dir, f"image_quality_{quality}.jpg")
    jpeg_img = cv2.imread(jpeg_path)
    axes[i].imshow(cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"Якість {quality}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"Завершено. Перевірте збережені зображення в папці: {output_dir}")
