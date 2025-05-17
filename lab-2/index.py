import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = r'C:\Users\riabk\Documents\TRZ_labs\lab-2\output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def gamma_correction(img, gamma, in_range=(0, 1), out_range=(0, 1)):
    """
    Застосовує гамма-корекцію до зображення.

    Args:
        img (numpy.ndarray): Вхідне зображення.
        gamma (float): Коефіцієнт гамма-корекції.
        in_range (tuple): Вхідний діапазон (min, max) для значень пікселів.
        out_range (tuple): Вихідний діапазон (min, max) для значень пікселів.

    Returns:
        numpy.ndarray: Зображення з гамма-корекцією.
    """
    img_float = img.astype(np.float32) / 255.0
    in_min, in_max = in_range
    out_min, out_max = out_range

    output = ((img_float - in_min) / (in_max - in_min)
              ) ** gamma * (out_max - out_min) + out_min

    output = np.clip(output, out_min, out_max)

    output = (output * 255).astype(np.uint8)
    return output


def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """
    Регулює яскравість та контрастність зображення.

    Args:
        img (numpy.ndarray): Вхідне зображення.
        brightness (int): Значення яскравості (від -255 до 255).
        contrast (int): Значення контрастності (від -127 до 127).

    Returns:
        numpy.ndarray: Зображення з відрегульованою яскравістю та контрастністю.
    """
    brightness = int(brightness)
    contrast = int(contrast)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max_value = 255
        else:
            shadow = 0
            max_value = 255 + brightness
        alpha = (max_value - shadow) / 255
        gamma = shadow

        cal = cv2.addWeighted(img, alpha, img, 0, gamma)
    else:
        cal = img

    if contrast != 0:
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma = 127 * (1 - alpha)

        cal = cv2.addWeighted(cal, alpha, cal, 0, gamma)

    return cal


img = cv2.imread(
    'c:/Users/riabk/Documents/TRZ_labs/lab-2/images/image.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Помилка: Не вдалося відкрити або знайти зображення. Переконайтеся, що 'image.jpg' знаходиться у правильному шляху.")
    exit()

cv2.imwrite(os.path.join(output_folder, 'original_image.jpg'), img)

cv2.imshow('Оригінальне зображення', img)
cv2.waitKey(0)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title('Гістограма оригінального зображення')
plt.xlabel('Значення пікселя')
plt.ylabel('Частота')
plt.bar(np.arange(256), hist[:, 0], color='blue', width=1.0)
plt.savefig(os.path.join(output_folder, 'original_histogram.png'))
plt.show()

eq_img = cv2.equalizeHist(img)
cv2.imwrite(os.path.join(output_folder, 'equalized_image.jpg'), eq_img)

cv2.imshow('Зображення з еквалізацією', eq_img)
cv2.waitKey(0)

# Висновок про вплив еквалізації
print("Еквалізація гістограми покращує контрастність зображення, роблячи його більш чітким.")

hist_eq = cv2.calcHist([eq_img], [0], None, [256], [0, 256])
plt.figure()
plt.title('Гістограма зображення з еквалізацією')
plt.xlabel('Значення пікселя')
plt.ylabel('Частота')
# Використовуємо plt.hist
plt.hist(eq_img.ravel(), 256, [0, 256], color='blue')
plt.savefig(os.path.join(output_folder, 'equalized_histogram.png'))
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Оригінальне зображення')

plt.subplot(2, 2, 2)
plt.bar(np.arange(256), hist[:, 0], color='blue', width=1.0)
plt.title('Гістограма оригінального зображення')

plt.subplot(2, 2, 3)
plt.imshow(eq_img, cmap='gray')
plt.title('Зображення з еквалізацією')

plt.subplot(2, 2, 4)
# Використовуємо plt.hist
plt.hist(eq_img.ravel(), 256, [0, 256], color='blue')
plt.title('Гістограма зображення з еквалізацією')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'comparison.png'))
plt.show()

gamma_values = [1.0, 0.5, 2.1]
corrected_images = []
histograms = []

for gamma in gamma_values:
    corrected_img = gamma_correction(img, gamma)
    corrected_images.append(corrected_img)
    hist = cv2.calcHist([corrected_img], [0], None, [256], [0, 256])
    histograms.append(hist)

plt.figure(figsize=(18, 6))
for i in range(len(gamma_values)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(corrected_images[i], cmap='gray')
    plt.title(f'Gamma = {gamma_values[i]}')
    cv2.imwrite(os.path.join(
        output_folder, f'gamma_corrected_image_{gamma_values[i]}.jpg'), corrected_images[i])

    plt.subplot(2, 3, i + 4)
    plt.bar(np.arange(256), histograms[i][:, 0], color='blue', width=1.0)
    plt.title(f'Гістограма Gamma = {gamma_values[i]}')
    plt.savefig(os.path.join(output_folder,
                f'gamma_corrected_histogram_{gamma_values[i]}.png'))

plt.tight_layout()
plt.show()

# Аналіз впливу гамма-корекції
print("Гамма-корекція дозволяє змінювати яскравість та контрастність зображення. Значення gamma < 1 роблять зображення світлішим, а gamma > 1 - темнішим.")

in_range_values = [(0, 1), (0.2, 0.6)]
out_range_values = [(0.2, 0.6), (0, 1)]

corrected_images_ranges = []
histograms_ranges = []

for i in range(len(in_range_values)):
    corrected_img = gamma_correction(
        img, 1, in_range_values[i], out_range_values[i])
    corrected_images_ranges.append(corrected_img)
    cv2.imwrite(os.path.join(output_folder,
                f'range_corrected_image_{i}.jpg'), corrected_images_ranges[i])
    hist = cv2.calcHist([corrected_img], [0], None, [256], [0, 256])
    histograms_ranges.append(hist)

plt.figure(figsize=(12, 6))
for i in range(len(in_range_values)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(corrected_images_ranges[i], cmap='gray')
    plt.title(
        f'Вхідний діапазон = {in_range_values[i]}, Вихідний діапазон = {out_range_values[i]}')

    plt.subplot(2, 2, i + 3)
    plt.bar(np.arange(256),
            histograms_ranges[i][:, 0], color='blue', width=1.0)
    plt.title('Гістограма')
    plt.savefig(os.path.join(output_folder,
                f'range_corrected_histogram_{i}.png'))

plt.tight_layout()
plt.show()

# Аналіз впливу порогів обмеження
print("Регулювання діапазонів яскравості дозволяє виділити певні області яскравості на зображенні, покращуючи деталізацію в цих областях.")

# Застосування adjust_brightness_contrast
brightness_values = [50, -50]
contrast_values = [30, -30]

for brightness in brightness_values:
    for contrast in contrast_values:
        adjusted_img = adjust_brightness_contrast(img, brightness, contrast)
        cv2.imwrite(os.path.join(
            output_folder, f'adjusted_image_b{brightness}_c{contrast}.jpg'), adjusted_img)

        hist_adjusted = cv2.calcHist(
            [adjusted_img], [0], None, [256], [0, 256])
        plt.figure()
        plt.title(
            f'Гістограма зображення (Яскравість={brightness}, Контрастність={contrast})')
        plt.xlabel('Значення пікселя')
        plt.ylabel('Частота')
        plt.hist(adjusted_img.ravel(), 256, [0, 256], color='blue')
        plt.savefig(os.path.join(output_folder,
                    f'adjusted_histogram_b{brightness}_c{contrast}.png'))
        plt.show()

        cv2.imshow(
            f'Зображення (Яскравість={brightness}, Контрастність={contrast})', adjusted_img)
        cv2.waitKey(0)

# Висновок про вплив регулювання яскравості та контрастності
print("Регулювання яскравості та контрастності дозволяє змінювати загальний тон зображення та різницю між світлими та темними ділянками.")

cv2.destroyAllWindows()
