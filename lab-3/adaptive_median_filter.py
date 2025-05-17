import numpy as np


def adaptive_median_filter(img, Smax=7):
    """
    Реалізація адаптивного медіанного фільтра.
    img — вхідне зображення (grayscale, uint8)
    Smax — максимальний розмір маски (непарне число >= 3)
    """

    padded_img = np.pad(img, Smax // 2, mode='reflect')
    result = np.zeros_like(img)

    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            S = 3  # початковий розмір маски

            while S <= Smax:
                r = i + Smax // 2
                c = j + Smax // 2
                window = padded_img[r - S // 2:r + S // 2 + 1,
                                    c - S // 2:c + S // 2 + 1]

                Zmed = int(np.median(window))
                Zmin = int(window.min())
                Zmax = int(window.max())
                Zxy = int(img[i, j])
                A1 = Zmed - Zmin
                A2 = Zmed - Zmax

                if A1 > 0 and A2 < 0:
                    B1 = Zxy - Zmin
                    B2 = Zxy - Zmax
                    if B1 > 0 and B2 < 0:
                        # залишаємо без змін
                        result[i, j] = np.clip(Zxy, 0, 255)
                    else:
                        # замінюємо на медіану
                        result[i, j] = np.clip(Zmed, 0, 255)
                    break
                else:
                    S += 2  # збільшуємо маску
            else:
                # якщо маска перевищила Smax
                result[i, j] = np.clip(Zmed, 0, 255)

    return result.astype(np.uint8)
