"""
Извлечение ЦВЗ, встроенного методом фазового кодирования (5.4.2).
Для каждого блока 8x8 считается 2D-ДПФ и смотрится ЗНАК фазы нужных коэффициентов:
    фаза > 0 -> бит 0,   фаза <= 0 -> бит 1.
Исходный (пустой) контейнер не нужен — достаточно ключа: N, COEFS, CH.
"""
import os
import numpy as np
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===== пути =====
filled_path = 'results2/filled_solid.bmp'
output_path = 'results2/extracted_solid.bmp'

# ===== ключ (как в embed.py) =====
N     = 8
COEFS = [(1, 2), (2, 1), (1, 3), (3, 1)]
CH    = 2


def read_bits(ch, count):
    H, W = ch.shape
    bits = []
    for by in range(H // N):                     # тот же порядок обхода, что при встраивании
        for bx in range(W // N):
            y, x = by * N, bx * N
            F = np.fft.fft2(ch[y:y + N, x:x + N])
            for (u, v) in COEFS:
                if len(bits) == count:
                    return np.array(bits, np.uint8)
                bits.append(0 if np.angle(F[u, v]) > 0 else 1)
    return np.array(bits, np.uint8)


if __name__ == '__main__':
    img = np.array(Image.open(filled_path).convert('RGB'))
    ch = img[:, :, CH].astype(np.float64)

    header = read_bits(ch, 32)                   # первые 32 бита — размеры ЦВЗ
    w = int(''.join(map(str, header[:16])), 2)
    h = int(''.join(map(str, header[16:])), 2)
    print(f'Размер ЦВЗ из заголовка: {w} x {h}')

    bits = read_bits(ch, 32 + w * h)[32:]
    wm = bits.reshape(h, w)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(wm * 255).save(output_path)
    print(f'Сохранено: {output_path}')

    if os.path.exists('results/watermark_used.bmp'):   # сравнение с тем, что встраивали
        orig = (np.array(Image.open('results/watermark_used.bmp')) > 127).astype(np.uint8)
        if orig.shape == wm.shape:
            print(f'Ошибочных бит (BER): {np.mean(orig != wm) * 100:.2f}%')
