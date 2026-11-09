"""
Встраивание ЦВЗ методом фазового кодирования (5.4.2).

Идея метода: каждый блок 8x8 канала B переводится в частотную область (2D-ДПФ).
Каждый коэффициент ДПФ имеет амплитуду A и фазу phi. Бит ЦВЗ записывается в ФАЗУ:
    бит 0 -> phi = +pi/2,   бит 1 -> phi = -pi/2.
Амплитуда (то есть "сила" частоты) почти не меняется, поэтому изображение
визуально остаётся прежним.
"""
import os
import numpy as np
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ===== пути =====
container_path = 'solid.bmp'
watermark_path = 'w2.bmp'
output_path    = 'results2/filled_solid.bmp'

# ===== параметры алгоритма (ключ — должен совпадать с extract.py) =====
N     = 8                                   # размер блока
COEFS = [(1, 2), (2, 1), (1, 3), (3, 1)]    # коэффициенты ДПФ: 1 бит в каждый -> 4 бита на блок
A_MIN = 30.0                                # минимальная амплитуда (для однотонных участков)
CH    = 2                                   # канал: 0-R, 1-G, 2-B


def prepare_watermark(path, capacity):
    """ЦВЗ -> чёрно-белая матрица 0/1. Белые поля обрезаются, при нехватке места ЦВЗ уменьшается."""
    img = Image.open(path).convert('L')
    a = np.array(img)
    ys, xs = np.where(a < 128)                           # где есть чёрные штрихи
    if len(xs):
        pad = 10
        img = img.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                        min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0])))
    w, h = img.size
    max_bits = capacity - 32                             # 32 бита уходят на заголовок
    if w * h > max_bits:                                 # уменьшаем с сохранением пропорций
        k = np.sqrt(max_bits / (w * h)) * 0.95
        w, h = int(w * k), int(h * k)
        img = img.resize((w, h), Image.LANCZOS)
    return (np.array(img) > 127).astype(np.uint8)       # 1 - белый, 0 - чёрный


def to_bits(wm):
    """Заголовок (16 бит ширина + 16 бит высота) + пиксели ЦВЗ построчно."""
    h, w = wm.shape
    header = [int(b) for b in f'{w:016b}{h:016b}']
    return np.concatenate([np.array(header, np.uint8), wm.reshape(-1)])


def embed(container, bits):
    img = container.astype(np.float64)
    ch = img[:, :, CH]
    H, W = ch.shape
    k = 0
    for by in range(H // N):                             # обход блоков слева направо, сверху вниз
        for bx in range(W // N):
            if k >= len(bits):
                break
            y, x = by * N, bx * N
            F = np.fft.fft2(ch[y:y + N, x:x + N])       # 1) прямое 2D-ДПФ блока
            for (u, v) in COEFS:
                if k >= len(bits):
                    break
                A = max(abs(F[u, v]), A_MIN)             # 2) амплитуда (не меньше A_MIN)
                phi = np.pi / 2 if bits[k] == 0 else -np.pi / 2   # 3) фаза = бит
                F[u, v] = A * np.exp(1j * phi)
                F[-u % N, -v % N] = A * np.exp(-1j * phi)  # сопряжённый коэффициент
                k += 1
            ch[y:y + N, x:x + N] = np.real(np.fft.ifft2(F))  # 4) обратное ДПФ
    img[:, :, CH] = ch
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


if __name__ == '__main__':
    container = np.array(Image.open(container_path).convert('RGB'))
    H, W = container.shape[:2]
    capacity = (H // N) * (W // N) * len(COEFS)

    wm = prepare_watermark(watermark_path, capacity)
    bits = to_bits(wm)
    filled = embed(container, bits)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(filled).save(output_path)
    Image.fromarray(wm * 255).save('results/watermark_used.bmp')   # для сравнения при извлечении

    diff = np.abs(filled.astype(int) - container.astype(int))
    mse = np.mean(diff ** 2)
    print(f'Контейнер: {container_path}  {container.shape}')
    print(f'Ёмкость контейнера: {capacity} бит ({len(COEFS)} бита на блок {N}x{N})')
    print(f'ЦВЗ: {wm.shape[1]}x{wm.shape[0]} = {wm.size} бит')
    print(f'Всего бит к встраиванию (с заголовком): {len(bits)}')
    print(f'PSNR = {10 * np.log10(255 ** 2 / mse):.2f} дБ')
    print(f'Сохранено: {output_path}')
