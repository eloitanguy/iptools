from PIL import Image
import numpy as np


def normalise(array):
    a, b = np.amin(array), np.amax(array)
    return 255 * (array - a) / (b - a)


def imshow(u, norm=True):
    a = normalise(u) if norm else u
    img = Image.fromarray(np.uint8(a), 'L')
    img.show()


def show_fft(u):
    f = np.fft.fftshift(np.fft.fft2(u))
    a = np.angle(f)
    left = normalise(np.log(1+np.abs(f)))
    right = normalise(a)
    v = np.concatenate([left, right], axis=0)
    imshow(v)
