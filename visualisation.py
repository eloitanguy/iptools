from PIL import Image
import numpy as np


def normalise(array):
    a, b = np.amin(array), np.amax(array)
    return 255 * (array - a) / (b - a)


def imshow(u, norm=False):
    a = normalise(u) if norm else u
    img = Image.fromarray(np.uint8(a), 'L')
    img.show()


def fft_show(u):
    f = np.fft.fftshift(np.fft.fft2(u))
    a = np.angle(f)
    left = normalise(np.log(1 + np.abs(f)))
    right = normalise(a)
    v = np.concatenate([left, right], axis=0)
    imshow(v)


def save_comparison_gif(u, v, file_name, norm=False):
    """
    Saves a GIF that flickers between the two images
    """
    if file_name[-4:] != '.gif':
        print("Please input a valid file_name ___.gif, {} is not valid".format(file_name))

    un, vn = (normalise(u), normalise(v)) if norm else (u, v)
    images = Image.fromarray(np.uint8(un), 'L'), Image.fromarray(np.uint8(vn), 'L')
    images[0].save(file_name,
                   save_all=True,
                   append_images=images[1:],
                   duration=1000,
                   loop=0)
