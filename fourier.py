import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def fftzoom(u, z):
    """
    Shannon's interpolation zoom
    :param u: 2D array (image)
    :param z: zoom factor
    :return: a zoom with Shannon interpolation of factor z
    """
    ny, nx = u.shape
    my, mx = int(z*ny), int(z*nx)  # output shape
    # the central sub-square in Fourier space will be at indices [dy:dy+ny, dx:dx+nx]
    dy, dx = abs(ny // 2 - my // 2), abs(nx // 2 - mx // 2)

    if z >= 1:  # zoom in (super-resolution) by zero-padding
        f = np.zeros((my, mx), dtype=complex)  # future FT of the result: larger image with FTu at the centre
        f[dy:dy+ny, dx:dx+nx] = z**2 * fftshift(fft2(u))

    else:  # zoom out (sub-sampling) by spectral cutting
        # FT of the result: truncate the spectrum
        f = z**2 * fftshift(fft2(u))[dy:dy+my, dx:dx+mx]
        if mx % 2 == 0:  # cut non-Shannon frequencies in x
            f[:, 0] = 0
        if my % 2 == 0:  # cut non-Shannon frequencies in y
            f[0, :] = 0

    return np.real(ifft2(ifftshift(f)))

