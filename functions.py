import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy.matlib as npm


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


def perdecomp(u):
    """
    Decomposes u into its periodic and smooth components
    :param u: 2D array (image)
    :return: two 2D arrays: periodic and smooth components
    """
    if u.dtype != 'float64':
        w = u.astype(np.float)
        return perdecomp(w)

    ny, nx = u.shape
    X, Y = np.arange(nx), np.arange(ny)
    v = np.zeros((ny, nx))
    v[0, :] = u[0, :] - u[-1, :]
    v[-1, :] = -v[0, :]
    v[:, 0] = v[:, 0] + u[:, 0] - u[:, -1]
    v[:, -1] = v[:, -1] - u[:, 0] + u[:, -1]
    fx = npm.repmat(np.cos(2 * np.pi * X.reshape((1, nx))/nx), ny, 1)
    fy = npm.repmat(np.cos(2 * np.pi * Y.reshape((ny, 1))/ny), 1, nx)
    fx[0, 0] = 0.  # avoiding /0
    f = fft2(v) * .5 / (2 - fx - fy)
    s = np.real(ifft2(f))
    p = u - s
    return p, s


def ffttrans(u, tx, ty):
    """
    Sub-pixel translation of image u by (tx, ty) using Shannon interpolation
    :param u: 2D array (image)
    :param tx: translation in x (second coordinate of u): lines
    :param ty: translation in y (first coordinate of u): columns
    :return: v, the translated image
    """
    if u.dtype != 'float64':
        w = u.astype(np.float)
        return ffttrans(w, tx, ty)

    ny, nx = u.shape
    my, mx = ny // 2, nx // 2
    # fourier-domain arguments p,q in "[-n/2,n/2]"
    p = np.arange(mx, mx + nx) % nx - mx
    p = p.astype(np.complex)
    q = np.arange(my, my + ny) % ny - my
    q = q.astype(np.complex)
    fx = np.exp(-2 * 1j * np.pi * tx / nx * p).reshape((1, nx))
    fy = np.exp(-2 * 1j * np.pi * ty / ny * q).reshape((ny, 1))
    return np.real(ifft2(fft2(u) * (fy @ fx)))
