###Utility function for handling qgmodel fields
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

##file i/o
def read_data_bin(filename, kmax, nz, k, r=0):
    nkx = 2*kmax+1
    nky = kmax+1
    fieldk = np.zeros((nkx, nky), dtype=complex)
    with open(filename, 'rb') as f:
        for i in range(nky):
            f.seek(((2*r*nz+k)*nky+i)*nkx*8)
            fieldk[:, i].real = np.fromfile(f, dtype=float, count=nkx)
            f.seek(((2*r*nz+nz+k)*nky+i)*nkx*8)
            fieldk[:, i].imag = np.fromfile(f, dtype=float, count=nkx)
        return fieldk


def write_data_bin(filename, fieldk, kmax, nz, k, r=0):
    ##note: filename should exists before calling this to write content to it
    nkx = 2*kmax+1
    nky = kmax+1
    assert fieldk.shape == (nkx, nky), 'input fieldk shape mismatch with kmax'
    with open(filename, 'r+b') as f:
        for i in range(nky):
            f.seek(((2*r*nz+k)*nky+i)*nkx*8)
            f.write(fieldk[:, i].real.tobytes())
            f.seek(((2*r*nz+nz+k)*nky+i)*nkx*8)
            f.write(fieldk[:, i].imag.tobytes())


##conversion between physical and spectral spaces
def fullspec(hfield):
    nkx, nky = hfield.shape
    kmax = nky-1
    hres = nkx+1
    sfield = np.zeros((hres, hres), dtype=complex)
    fup = hfield
    fup[kmax-1::-1, 0] = np.conjugate(fup[kmax+1:nkx, 0])
    fdn = np.conjugate(fup[::-1, nky-1:0:-1])
    sfield[1:hres, nky:hres] = fup
    sfield[1:hres, 1:nky] = fdn
    return sfield


def halfspec(sfield):
    n1, n2 = sfield.shape
    kmax = int(n1/2)-1
    hfield = sfield[1:n1, kmax+1:n2]
    return hfield


def spec2grid(fieldk):
    nkx, nky = fieldk.shape
    nx = nkx+1
    ny = 2*nky
    fieldk = ifftshift(fullspec(fieldk))
    field = nx*ny*np.real(ifft2(fieldk))
    return field


def grid2spec(field):
    nx, ny = field.shape
    nkx = nx-1
    nky = int(ny/2)
    fieldk = fft2(field)/nx/ny
    fieldk = halfspec(fftshift(fieldk))
    return fieldk


##conversion between variables in spectral space
def get_wn(psik):
    nkx, nky = psik.shape
    kmax = nky-1
    kx_, ky_ = np.mgrid[-kmax:kmax+1, 0:kmax+1]
    return kx_, ky_


def psi2zeta(psik):
    kx_, ky_ = get_wn(psik)
    zetak = -(kx_**2 + ky_**2) * psik
    return zetak


def psi2temp(psik):
    kx_, ky_ = get_wn(psik)
    tempk = -np.sqrt(kx_**2 + ky_**2) * psik
    return tempk


def psi2u(psik):
    kx_, ky_ = get_wn(psik)
    uk = -1j * ky_ * psik
    return uk


def psi2v(psik):
    kx_, ky_ = get_wn(psik)
    vk = 1j * kx_ * psik
    return vk


def uv2zeta(uk, vk):
    kx_, ky_ = get_wn(uk)
    zetak = 1j*kx_*vk - 1j*ky_*uk
    return zetak


def zeta2psi(zetak):
    nkx, nky = zetak.shape
    kmax = nky-1
    kx_, ky_ = get_wn(zetak)
    k2_ = kx_**2 + ky_**2
    k2_[kmax, 0] = 1  #set irrelavent point to 1 to avoid singularity in inversion
    psik = -(1.0/k2_) * zetak
    return psik


def temp2psi(tempk):
    nkx, nky = tempk.shape
    kmax = nky-1
    kx_, ky_ = get_wn(tempk)
    k1_ = np.sqrt(kx_**2 + ky_**2)
    k1_[kmax, 0] = 1
    psik = -(1.0/k1_) * tempk
    return psik

