import numpy as np
import scipy
from numpy import pi
import matplotlib.pyplot as plt
import scipy.special as spsp
from scipy.optimize import fsolve, root
from scipy.signal import convolve2d as c2d
import scipy.integrate as itg
from numpy.linalg import inv
from colorsys import hsv_to_rgb


def add_arrays(sz1, sz2):
    # Joins two sensor arrays' sizes, for KP [1] or CKP [else]
    if sz1[2] == 1 or sz2[2] == 1:
        lst = [sz1[i] * sz2[i] for i in range(2)]
        lst.append(1)
    else:
        lst = [sz1[i] + sz2[i] - 1 for i in range(2)]
        lst.append(0)

    return lst


def kronSum(mat1, mat2, fac=1 + 1e-4):
    # Kronecker sum operation
    mat1_ = np.power(fac, mat1)
    mat2_ = np.power(fac, mat2)
    mat3_ = np.kron(mat1_, mat2_)
    mat3 = np.emath.logn(fac, mat3_)
    mat3R = np.real(mat3)
    mat3I = np.imag(mat3)
    mat3 = mat3R + 1j * mat3I

    return mat3


def conv_beam_sub(h1, h2, My1, My2, scale_):
    # Convolves beampatterns, for CKP [0] or KP [else]
    if scale_ == 0:
        return vect(c2d(ivect(h1, My1), ivect(h2, My2)))
    else:
        return vect(np.kron(ivect(h1, My1), ivect(h2, My2)))


def conv_beam(Arr1, Arr2, mode='h', scale=0):
    # Convolves two arrays' beampatterns
    if mode == 'h':
        return conv_beam_sub(Arr1.h, Arr2.h, Arr1.My, Arr2.My, max(scale, Arr1.scale, Arr2.scale))
    else:
        return conv_beam_sub(Arr1.d_td, Arr2.d_td, Arr1.My, Arr2.My, max(scale, Arr1.scale, Arr2.scale))


def vect(mat):
    # Vectorizes a matrix
    return mat.flatten('F').reshape(-1, 1)


def ivect(mat, M):
    # Inverse vectorization (2D)
    omat = np.zeros([M, mat.size//M], dtype=complex)
    if mat.size//M == 1:
        omat = mat
    else:
        for i in range(mat.size//M):
            omat[:, i] = mat[i*M:(i+1)*M].reshape(-1,)
    return omat


def err(mat1, mat2, dec=2):
    mat_err = mat1 - mat2
    mat_err = fixDec(mat_err, dec)
    return mat_err


def fixDec(in_mat, dec=2):
    # Fixes decimal places
    if type(in_mat) not in (list, tuple, set):
        in_mat = [in_mat]
    mats = list(in_mat)
    for idx in range(len(mats)):
        mat = mats[idx]
        # fac = 10 ** dec
        # mat = (1 / fac) * np.around(mat * fac)
        mat = np.around(mat, decimals=dec)
        mats[idx] = mat
    mats = tuple(mats)
    return mats


def calcSV(Rm, Psim, t, f, c):
    # Calculates steering vector (SV)
    omega = 2 * pi * f
    D1_m = Rm * np.cos(t - Psim)
    T1_m = D1_m / c
    d1_m = np.exp(-1j * omega * T1_m)
    return d1_m


def he(mat):
    # Hermitian of a matrix
    mat = mat.T
    mat = np.conj(mat)
    return mat


def calcbeam(h, d):
    # Calculates beampattern
    A = he(h) @ d
    A = A.item()
    return A


def calcGain(h, d):
    # Calculates gain

    return dB(calcbeam(h, d))


def calcwng(h, d):
    # Calculates white noise gain
    A = np.abs(he(h) @ d) ** 2 / (he(h) @ h)
    A = np.real(A).item()

    return A


def calcsnr(Arr, varx, ti, varis, vard, vara, f, c):
    # Calculates SNR
    CorrV = np.zeros([Arr.M, Arr.M], dtype=complex)
    for idx, t in enumerate(ti):
        d_ti = Arr.calc_sv(t, f, c, False)
        CorrV += varis[idx] * d_ti @ he(d_ti)
    CorrV += np.identity(Arr.M) * vard
    CorrV += np.ones_like(CorrV) * vara

    h = Arr.h
    d = Arr.d_td
    A = (varx * np.abs(he(h) @ d) ** 2 / (he(h) @ CorrV @ h)) / (varx / CorrV[0, 0])
    A = np.real(A).item()

    return A


def calcfnbw(beam, idx_td, angles):
    # Calculates first-null beamwidth
    beam = np.abs(beam)
    idx_p = idx_td
    idx_m = idx_td
    idx = -1
    while True:
        if idx_p >= angles.size or idx_m <= 0:
            break
        if beam[idx_p] < beam[idx_p-1] and beam[idx_p] < beam[idx_p+1] and beam[idx_p] < 0.05:
            idx = idx_p
            break
        if beam[idx_m] < beam[idx_m-1] and beam[idx_m] < beam[idx_m+1] and beam[idx_m] < 0.05:
            idx = idx_m
            break
        idx_p += 1
        idx_m -= 1

    return abs(angles[idx])  # * 2


def calcdsdi(h, d):
    # Calculates desired signal distortion index
    A = np.abs(he(h) @ d - 1) ** 2
    A = np.real(A).item()

    return A


def calcbpt(h, d):
    # Alias for calculating beampattern (why?)
    A = calcbeam(h, d)

    return A


def calcGzp(Arr, f, c, vals, epsilon=0):
    # Calculates Gamma matrix from 0 to pi (zp)
    M = Arr.M
    Gzp = np.zeros([M, M], dtype=complex)
    for i in range(M):
        for j in range(M):
            Posi = Arr.Pos[i, 0]
            Posj = Arr.Pos[j, 0]
            dij = abs(Posi-Posj)
            Gzp[i, j] = spsp.sinc(2*pi*f*dij/c * 1/pi)

    Gzp = Gzp + np.identity(M)*epsilon

    return Gzp, vals


def calcdf(Arr, f, c, vals):
    # Calculates directivity factor
    Gzp, vals = calcGzp(Arr, f, c, vals)
    h = Arr.h
    d_td = Arr.d_td

    df = np.abs(calcbeam(h, d_td)) ** 2 / (he(h) @ Gzp @ h)
    df = df.item()
    df = np.real(df)

    return df, vals


def show_array(pos, sz, label):
    # Shows array of sensors
    x = np.real(pos)
    y = np.imag(pos)
    plt.scatter(y, x, s=sz, label=label, linewidths=0)


def ang_f2s(ref, angs):
    # Converts angle from front (reference: desired = 0) to side (reference: desired != 0)
    nangs = list(angs)
    for i in range(len(nangs)):
        nangs[i] = ref + angs[i]
    nangs = tuple(nangs)

    return nangs


def cart2pol(A):
    # Converts cartesian to polar
    Ar = np.abs(A)
    Ap = np.angle(A)
    Ap[Ap < 0] = 2 * pi + Ap[Ap < 0]

    return Ar, Ap


def pol2cart(Ar, Ap):
    # Converts polar to cartesian
    A = Ar * np.exp(1j * Ap)

    return A


def dB(A):
    # Calculates value in dB
    if type(A) in [list, tuple, set]:
        B = list(A)
    else:
        B = [A]
    C = []
    for b in B:
        c = 10 * np.log10(np.abs(b) + 1e-6)
        C.append(c)
    if type(A) in [list, tuple, set]:
        C = tuple(C)
    else:
        C = C[0]

    return C


class Sensor:
    # Class Sensor
    def __init__(self, mx, my, params):
        self.mx = mx
        self.my = my

        self.Pos = None
        self.r = None
        self.p = None
        self.d_td = None
        self.calc_pos(params.dx, params.dy)
        self.calc_rp()

    def calc_pos(self, dx, dy, P00=0):
        mx = self.mx
        my = self.my

        Pos = P00 + mx*dx + 1j*my*dy

        self.Pos = Pos

        return self.Pos

    def calc_rp(self):
        Pos = self.Pos
        r = np.abs(Pos)
        p = np.angle(Pos)
        self.r = r
        self.p = p
        return r, p

    def calc_sv(self, td, f, c, is_td):
        d_td = calcSV(self.r, self.p, td, f, c)
        if is_td:
            self.d_td = d_td
        return d_td


class Array:
    # Class Array (of sensors)
    def __init__(self, M, name):
        My, Mx, scale = M
        self.name = name
        self.Mx = Mx
        self.My = My
        self.M = Mx*My
        self.scale = scale
        self.Pos_ = None
        self.Pos = None

        self.r = None
        self.p = None

        self.h = None
        self.d_td = None
        self.g_td = None
        self.beam = None
        self.wng = None
        self.dsdi = None
        self.bpt = None
        self.df = None
        self.snr = None
        self.fnbw = None

    def calc_vals(self, params, Arr_PA):
        # Position - Cartesian
        Mx = self.Mx
        My = self.My
        PA_Mx = Arr_PA.Mx
        PA_My = Arr_PA.My
        Pos_ = np.zeros([My, Mx], dtype=complex)
        if self.scale == 1:
            facx = PA_Mx // Mx
            facy = PA_My // My
        else:
            facx = 1
            facy = 1
        for p in range(My):
            for q in range(Mx):
                sensor = Sensor(facx*q, facy*p, params)
                Pos_[p, q] = sensor.Pos

        self.Pos_ = Pos_
        self.Pos = vect(Pos_)

        # Position - Polar
        Pos = self.Pos
        r = np.abs(Pos)
        p = np.angle(Pos)
        self.r = r
        self.p = p

    def calc_sv(self, t, f, c, is_td):
        d_td = calcSV(self.r, self.p, t, f, c)
        if is_td:
            self.d_td = d_td
        return d_td

    def calc_gain(self):
        g_td = calcGain(self.h, self.d_td)
        self.g_td = g_td
        return g_td

    def init_metrics(self, len_f, len_a, Ni):
        self.beam = np.zeros([len_f, len_a], dtype=complex)
        self.wng = np.zeros([len_f])
        self.df = np.zeros([len_f])
        self.dsdi = np.zeros([len_f])
        self.bpt = np.zeros([len_f, Ni + 1], dtype=complex)
        self.snr = np.zeros([len_f])
        self.fnbw = np.zeros([len_f])


class Params:
    # Class Params (to group various parameters)
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def add(self, **kwds):
        self.__dict__.update(kwds)


def RGB_to_HTML(RGB):
    # Converts RGB to HTML
    HTMLs = ['{:02X}'.format(v) for v in RGB]
    HTML = ''.join(HTMLs)
    return HTML


def gen_palette(S, V, ncolors, hshift=0):
    # Generates palette of colors
    letters = 'ACEBDF'
    hs = np.arange(ncolors)/ncolors
    colors = []
    for h in hs:
        rgb = hsv_to_rgb(h+hshift/360, S/100, V/100)
        RGB = tuple(round(i * 255) for i in rgb)
        HTML = RGB_to_HTML(RGB)
        colors.append(HTML)
    ncolors = [r'\definecolor{Light' + letters[i] + r'}{HTML}{' + colors[i] + r'}' for i in range(len(colors))]
    return ncolors


if __name__ == '__main__':
    colors = gen_palette(80, 60, 6, 345)
    print('\n'.join(colors))

    circles = []
    fig, ax = plt.subplots()
    for idx, col in enumerate(colors):
        c = plt.Circle((0.5+idx % 3, -0.5 - idx // 3), 0.2, color='#'+col)
        ax.add_patch(c)
    plt.xlim(0, 3)
    plt.ylim(-idx // 3, 0)
    plt.show()

