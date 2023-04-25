import numpy as np
import scipy
from numpy import pi
import matplotlib.pyplot as plt
import scipy.special as spsp
from scipy.optimize import fsolve, root
import scipy.integrate as itg
from numpy.linalg import inv


def add_arrays(sz1, sz2):
    return tuple([sz1[i] + sz2[i] - 1 for i in range(len(sz1))])


def kronSum(mat1, mat2, fac=1 + 1e-4):
    mat1_ = np.power(fac, mat1)
    mat2_ = np.power(fac, mat2)
    mat3_ = np.kron(mat1_, mat2_)
    mat3 = np.emath.logn(fac, mat3_)
    mat3R = np.real(mat3)
    mat3I = np.imag(mat3)
    # mat3R = np.around(mat3R, 1)
    # mat3I = np.around(mat3I, 1)
    mat3 = mat3R + 1j * mat3I
    return mat3


def conv_beam(Arr1, Arr2):
    return vect(scipy.signal.convolve2d(ivect(Arr1.h, Arr1.My), ivect(Arr2.h, Arr2.My)))


def vect(mat):
    return mat.flatten('F').reshape(-1, 1)


def ivect(mat, M):
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
    omega = 2 * pi * f
    D1_m = Rm * np.cos(t - Psim)
    T1_m = D1_m / c
    d1_m = np.exp(-1j * omega * T1_m)
    return d1_m


def he(mat):
    mat = mat.T
    mat = np.conj(mat)
    return mat


def calcbeam(h, d):
    A = he(h) @ d
    A = A.item()
    return A


def calcGain(h, d):
    return dB(calcbeam(h, d))


def calcwng(h, d):
    A = np.abs(he(h) @ d) ** 2 / (he(h) @ h)
    A = np.real(A).item()
    # A = A.astype(np.float16)
    return A


def calcsnr(Arr, varx, ti, varis, vard, vara, f, c):
    CorrV = np.zeros([Arr.M, Arr.M], dtype=complex)
    for idx, t in enumerate(ti):
        d_ti = Arr.calc_sv(t, f, c, False)
        CorrV += varis[idx] * d_ti @ he(d_ti)
    CorrV += np.identity(Arr.M) * vard
    CorrV += np.ones_like(CorrV) * vara
    # print(fixDec(np.abs(CorrV)))
    # input()
    h = Arr.h
    d = Arr.d_td
    A = (varx * np.abs(he(h) @ d) ** 2 / (he(h) @ CorrV @ h)) / (varx / CorrV[0, 0])
    A = np.real(A).item()
    return A


def calcfnbw(beam, idx_td, angles):
    # print(beam[idx_td-20:idx_td+20])
    # input()
    beam = np.abs(beam)
    idx_p = idx_td
    idx_m = idx_td
    idx = -1
    while True:
        # print(beam.size)
        # print(angles.size)
        # input()
        if idx_p >= angles.size or idx_m <= 0:
            break
        if beam[idx_p-1] > beam[idx_p] and beam[idx_p+1] > beam[idx_p]:
            idx = idx_p
            break
        if beam[idx_m-1] > beam[idx_m] and beam[idx_m+1] > beam[idx_m]:
            idx = idx_m
            break
        idx_p += 1
        idx_m -= 1
    return abs(angles[idx])


def calcdsdi(h, d):
    A = np.abs(he(h) @ d - 1) ** 2
    A = np.real(A).item()
    # A = A.astype(np.float16)
    return A


def calcbpt(h, d):
    A = calcbeam(h, d)
    return A


def calcGzp(Arr, f, c, vals):
    def integrand(t, ri, rj, pi, pj, f, c):
        di_t = calcSV(ri, pi, t, f, c).item()
        dj_t = calcSV(rj, pj, t, f, c).item()
        a = di_t * np.conj(dj_t) * np.sin(t)
        # a = np.exp(-1j*2*np.pi*f/c*(ri * np.cos(t - pi) - rj * np.cos(t - pj))) * np.sin(t)
        return np.real(a)

    M = Arr.M
    Gzp = np.zeros([M, M], dtype=complex)
    for i in range(M):
        for j in range(M):
            Posi = Arr.Pos[i, 0]
            Posj = Arr.Pos[j, 0]
            dij = abs(Posi-Posj)
            Gzp[i, j] = spsp.j0(2*pi*f*dij/c)
    return Gzp, vals


def calcdf(Arr, f, c, vals):
    Gzp, vals = calcGzp(Arr, f, c, vals)
    h = Arr.h
    d_td = Arr.d_td

    df = np.abs(calcbeam(h, d_td)) ** 2 / (he(h) @ Gzp @ h)
    df = df.item()
    df = np.real(df)
    return df, vals


def show_array(pos, sz, label):
    x = np.real(pos)
    y = np.imag(pos)
    plt.scatter(y, x, s=sz, label=label, linewidths=0)


def ang_f2s(ref, angs):  # Angle - Front to side
    nangs = list(angs)
    for i in range(len(nangs)):
        nangs[i] = ref + angs[i]
    nangs = tuple(nangs)
    return nangs


def cart2pol(A):
    Ar = np.abs(A)
    Ap = np.angle(A)
    Ap[Ap < 0] = 2 * pi + Ap[Ap < 0]
    return Ar, Ap


def pol2cart(Ar, Ap):
    A = Ar * np.exp(1j * Ap)
    return A


def dB(A):
    # print(type(A))
    if type(A) in [list, tuple, set]:
        B = list(A)
    else:
        B = [A]
    C = []
    for b in B:
        c = 10 * np.log10(np.abs(b))
        C.append(c)
    if type(A) in [list, tuple, set]:
        C = tuple(C)
    else:
        C = C[0]
    # print(type(C))
    return C


class Sensor:
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
    def __init__(self, M, name):
        My, Mx = M
        self.name = name
        self.Mx = Mx
        self.My = My
        self.M = Mx*My
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

    def calc_vals(self, params):
        # Position - Cartesian
        Mx = self.Mx
        My = self.My
        Pos_ = np.zeros([My, Mx], dtype=complex)
        for p in range(My):
            for q in range(Mx):
                sensor = Sensor(q, p, params)
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
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

