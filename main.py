import copy
import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv
from functions import *
import scipy.special as spsp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tikzplotlib as tpl
import scipy as sp


def simulation(result: str, sizes: Params, name: str):
    """
    Parameters
    ----------
    result: str
        Which result to produce, from the available.
    sizes: Params
        Struct of names and sizes of the desired sub-array beamformers.
    name: str
        Name of current simulation.

    Returns
    -------

    """

    # ---------------
    # Possible values for parameter 'result'

    RESULT_s = {
        'beam': 'Heatmap - Beampattern (abs)',
        'bpt': 'Lineplot - Beampattern (abs and angle)',
        'wng': 'Lineplot - White noise gain (real)',
        'df': 'Lineplot - Directivity factor (real)',
        'snr': 'Lineplot - Signal-to-noise ratio (real)',
        'polar': "Polar plot - Beampatterns for all BF's and ",
        'polar2': 'Polar plot - Beampattern for a given frequency',
        'gendata': "Multiplot - Generates 'beam' and 'polar' for the final beamformer"
    }

    # ---------------
    # Sensor array

    sizes.ss = add_arrays(sizes.sd, sizes.ds)
    sizes.CS = add_arrays(sizes.ss, sizes.cb)
    sizes.PA = add_arrays(sizes.LC, sizes.CS)

    Arr_PA = Array(sizes.PA, "PA")
    Arr_LC = Array(sizes.LC, "LC")
    Arr_CS = Array(sizes.CS, "CS")
    Arr_sd = Array(sizes.sd, "sd")
    Arr_ds = Array(sizes.ds, "ds")
    Arr_ss = Array(sizes.ss, "ss")
    Arr_cb = Array(sizes.cb, "cb")

    Arrs = [
        Arr_PA,
        Arr_LC,
        Arr_CS,
        Arr_sd,
        Arr_ds,
        Arr_ss,
        Arr_cb
    ]

    # ---------------
    # Constants

    dx = 0.015   # x-axis spacing
    dy = 0.03    # y-axis spacing
    c = 343      # wave speed (speed of sound)
    tB_deg = 40  # FNBW aperture

    ## ---------------
    ## Signal Info

    # ---------------
    # Desired source
    td_deg = 0  # desired source direction
    td_rad = np.deg2rad(td_deg)
    Var_X = 5   # variance of X

    # ---------------
    # Interfering source
    Ni = Arr_LC.M - 1  # total number of interfering sources

    all_tis_deg = [60, -90, 130]  # (un)desired LCMV angles (must be Ni <= len(all_tis_deg))
    tis_deg = []

    for f_idx in range(Ni):
        tis_deg.append(td_deg + all_tis_deg[f_idx])

    tis_deg = np.array(tis_deg)
    tis_rad = np.deg2rad(tis_deg)
    Var_Vis = np.ones_like(tis_deg)

    Var_Vd = 1  # Variance of uncorrelated noise
    Var_Va = 1  # Variance of fully correlated noise

    ## ---------------
    ## Simulation Info

    tB_rad = np.deg2rad(tB_deg)
    d_tB = ang_f2s(td_deg, (-tB_deg/2, +tB_deg/2))
    f0 = 4 * 1e3
    f1 = 8 * 1e3
    fpoints = 20
    sym_freqs = np.logspace(np.log10(f0), np.log10(f1), fpoints)

    prec_around_angle = np.linspace(-5, 5, 21)  # precision around important angles

    calc_angles = [td_deg] + list(tis_deg) + list(-tis_deg) \
                  + [ang for ang in list(d_tB[0] + prec_around_angle)] \
                  + [ang for ang in list(d_tB[1] + prec_around_angle)]

    sym_angles = [td_deg]
    step = 3
    while sym_angles[-1] - sym_angles[0] != 360:
        l = len(sym_angles)
        in0 = sym_angles[0] - step
        in1 = sym_angles[-1] + step
        sym_angles.insert(l, in1)
        sym_angles.insert(0, in0)

    a0 = sym_angles[0]
    a1 = sym_angles[-1]

    for ang in calc_angles:
        check = (np.array(sym_angles) - ang) % 360
        if not any(check == 0):
            sym_angles.append(ang)
    sym_angles = list(set(sym_angles))
    sym_angles.sort()

    sym_angles = np.sort(sym_angles)

    params = Params(dx=dx, dy=dy,
                    sym_freqs=sym_freqs, sym_angles=sym_angles,
                    td_rad=td_rad, c=c, wB_rad=tB_rad, aleph=1)

    ## ---------------
    ## Sub-array matrices
    for Arr in Arrs:
        Arr.calc_vals(params, Arr_PA)  # Calculates position for all sensors of sub-array
        Arr.init_metrics(sym_freqs.size, sym_angles.size, Ni)  # Initializes metrics

    ## ---------------
    ## Simulation loop - For each frequency
    for f_idx, f in enumerate(sym_freqs):
        params.add(f=f)

        ## ---------------
        ## LCMV beamformer

        # ---------------
        # Desired source
        Arr_LC.calc_sv(td_rad, f, c, True)
        Corr_X = Var_X * Arr_LC.d_td @ he(Arr_LC.d_td)

        # ---------------
        # Interfering source
        C1 = []
        Corr_V = np.zeros_like(Corr_X)
        for i, ti_rad in enumerate(tis_rad):
            d_ti_LC = Arr_LC.calc_sv(ti_rad, f, c, False)
            C1.append(d_ti_LC)

            Var_Vi = Var_Vis[i]
            Corr_V += Var_Vi * d_ti_LC @ he(d_ti_LC)
        Corr_Vd = Var_Vd * np.identity(Corr_V.shape[0])
        Corr_V += Corr_Vd
        Corr_Va = Var_Va * np.ones_like(Corr_V)
        Corr_V += Corr_Va
        iCorr_V = inv(Corr_V)
        iCorr_Vd = inv(Corr_Vd)

        # ---------------
        # Beamforming
        C1.insert(0, Arr_LC.d_td)
        C1 = np.concatenate(C1, axis=1)
        q1 = np.zeros([Ni + 1, 1])
        q1[0] = 1

        Arr_LC.h = iCorr_V @ C1 @ inv(he(C1) @ iCorr_V @ C1) @ q1

        ## ---------------
        ## SD beamformer

        # ---------------
        # Desired source
        sd_size = Arr_sd.M
        Arr_sd.calc_sv(td_rad, f, c, True)

        # ---------------
        # Beamforming
        G_sd, _ = calcGzp(Arr_sd, f, c, epsilon=1e-7)

        iG_sd = np.linalg.inv(G_sd)
        Arr_sd.h = (iG_sd @ Arr_sd.d_td) / (he(Arr_sd.d_td) @ iG_sd @ Arr_sd.d_td)

        ## ---------------
        ## DS beamformer

        # ---------------
        # Desired source
        ds_size = Arr_ds.M
        Arr_ds.calc_sv(td_rad, f, c, True)

        # ---------------
        # Beamforming
        Arr_ds.h = Arr_ds.d_td / ds_size

        ## ---------------
        ## CB beamformer

        cb_size = Arr_cb.M

        Arr_cb.calc_sv(td_rad, f, c, True)
        p0, p1, p2, p3, p4, p5 = (26, 12, 0.76608, 13.26, 0.4, 0.09834)
        Asl = p0 * cb_size * dy * f / c * np.sin(tB_rad/2) - p1
        beta = p2 * np.power(Asl - p3, p4, dtype=complex) + p5 * (Asl - p3)
        w = np.zeros([cb_size, 1], dtype=complex)
        for m in range(w.shape[0]):
            w[m] = spsp.iv(0, beta * np.sqrt(1 - (2 * m / (cb_size - 1) - 1) ** 2)) / spsp.iv(0, beta)

        w_ = w * 1 / np.sum(w) / np.conj(Arr_cb.d_td)
        Arr_cb.h = w_

        ## ---------------
        ## Full-array synthesis

        for Arr in Arrs:
            Arr.calc_sv(td_rad, f, c, True)

        Arr_ss.h = conv_beam(Arr_sd, Arr_ds)
        Arr_CS.h = conv_beam(Arr_ss, Arr_cb)
        Arr_PA.h = conv_beam(Arr_LC, Arr_CS)

        for Arr in Arrs:
            Arr.calc_gain()

        # Output metrics
        for arr_idx, Arr in enumerate(Arrs):
            if result in ('gendata', 'beam', 'polar', 'polar2', 'fnbw'):
                # Beampattern
                for a_idx, angle in enumerate(sym_angles):
                    angle = np.deg2rad(angle)
                    d_ta = Arr.calc_sv(angle, f, c, False)
                    Arr.beam[f_idx, a_idx] = calcbeam(Arr.h, d_ta)

            if result in ('gendata', 'dsdi'):
                # DSDI
                Arr.dsdi[f_idx] = calcdsdi(Arr.h, Arr.d_td)

            if result in ('gendata', 'wng'):
                # WNG
                Arr.wng[f_idx] = calcwng(Arr.h, Arr.d_td)

            if result in ('gendata', 'df'):
                # DF
                df = calcdf(Arr, f, c)
                Arr.df[f_idx] = df

            if result in ('gendata', 'snr'):
                # SNR
                Arr.snr[f_idx] = calcsnr(Arr, Var_X, tis_rad, Var_Vis, Var_Vd, Var_Va, f, c)

            if result in ('gendata', 'fnbw'):
                # FNBW
                idx_td = np.where(sym_angles == td_deg)[0][0]
                Arr.fnbw[f_idx] = calcfnbw(Arr.beam[f_idx, :], idx_td, sym_angles)

            if result in ('gendata', 'bpt'):
                # BPT
                bpt = calcbpt(Arr.h, Arr.d_td)
                bpt = [bpt]
                for ti_rad in tis_rad:
                    d_ti = Arr.calc_sv(ti_rad, f, c, False)
                    bpt.append(calcbpt(Arr.h, d_ti))
                Arr.bpt[f_idx, :] = np.array(bpt)
                Arrs[arr_idx] = Arr

    XY_f, XY_a = np.meshgrid(sym_angles, sym_freqs)

    match result:
        case 'beam':
            # fix, ax = plt.subplots(2, 2)
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(5, 5)
            ax1 = plt.subplot(gs[0:2, 0:2])
            ax2 = plt.subplot(gs[0:2, 3:5])
            ax3 = plt.subplot(gs[3:5, 1:4])

            axs = [ax1, ax2, ax3]
            beams = [Arr_LC.beam, Arr_CS.beam, Arr_PA.beam]
            xticks = [a0 + i * 45 for i in range(int(1 + (a1 - a0) // 45))]
            yticks = [f0, (f0 + f1) / 2, f1]

            for idx in range(len(axs)):
                ax = axs[idx]
                beam = beams[idx]
                dB_beam = dB(beam)
                c = ax.pcolormesh(XY_f, XY_a, dB_beam, cmap='viridis', vmin=-50)
                ax.set_xticks(ticks=xticks)
                ax.set_yticks(ticks=yticks, labels=['{:.1f}k'.format(ytick / 1000) for ytick in yticks])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(c, cax=cax, orientation='vertical')
                ax.axvline(x=td_deg, c='y', linestyle=':')
                ax.axvline(x=d_tB[0], c='b', linestyle=':')
                ax.axvline(x=d_tB[1], c='b', linestyle=':')
                ax.set_xlabel('Angle (º)')
                ax.set_ylabel('Frequency (Hz)')
                c.set_clim(-60, 0)
            plt.show()

        case 'wng':
            fig, ax = plt.subplots()
            for arr_idx, Arr in enumerate(Arrs):
                wng_dB = dB(Arr.wng)
                label = Arr.name
                ax.plot(sym_freqs, wng_dB, label=label, linewidth=10 - arr_idx, linestyle=(0, (3, arr_idx)))
            plt.xlim(f0, f1)
            # plt.ylim([0, 20])
            plt.legend()
            plt.show()

        case 'df':
            fig, ax = plt.subplots()
            for arr_idx, Arr in enumerate(Arrs):
                df_plot = dB(Arr.df)
                # df_plot = Arr.df
                label = Arr.name
                ax.plot(sym_freqs, df_plot, label=label, linewidth=10 - arr_idx, linestyle=(0, (3, arr_idx)))
            plt.xlim(f0, f1)
            # plt.ylim([0, 20])
            plt.legend()
            plt.show()

        case 'snr':
            fig, ax = plt.subplots()
            for arr_idx, Arr in enumerate(Arrs):
                snr_plot = dB(Arr.snr)
                # df_plot = Arr.df
                label = Arr.name
                ax.plot(sym_freqs, snr_plot, label=label, linewidth=10 - arr_idx, linestyle=(0, (3, arr_idx)))
            plt.xlim(f0, f1)
            # plt.ylim([0, 20])
            plt.legend()
            plt.show()

        case 'fnbw':
            fig, ax = plt.subplots()
            for arr_idx, Arr in enumerate(Arrs):
                fnbw_plot = Arr.fnbw
                # df_plot = Arr.df
                label = Arr.name
                ax.plot(sym_freqs, fnbw_plot, label=label, linewidth=10 - arr_idx, linestyle=(0, (3, arr_idx)))
            plt.xlim(f0, f1)
            plt.ylim([0, 50])
            plt.legend()
            plt.show()
            # plt.plot(sym_angles, Arrs[0].beam[-1, :])
            # plt.show()

        case 'polar':
            fig, axs = plt.subplots(2, 4, subplot_kw={'projection': 'polar'})
            fs = [0, -1]
            beams = [Arr_LC.beam, Arr_CS.beam, Arr_PA.beam, Arr_LC.beam * Arr_CS.beam]
            for f_i in range(len(fs)):
                for b_i in range(len(beams)):
                    f_i_ = fs[f_i]
                    f = sym_freqs[f_i_]
                    pbeam = beams[b_i]
                    beam_f = dB(pbeam[f_i_, :])
                    ax = axs[f_i, b_i]
                    ax.plot(np.deg2rad(sym_angles), beam_f)
                    ax.set_ylim([-61, 0])
                    ax.axvline(x=td_rad, c='r')
                    for i in range(Ni):
                        ti_rad = tis_rad[i]
                        ax.axvline(x=ti_rad, c='k', linestyle='--')
                    ax.axvline(x=td_rad - tB_rad/2, c='k', linestyle=':')
                    ax.axvline(x=td_rad + tB_rad/2, c='k', linestyle=':')
                    axs[f_i, b_i] = ax
            plt.show()

        case 'polar2':
            fig, axs = plt.subplots(2, 4, subplot_kw={'projection': 'polar'})
            fs = [0, -1]
            beams = [Arr_LC.beam, Arr_CS.beam, Arr_ds.beam, Arr_PA.beam]
            for f_i in range(len(fs)):
                for b_i in range(len(beams)):
                    f_i_ = fs[f_i]
                    f = sym_freqs[f_i_]
                    pbeam = beams[b_i]
                    beam_f = dB(pbeam[f_i_, :])
                    ax = axs[f_i, b_i]
                    ax.plot(np.deg2rad(sym_angles), beam_f)
                    ax.set_ylim([-61, 0])
                    ax.axvline(x=td_rad, c='r')
                    for i in range(Ni):
                        ti_rad = tis_rad[i]
                        ax.axvline(x=ti_rad, c='k', linestyle='--')
                    ax.axvline(x=td_rad - tB_rad/2, c='k', linestyle=':')
                    ax.axvline(x=td_rad + tB_rad/2, c='k', linestyle=':')
                    axs[f_i, b_i] = ax
            plt.show()

        case 'gendata':
            for Arr in Arrs:
                freqs = sym_freqs.reshape(-1, ) / 1000
                angles = sym_angles.reshape(-1, )
                wng = dB(Arr.wng.reshape(-1, ))
                df = dB(Arr.df.reshape(-1, ))
                snr = dB(Arr.snr.reshape(-1, ))
                beam = dB(vect(Arr.beam).reshape(-1, ))
                fnbw = Arr.fnbw.reshape(-1, )
                params = [freqs, angles, wng, df, beam, snr, fnbw]
                params = fixDec(params)
                params = list(params)
                for idx, param in enumerate(params):
                    params[idx] = list(param)

                freqs, angles, wng, df, beam, snr, fnbw = tuple(params)

                wng_ = list(zip(freqs, wng))
                df_ = list(zip(freqs, df))
                snr_ = list(zip(freqs, snr))
                fnbw_ = list(zip(freqs, fnbw))
                b_freqs = freqs * len(angles)
                b_angles = []
                for angle in angles:
                    b_angles += [angle] * len(freqs)
                beam_ = list(zip(b_freqs, b_angles, beam))

                wng_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in wng_])
                df_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in df_])
                beam_ = 'freq,ang,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in beam_])
                snr_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in snr_])
                fnbw_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in fnbw_])

                filename = 'res' + name + '_' + Arr.name + '_'
                folder = 'results' + name + '/'
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                with open(folder + filename + 'wng.csv', 'w') as f:
                    f.write(wng_)
                    f.close()
                with open(folder + filename + 'df.csv', 'w') as f:
                    f.write(df_)
                    f.close()
                with open(folder + filename + 'snr.csv', 'w') as f:
                    f.write(snr_)
                    f.close()
                with open(folder + filename + 'fnbw.csv', 'w') as f:
                    f.write(fnbw_)
                    f.close()
                with open(folder + filename + 'beam.csv', 'w') as f:
                    f.write(beam_)
                    f.close()

            beam_min = -60
            beam_max = 0

            # data_defs
            beam_min = r'\def\ymin{{{}}}'.format(beam_min)
            beam_max = r'\def\ymax{{{}}}'.format(beam_max)
            meshcols = r'\def\meshcols{{{}}}'.format(sym_freqs.shape[0])
            meshrows = r'\def\meshrows{{{}}}'.format(len(sym_angles))
            colors = gen_palette(80, 60, 6, 345)
            lightK = r'\definecolor{LightG}{HTML}{3F3F3F}'

            data_defs = [beam_min, beam_max, meshcols, meshrows] + colors + [lightK]
            data_defs = '\n'.join(data_defs)

            with open('data_defs.tex', 'w') as f:
                f.write(data_defs)
                f.close()

        case _:
            pass


def main():
    result = 'beam'
    sizes = {
        'A': Params(LC=(2, 2, 0),
                    sd=(1, 2, 0),
                    ds=(1, 2, 0),
                    cb=(8, 1, 0),
                    ),
        'B': Params(LC=(2, 2, 0),
                    sd=(1, 1, 0),
                    ds=(1, 1, 0),
                    cb=(17, 1, 0),
                    ),
        'C': Params(LC=(4, 1, 0),
                    sd=(2, 1, 0),
                    ds=(2, 1, 0),
                    cb=(31, 1, 0),
                    ),
        'D': Params(LC=(4, 1, 0),
                    sd=(1, 1, 0),
                    ds=(1, 1, 0),
                    cb=(33, 1, 0),
                    ),
        'E': Params(LC=(1, 4, 1),
                    sd=(1, 2, 0),
                    ds=(1, 2, 0),
                    cb=(3, 1, 0),
                    ),
        'F': Params(LC=(1, 4, 1),
                    sd=(1, 1, 0),
                    ds=(1, 1, 0),
                    cb=(9, 1, 0),
                    ),
    }

    for name in sizes.keys():
        size = sizes[name]
        print(name)
        simulation(result, size, name)


if __name__ == '__main__':
    main()
