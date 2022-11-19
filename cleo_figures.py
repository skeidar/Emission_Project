from utilities import *
from field_parsing import *
from playground import *
import matplotlib.gridspec as gridspec
from wavefunc import load_wavefunction, plot_wavefunctions_along_z
import matplotlib.image as mpimg
from matplotlib import figure as fg
plt.rcParams['svg.fonttype'] = 'none'


def cleo_plot(Elist, folder_path, wv_path):
    generate_fig_1(Elist, folder_path, wv_path)



def generate_fig_1(Elist, folder_path, wv_path):
    fig = plt.figure(figsize=(11,4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    # save space for the illustration of the device at gs[0]
    #gs[1] goes for the zooming in
    wv_layer_plot(wv_path, Elist, fig, gs[1])
    #gs[2] goes for the Purcell vs z plot
    purcell_factor_plot(Elist, folder_path, fig, gs[2])
    plt.show()

def modes_plot(Elist, fig, gspc):
    #fig = plt.figure(figsize=(10, 3))
    ims = []
    #gs = gridspec.GridSpec(1, 2, wspace=0.25, top=0.8, bottom=0.14, left=0.16, right=0.84, width_ratios=[1, 1], height_ratios=[1])
    #gs = gridspec.GridSpecFromSubplotSpec(1,2, gspc)
    font_size = 8
    legend_list = []
    #zplt = fig.add_subplot(gs[0])
    pplt = fig.add_subplot(gspc, frame_on=False, xticks=[], yticks=[])
    pplt.set_xlabel(r'x [$\mu$m]', fontsize=font_size, labelpad=15)
    pplt.set_ylabel(r'y [$\mu$m]', fontsize=font_size, labelpad=15)

    for idx, Efield in enumerate(Elist):
        #in_gs = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0], wspace=0.5, hspace=0.5)
        polar_gs = gridspec.GridSpecFromSubplotSpec(2, 3, gspc, width_ratios=[1, 1,  0.1], height_ratios=[1, 1], wspace=0, hspace=0)
        if idx == 4 or idx == 5:
            break
        polar_e = Efield.return_polar()
        xa, ya, za = polar_e
        splt = fig.add_subplot(polar_gs[int(idx / 2), idx % 2])
        #im = splt.contourf(xa / 1e-6, ya / 1e-6, za, 50, cmap=plt.cm.plasma)
        im = splt.pcolormesh(xa / 1e-6, ya / 1e-6, za, cmap=plt.cm.plasma, rasterized=True)
        #splt.set_aspect('equal', adjustable='box')
        if idx == 1:
            splt.set_xticks([])
            splt.set_yticks([])
            splt.set_xticklabels([])
            splt.set_yticklabels([])
        elif idx == 0:
            splt.set_xticks([])
            splt.set_xticklabels([])
            splt.set_yticks([-2, 0, 2])
            splt.set_yticklabels([-2, 0, 2], fontsize=font_size)
        elif idx == 2:
            splt.set_xticks([-2, 0, 2])
            splt.set_xticklabels([-2, 0, 2], fontsize=font_size)
            splt.set_yticks([-2, 0, 2])
            splt.set_yticklabels([-2, 0, 2], fontsize=font_size)
        elif idx == 3:
            splt.set_yticks([])
            splt.set_yticklabels([])
            splt.set_xticks([-2, 0, 2])
            splt.set_xticklabels([-2, 0, 2], fontsize=font_size)



    cax = fig.add_subplot(polar_gs[:,2])
    cbar = plt.colorbar(im, cax, ticks=[0, 15])
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.set_yticklabels(['0', '15'])
    cbar.set_label('$|E_k(x,y)|^2$', rotation=270, fontsize=font_size, labelpad=-6)

    return

def purcell_factor_plot(Elist, folder_path, fig, gspc):
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, gspc, hspace=0)
    font_size = 8
    ef = float(Elist[0].ef)
    rad = float(Elist[0].radius)
    z_linspace, spont_emission, total_k_rate_list, total_k_div_rate_list, avg_G_k_list, f_k_list, dip, ef, rad, interp_bandplot = extract_emission_set(
        folder_path, ef, rad)
    z_nm = z_linspace * 1e9
    k_div_rate = np.sum(np.real(total_k_div_rate_list), 1)
    Gk_rate = np.sum(np.array(avg_G_k_list), 1)
    dipole = 0

    # fig (ii) - bottom
    splt = fig.add_subplot(gs[1])
    splt.plot(z_nm, k_div_rate[dipole] / spont_emission[dipole], z_nm,
              Gk_rate[dipole] / spont_emission[dipole])
    splt.set_xlabel('z [nm]', fontsize=font_size, labelpad=0.5)
    splt.set_ylabel('Active Regions $F_p$ [1]', fontsize=font_size, labelpad=0.5)
    # how to change to 1e4 and its size:
    #splt.ticklabel_format(axis='y', scilimits=[-3, 3])
    #ofst = splt.yaxis.get_offset_text()
    #ofst.set_size(font_size)
    splt.xaxis.set_tick_params(labelsize=font_size)
    splt.yaxis.set_tick_params(labelsize=font_size)
    splt.set_yticks([0 + 4000 * i for i in range(4)])
    splt.set_yticklabels([str(0 + 4000 * i) for i in range(4)])
    splt.legend(['Non-dipole', 'Dipole-approx.'], fontsize=font_size)

    # fig (i) - top
    xplt = fig.add_subplot(gs[0])
    mode_non_dipole = np.real(total_k_div_rate_list[dipole][-1])
    mode_dipole = avg_G_k_list[dipole][-1]
    xplt.plot(z_nm, mode_non_dipole / spont_emission[dipole], z_nm,
              mode_dipole / spont_emission[dipole])
    xplt.xaxis.set_tick_params(labelsize=font_size)
    xplt.yaxis.set_tick_params(labelsize=font_size)
    xplt.set_xticks([])
    xplt.set_xticklabels([])
    xplt.legend(['Non-dipole', 'Dipole-approx.'], fontsize=font_size)
    xplt.set_ylabel('Active Regions $F_p$ [1]', fontsize=font_size, labelpad=0.5)
    xplt.set_yticks([0 + 2000 * i for i in range(3)])
    xplt.set_yticklabels([str(0 + 2000 * i) for i in range(3)])
    return


def wv_layer_plot_with_img(wv_path):
    fig = plt.figure(figsize=(11, 3))
    font_size = 8
    z, psi_f, psi_uls, psi_inj, bandplot = wavefunction_for_plot(wv_path)
    z = z * 1e9
    gs = gridspec.GridSpec(1, 2, bottom=0.14,width_ratios=[1.2,1], height_ratios=[1], left=0.05, right=0.95, top=0.85)

    img_path = r"C:\Shaked\Technion\QCL_Project\CLEO\submission\cavity and layers.png"
    im = mpimg.imread(img_path)
    cplt = fig.add_subplot(gs[0], frame_on=False, xticks=[], yticks=[])
    cav = cplt.imshow(im, aspect=1)
    cplt.text(0.95 * len(im[0,:]), 0.01 * len(im[:,0]), '(a)', fontsize=font_size)

    in_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[1], hspace=0,height_ratios=[1,1.5])
    wvax = fig.add_subplot(in_gs[1])
    wvax.plot(z, bandplot, label='Bands')
    wvax.plot(z, psi_f, label=r'|$\psi$$_f$$(z)|^2$')
    wvax.plot(z, psi_uls, label=r'|$\psi$$_u$$_l$$_s$$(z)|^2$')
    wvax.plot(z, psi_inj, label=r'|$\psi$$_i$$_n$$_j$$(z)|^2$')
    wvax.legend(fontsize=font_size, loc='upper left')
    wvax.set_xlim([min(z), max(z)])
    wvax.xaxis.set_tick_params(labelsize=font_size)
    wvax.set_xlabel('z [nm]', fontsize=font_size)
    wvax.set_yticks([])
    wvax.set_yticklabels([])
    lax = fig.add_subplot(in_gs[0])
    lax.text(max(z) * 0.94, 1 * 0.90, '(b)', fontsize=font_size)
    wvax.text(max(z) * 0.94, max(bandplot) * 0.90, '(c)', fontsize=font_size)
    show_layers(z, lax)

def wv_layer_plot(wv_path, Elist, fig, gspc):
#    fig = plt.figure(figsize=(10, 3))
    font_size = 8
    nQW = 2
    #z, psi_f, psi_uls, psi_inj, bandplot = wavefunction_for_plot(wv_path)

    #gs = gridspec.GridSpec(2, 1, bottom=0.14, hspace=0, left=0.3, right=0.7, height_ratios=[1,1.5])
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, gspc, hspace=0, height_ratios=[1,1,1])


    wvax = fig.add_subplot(gs[1])
    z = wavefunction_and_field(wv_path, Elist, wvax, nQW)
    z = z * 1e9
    wvax.xaxis.set_tick_params(labelsize=font_size)
    wvax.set_yticks([])
    wvax.set_yticklabels([])
    wvax.set_xticks([])
    wvax.set_xticklabels([])
    lax = fig.add_subplot(gs[0])


    eax = fig.add_subplot(gs[2])
    wvax.set_xlabel('z [nm]', fontsize=font_size)

    show_layers(z, lax, nQW)



def show_layers(z, ax, nQW):
    perLen = 30.68
    gaas_label = "GaAs"
    alas_label = "AlAs"
    ar_label = "Active Region"
    for i in range(nQW):
        offset = perLen * i
        gaas1 = 8.28
        alas2 = 1.97
        gaas3 = 8.7
        gaas4 = 3.0
        gaas5 = 5.35
        alas6 = 3.38
        font_size = 8
        ax.fill_between(z, 0, 1, where=np.logical_or(np.logical_and(offset < z, z < gaas1 + offset), np.logical_and(gaas1 + alas2 + offset < z ,z < gaas1+alas2+gaas3+gaas4+gaas5+ offset)),color='#66B2FF', label=gaas_label, alpha=0.8, transform=ax.get_xaxis_transform())
        ax.fill_between(z, 0, 1, where=np.logical_or(np.logical_and(z > gaas1+alas2+gaas3+gaas4+gaas5+ offset, z < gaas1+alas2+gaas3+gaas4+gaas5+alas6+ offset), np.logical_and(gaas1+ offset < z , z < gaas1+alas2+ offset)), color='#CC99FF', alpha=0.8, label=alas_label, transform=ax.get_xaxis_transform())
        ax.fill_between(z, 0, 1, where=np.logical_and(gaas1+alas2+gaas3+ offset < z , z < gaas1+alas2+gaas3+gaas4+ offset), edgecolor='#33FF33', facecolor='none', hatch='//', label=ar_label, alpha=0.8, transform=ax.get_xaxis_transform())
        ax.set_xlim([min(z), max(z)])
        ax.xaxis.set_tick_params(labelsize=font_size)
        ax.axvline(gaas1 + offset, linestyle='dashed')
        ax.axvline(gaas1 + alas2 + offset, linestyle='dashed')
        ax.axvline(gaas1 + alas2 + gaas3 + offset, linestyle='dashed')
        ax.axvline(gaas1 + alas2 + gaas3+ gaas4 + offset, linestyle='dashed')
        ax.axvline(gaas1 + alas2 + gaas3 + gaas4 + gaas5 + offset ,linestyle='dashed')
        ax.axvline(gaas1 + alas2 + gaas3 + gaas4 + gaas5 + alas6 + offset, linestyle='dashed')
        gaas_label = "_nolegend_"
        alas_label = "_nolegend_"
        ar_label = "_nolegend_"
    ax.legend(fontsize=font_size)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def wavefunction_and_field(wv_path, Elist, subplt, nQW):
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    font_size = 8
    z_linspace = np.linspace(0, 360e-9, 10000)
    z_wv = z_wv * 1e-9
    bandplot[:, 0] = bandplot[:, 0] * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7
    psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
    psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
    psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
    z_band = np.array([])
    band_plot = np.array([])
    perLen = 30.68e-9
    f_energy_diff = (levelstot[0] - levelstot[7]) * 1000
    uls_energy_diff = (levelstot[2] - levelstot[9]) * 1000
    inj_energy_diff = (levelstot[1] - levelstot[8]) * 1000
    band_energy_diff = (levelstot[0] - levelstot[7]) * 1000
    Efield = Elist[3]
    for i in range(nQW):
        OFFSET = 0e-9
        z_qw = z_wv + perLen * i + OFFSET
        z_per_min = perLen * i
        z_per_max = perLen * (i+1)
        interp_psi_f_func = interpolate.interp1d(z_qw, abs(psi_f) ** 2, kind='cubic',
                                                 fill_value=0, bounds_error=False)
        interp_psi_uls_func = interpolate.interp1d(z_qw, abs(psi_uls) ** 2, kind='cubic',
                                                 fill_value=0, bounds_error=False)
        interp_psi_inj_func = interpolate.interp1d(z_qw, abs(psi_inj) ** 2, kind='cubic',
                                                 fill_value=0, bounds_error=False)
        interp_bandplot_func = interpolate.interp1d(bandplot[:,0] + perLen * i + OFFSET, bandplot[:,1], kind='linear',
                                                 fill_value=0, bounds_error=False)
        periods_args = np.logical_and(z_linspace >= (z_qw).min(), z_linspace < (z_qw).max())
        band_period = np.logical_and(z_linspace >= z_per_min, z_linspace <= z_per_max)

        interp_psi_f = interp_psi_f_func(z_linspace) / 2e6 - f_energy_diff * i + levelstot[0] * 1000
        interp_psi_uls = interp_psi_uls_func(z_linspace) / 2e6 - uls_energy_diff * i + levelstot[2] * 1000
        interp_psi_inj = interp_psi_inj_func(z_linspace) / 2e6 - inj_energy_diff * i + levelstot[2] * 1000
        interp_bandplot = interp_bandplot_func(z_linspace) - band_energy_diff * i #+ levelstot[0] * 10
        z_band = np.concatenate([z_band, z_linspace[band_period]])
        band_plot = np.concatenate([band_plot, interp_bandplot[band_period]])
        if i % 2 == 0:
            linestyle = 'solid'
            f_label  = r'$|\psi$$_f$|$^2$'
            uls_label = r'|$\psi$$_U$$_L$$_S$|$^2$'
            inj_label = r'|$\psi$$_I$$_N$$_J$|$^2$'
        else:
            linestyle = 'dashed'
            f_label  = "_nolegend_"
            uls_label = "_nolegend_"
            inj_label = "_nolegend_"
        subplt.plot(z_linspace[periods_args] * 1e9, interp_psi_f[periods_args], 'orange', linestyle=linestyle, label=f_label)
        subplt.plot(z_linspace[periods_args] * 1e9, interp_psi_uls[periods_args], 'green', linestyle=linestyle, label=uls_label)
        subplt.plot(z_linspace[periods_args]  *1e9, interp_psi_inj[periods_args], 'red', linestyle=linestyle, label=inj_label)
    subplt.plot(z_band * 1e9, band_plot, 'k', label="Band")

    z_linspace, Ez = Efield.return_z_field()
    subplt.plot(z_linspace * 1e9, Ez[::-1] * 10, 'b', label= r'|E$_k$$|^2$')
    subplt.set_xlim([0, z_band.max() * 1e9])
    subplt.legend(fontsize=font_size, loc='lower right')
    return z_band

def wavefunction_for_plot(wv_path):
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    z_linspace = np.linspace(0, 2 * 30.68e-9, 10000)
    z_wv = z_wv * 1e-9
    bandplot[:, 0] = bandplot[:, 0] * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7
    psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
    psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
    psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
    interp_psi_f_func = interpolate.interp1d(z_wv, abs(psi_f) ** 2, kind='cubic', fill_value=0, bounds_error=False)
    interp_psi_uls_func = interpolate.interp1d(z_wv, abs(psi_uls) ** 2, kind='cubic', fill_value=0, bounds_error=False)
    interp_psi_inj_func = interpolate.interp1d(z_wv, abs(psi_inj) ** 2, kind='cubic', fill_value=0, bounds_error=False)
    interp_bandplot_func = interpolate.interp1d(bandplot[:, 0], bandplot[:, 1], kind='linear', fill_value=0, bounds_error=False)
    interp_psi_f = interp_psi_f_func(z_linspace) / 1e6 + levelstot[0] * 1000
    interp_psi_uls = interp_psi_uls_func(z_linspace) / 1e6 + levelstot[2] * 1000
    interp_psi_inj = interp_psi_inj_func(z_linspace) / 1e6 + levelstot[1] * 1000
    interp_bandplot = interp_bandplot_func(z_linspace)

    return z_linspace, interp_psi_f, interp_psi_uls, interp_psi_inj, interp_bandplot

def modes_tight_polat(Elist):
    fig = plt.figure(figsize=(4, 3))
    ims = []
    gs = gridspec.GridSpec(1, 2, wspace=0.05, top=0.75, right=0.9, width_ratios=[1, 0.05], height_ratios=[1])
    font_size = 8
    s = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, Efield in enumerate(Elist):
        in_gs = gridspec.GridSpecFromSubplotSpec(2, 3, gs[0], wspace=0, hspace=0)
        #Efield.normalize(freq2energy(Efield.frequency))
        polar_e = Efield.return_polar()
        xa, ya, za = polar_e
        splt = fig.add_subplot(in_gs[int(idx / 3), idx % 3])

        im = splt.contourf(xa / 1e-6, ya / 1e-6, za, 50, cmap=plt.cm.plasma)
        splt.set_aspect('equal', adjustable='box')
        splt.set_xticks([])
        splt.set_yticks([])
        splt.set_xticklabels([])
        splt.set_yticklabels([])
        splt.text(min(xa / 1e-6) * 0.97, max(ya / 1e-6) * 0.97, s[idx], fontsize=font_size, va='top', ha='left')
        ims.append(im)
    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(im, cax, ticks=[0, 10])
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.set_yticklabels(['0', '10.0'])
    cbar.set_label('$|E_k(x,y)|^2$', rotation=270, fontsize=font_size, labelpad=-8)
    return ims



def modes_polar_plot(Elist):
    fig = plt.figure(figsize=(10, 3))
    ims = []
    gs = gridspec.GridSpec(2, 4, hspace=0.7, right=0.75, left=0.25, width_ratios=[1, 1, 1, 0.1])
    font_size = 8
    s = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, Efield in enumerate(Elist):
        #Efield.normalize(freq2energy(Efield.frequency))
        polar_e = Efield.return_polar()
        xa, ya, za = polar_e
        splt = fig.add_subplot(gs[int(idx / 3), idx % 3])

        im = splt.contourf(xa / 1e-6, ya / 1e-6, za, 50, cmap=plt.cm.plasma)
        splt.set_aspect('equal', adjustable='box')
        splt.set_title(r'{} $f_k$={}THz'.format(s[idx], Efield.freq_str), fontsize=font_size)
        splt.set_xlabel(r'x [$\mu$m]', fontsize=font_size, labelpad=-1.2)
        splt.set_ylabel(r'y [$\mu$m]', fontsize=font_size, labelpad=-6)
        splt.xaxis.set_tick_params(labelsize=font_size)
        splt.yaxis.set_tick_params(labelsize=font_size)
        ims.append(im)
    cax = fig.add_subplot(gs[:, 3])
    cbar = plt.colorbar(im, cax, ticks=[0, 10])
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.set_yticklabels(['0', '10.0'])
    cbar.set_label('$|E_k(x,y)|^2$', rotation=270, fontsize=font_size, labelpad=-8)
    return ims


def mode_z_plot(Elist):
    fig = plt.figure(figsize=(4, 3))
    Ezlist = []
    for Efield in Elist:
        z_linspace, Ez = Efield.return_z_field()
        z_nm = z_linspace[::-1] * 1e9
        plt.plot(z_nm, Ez)
        Ezlist.append(Ez)
    return Ezlist, z_nm



def cavity_plot():
    fig = plt.figure(figsize=(6.3, 2.1))
    font_size = 8
    img_path = r"C:\Shaked\Technion\QCL_Project\CLEO\submission\cavity and layers.png"
    im = mpimg.imread(img_path)
    cplt = fig.add_subplot(frame_on=False)
    cav = cplt.imshow(im)


def spectrum_plot():

    return
