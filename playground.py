from field_parsing import *


def interpolation_div_SNR_test(points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    K_array = np.array([2 ** i for i in range(-3, 5)])
    Kx = K_array * 2 * np.pi / Lx
    Kz = K_array * 2 * np.pi / Lz
    resolutions = np.linspace(20, 70, 11)
    x_div_snr = test_div_x_interpolation(Kx, resolutions, points)
    z_div_snr = test_div_z_interpolation(Kz, resolutions, points)

    ax1 = plt.gca()
    im1 = ax1.imshow(x_div_snr, cmap='RdYlGn')
    ax1.set_xticks(np.linspace(0, len(resolutions) - 1, len(resolutions)))
    ax1.set_xticklabels(resolutions)
    ax1.set_yticks(np.linspace(0, len(Kx) - 1, len(Kx)))
    ax1.set_yticklabels(Kx / (2 * np.pi / Lx))
    ax1.set_title(r'Interpolation & divergence SNR -$K_x$ sweeping')
    plt.xlabel('Resolution [# of points]')
    plt.ylabel(r'$K_x$ / (2$\pi$$/L_x$)')
    clb1 = plt.colorbar(im1)
    clb1.ax.set_title('SNR [dB]', fontsize=8)


    plt.figure()
    ax2 = plt.gca()
    im2 = ax2.imshow(z_div_snr, cmap='RdYlGn')
    ax2.set_xticks(np.linspace(0, len(resolutions) - 1, len(resolutions)))
    ax2.set_xticklabels(resolutions)
    ax2.set_yticks(np.linspace(0, len(Kz) - 1, len(Kz)))
    ax2.set_yticklabels(Kz / ( 2 * np.pi / Lz))
    ax2.set_title(r'Interpolation & divergence SNR -$K_z$ sweeping')
    plt.xlabel('Resolution [# of points]')
    plt.ylabel(r'$K_z$ / (2$\pi$$/L_z$)')
    clb2  = plt.colorbar(im2)
    clb2.ax.set_title('SNR [dB]', fontsize=8)
    plt.show()

def interpolation_SNR_test(points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    K_array = np.array([2 ** i for i in range(-3, 5)])
    Kx = K_array * 2 * np.pi / Lx
    Kz = K_array * 2 * np.pi / Lz
    resolutions = np.linspace(20, 70, 11)
    x_div_snr = test_x_interpolation(Kx, resolutions, points)
    z_div_snr = test_z_interpolation(Kz, resolutions, points)

    ax1 = plt.gca()
    im1 = ax1.imshow(x_div_snr, cmap='RdYlGn')
    ax1.set_xticks(np.linspace(0, len(resolutions) - 1, len(resolutions)))
    ax1.set_xticklabels(resolutions)
    ax1.set_yticks(np.linspace(0, len(Kx) - 1, len(Kx)))
    ax1.set_yticklabels(Kx / (2 * np.pi / Lx))
    ax1.set_title(r'Interpolation SNR -$K_x$ sweeping')
    plt.xlabel('Resolution [# of points]')
    plt.ylabel(r'$K_x$ / (2$\pi$$/L_x$)')
    clb1 = plt.colorbar(im1)
    clb1.ax.set_title('SNR [dB]', fontsize=8)


    plt.figure()
    ax2 = plt.gca()
    im2 = ax2.imshow(z_div_snr, cmap='RdYlGn')
    ax2.set_xticks(np.linspace(0, len(resolutions) - 1, len(resolutions)))
    ax2.set_xticklabels(resolutions)
    ax2.set_yticks(np.linspace(0, len(Kz) - 1, len(Kz)))
    ax2.set_yticklabels(Kz / ( 2 * np.pi / Lz))
    ax2.set_title(r'Interpolation SNR -$K_z$ sweeping')
    plt.xlabel('Resolution [# of points]')
    plt.ylabel(r'$K_z$ / (2$\pi$$/L_z$)')
    clb2  = plt.colorbar(im2)
    clb2.ax.set_title('SNR [dB]', fontsize=8)
    plt.show()



def test_x_interpolation(K, resolutions, points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    x_snr = np.empty((len(K), len(resolutions)))
    for k_idx, Kx in enumerate(K):
        for r_idx, r in enumerate(resolutions):
            test = np.sin(Kx * x)
            interp_test, grid = interp3d(points, test, int(r), ignore_nan=False)
            xg, yg, zg = grid.T
            test_function = np.sin(Kx * xg)
            x_snr[k_idx, r_idx] = interpolation_SNR_score(test_function, interp_test, grid)
            print('---------------------------------', Kx / (2 * np.pi / Lx), r, x_snr[k_idx, r_idx])
    return x_snr


def test_div_x_interpolation(K, resolutions, points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    x_snr = np.empty((len(K), len(resolutions)))
    for k_idx, Kx in enumerate(K):
        for r_idx, r in enumerate(resolutions):

            test = np.array([np.sin(Kx * x), np.sin(Kx * x), np.sin(2 * np.pi * z/ Lz)])
            interp_test = []
            for i in range(3):
                interp_f, grid = interp3d(points, test[i, :], int(r), ignore_nan=False)
                interp_test.append(interp_f)
            interp_test = np.array(interp_test)
            xg, yg, zg = grid.T
            test_div = Kx * np.cos(Kx * xg) + 2 * np.pi / Lz * np.cos(2 * np.pi * zg / Lz)
            interp_div = regular_grid_div(interp_test, grid)
            x_snr[k_idx, r_idx] = interpolation_SNR_score(test_div, interp_div, grid)
            print('---------------------------------', Kx / (2 * np.pi / Lx), r, x_snr[k_idx, r_idx])
    return x_snr


def test_z_interpolation(K, resolutions, points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    z_snr = np.empty((len(K), len(resolutions)))
    for k_idx, Kz in enumerate(K):
        for r_idx, r in enumerate(resolutions):
            test = np.sin(Kz * z)
            interp_test, grid = interp3d(points, test, int(r), ignore_nan=False)
            xg, yg, zg = grid.T
            test_function = np.sin(Kz * zg)
            z_snr[k_idx, r_idx] = interpolation_SNR_score(test_function, interp_test, grid)
            print('---------------------------------', Kz / (2 * np.pi / Lz), r, z_snr[k_idx, r_idx])
    return z_snr


def test_div_z_interpolation(K, resolutions, points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    z_snr = np.empty((len(K), len(resolutions)))
    for k_idx, Kz in enumerate(K):
        for r_idx, r in enumerate(resolutions):

            test = np.array([np.sin(2 * np.pi * x / Lx), np.sin(2 * np.pi * y / Lx), np.sin(Kz * z)])
            interp_test = []
            for i in range(3):
                interp_f, grid = interp3d(points, test[i, :], int(r), ignore_nan=False)
                interp_test.append(interp_f)
            interp_test = np.array(interp_test)
            xg, yg, zg = grid.T
            test_div = 2 * np.pi / Lx * np.cos( 2 * np.pi * xg / Lx) + Kz * np.cos(Kz * zg)
            interp_div = regular_grid_div(interp_test, grid)
            z_snr[k_idx, r_idx] = interpolation_SNR_score(test_div, interp_div, grid)
            print('---------------------------------', Kz / (2 * np.pi / Lz), r, z_snr[k_idx, r_idx])
    return z_snr

def save_compared_gammas(folder_path, wv_path):
    paths = [
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um'
    ]
    for path in paths:
        if '0.15' in path and '2.6' in path:
            print("--- Bad data folder eV=0.15 rad=2.6um")
            continue
        E = load_modes(path)
        for Ek in E:
            dicts_list = compare_Gamma_k_methods(Ek, wv_path)
            file_name = r'\compare_gamma_k_Ef_{}_rad_{}_fk_{}_.pkl'.format(Ek.ef, Ek.radius, Ek.freq_str)
            print(file_name[1:])
            save_dict(dicts_list, folder_path + file_name)

def load_specific_compared_set(folder_path, ef, rad):
    mode_dicts = []
    for filename in os.listdir(folder_path):
        if "Ef_{}_rad_{}".format(ef, rad) in filename:
            mode_dicts.append(load_dict(os.path.join(folder_path, filename)))
    return mode_dicts

def plot_specific_compared_set(folder_path, ef, rad):
    mode_dicts = load_specific_compared_set(folder_path, ef, rad)
    #mode_dicts = [mode_dicts[0]]
    for mode in mode_dicts:
        for mode_dipole in mode:
            if 'z_linspace' not in mode_dipole.keys():
                z_linspace = np.load(os.path.join(folder_path, r'z_linspace_patch.npy'))
            else:
                z_linspace = mode_dipole['z_linspace']
            plot_comparison(mode_dipole, z_linspace)


def plot_comparison(mode_dipole, z_linspace=None):
    if 'z_linspace' not in mode_dipole.keys() and z_linspace is None:
        raise ValueError('z_linspace is None')
    else:
        z_nm = z_linspace * 1e9
        total_k_rate = mode_dipole['total_k_rate']
        spont_emission = mode_dipole['spont_emission']
        total_k_div_rate = mode_dipole['total_k_div_rate']
        avg_G_k = mode_dipole['avg_G_k']
        interp_bandplot = mode_dipole['interp_bandplot']
        ef = mode_dipole['Ef']
        rad = mode_dipole['radius']
        f_k = np.round(float(mode_dipole['f_k']) / 1e12, 2)
        dip = (mode_dipole['dipole'] == 2) * 'ULS' + (mode_dipole['dipole'] == 1) * 'INJ'
        plt.plot(z_nm, np.real(total_k_rate) / spont_emission, z_nm,
                 np.real(total_k_div_rate) / spont_emission, z_nm, np.real(avg_G_k) / spont_emission)
        plt.ylabel(r'$F_p$ enhancement [1]')
        plt.xlabel('z [nm]')
        plt.title('$F_p$ over active regions; Ef={}eV, rad={}, $f_k$={}THz, {}'.format(ef, rad, f_k, dip))
        interp_bandplot = interp_bandplot - min(interp_bandplot) # should have happend begore saving, bu the plots are wrong without it
        plt.plot(z_nm[:-10], interp_bandplot[:-10] / abs(max(interp_bandplot[:-10])) * max(np.real(total_k_rate) / spont_emission) / 0.95, 'k--', linewidth=0.3)
        plt.legend(
            ['Coloumn gauge', 'With DIV', 'Dipole approx.', 'Energy band [a.u]'])
        plt.show()

    return

def print_purcell_per_set(folder_path, ef, rad, fix_fp=False):
    spont_emission, theory_purcell_list, estimated_fp_list, f_k_list, dip, ef, rad = extract_purcells_set(folder_path, ef, rad, fix_fp)
    G_rate_fp, div_rate_fp = get_purcell_of_emission(folder_path, ef, rad)
    theory_purcell = np.sum(np.real(theory_purcell_list), 1)
    estimated_fp = np.sum(np.real(estimated_fp_list), 1)
    for i, dipole in enumerate(dip):
        print("--- Device = Ef={}eV, Radius={}m, dipole={}->LLS".format(ef, rad, dipole))
        print("     --- Theory Purcell factor = {}".format(np.round(theory_purcell[i], 2)))
        print("     --- Estimated Purcell factor = {}".format(np.round(estimated_fp[i], 2)))
        print("     --- Fp for dipole approximation = {}".format(np.round(G_rate_fp[i] ,2)))
        print("     --- Fp for full expression = {}".format(np.round(div_rate_fp[i], 2)))

def get_purcell_of_emission(folder_path, ef, rad):
    z_linspace, spont_emission, total_k_rate_list, total_k_div_rate_list, avg_G_k_list, f_k_list, dip, ef, rad, interp_bandplot = extract_emission_set(
        folder_path, ef, rad)
    k_rate = np.sum(np.real(total_k_rate_list), 1)
    k_div_rate = np.sum(np.real(total_k_div_rate_list), 1)
    Gk_rate = np.sum(np.array(avg_G_k_list), 1)
    dz_normalize = regular_integration_1d(dipole_locations(z_linspace), z_linspace)

    Gk_rate_Yz = Gk_rate / dz_normalize
    G_div_rate_Yz = k_div_rate / dz_normalize

    Ndip = len(dip)
    G_rate_fp = [regular_integration_1d(Gk_rate_Yz[i], z_linspace) / spont_emission[i] for i in range(Ndip)]
    div_rate_fp = [regular_integration_1d(G_div_rate_Yz[i], z_linspace) / spont_emission[i] for i in range(Ndip)]

    return G_rate_fp, div_rate_fp

def plot_total_set(folder_path, ef, rad):
    z_linspace, spont_emission, total_k_rate_list, total_k_div_rate_list, avg_G_k_list, f_k_list, dip, ef, rad, interp_bandplot = extract_emission_set(folder_path, ef, rad)
    z_nm = z_linspace * 1e9
    k_rate = np.sum(np.real(total_k_rate_list), 1)
    k_div_rate = np.sum(np.real(total_k_div_rate_list), 1)
    Gk_rate = np.sum(np.array(avg_G_k_list), 1)
    for i, dipole in enumerate(dip):
        plt.figure()
        plt.plot(z_nm, k_div_rate[i] / spont_emission[i], z_nm, k_rate[i] / spont_emission[i], z_nm, Gk_rate[i] / spont_emission[i])
        interp_bandplot = interp_bandplot - min(interp_bandplot)  # should have happend begore saving, bu the plots are wrong without it
        plt.plot(z_nm[:-10], interp_bandplot[:-10] / abs(max(interp_bandplot[:-10])) * max(np.real(k_rate[i]) / spont_emission[i]) / 0.95, 'k--', linewidth=0.3)
        plt.title(r'Emission enhancement - $E_f$={}eV, radius={}m, {}'.format(ef, rad, dipole))
        plt.xlabel('z [nm]')
        plt.ylabel('Active region $F_p$ [1]')
        plt.legend([r'With $\nabla$$\cdot$A', 'Coulomb gauge', 'Dipole approx.', 'Energy band [a.u.]'])
    plt.show()

def extract_emission_set(folder_path, ef, rad):
    mode_dicts = load_specific_compared_set(folder_path, ef, rad)
    # mode_dicts = [mode_dicts[0]]
    created_lists = False

    for mode in mode_dicts:
        if not created_lists:
            total_k_rate_list = [[] for i in range(len(mode))]
            total_k_div_rate_list = [[] for i in range(len(mode))]
            avg_G_k_list = [[] for i in range(len(mode))]
            f_k_list = [[] for i in range(len(mode))]
            spont_emission = [[] for i in range(len(mode))]
            dip = [[] for i in range(len(mode))]
            created_lists = True
        for i, mode_dipole in enumerate(mode):
            if 'z_linspace' not in mode_dipole.keys():
                z_linspace = np.load(os.path.join(folder_path, r'z_linspace_patch.npy'))
            else:
                z_linspace = mode_dipole['z_linspace']
            total_k_rate_list[i].append(mode_dipole['total_k_rate'])
            total_k_div_rate_list[i].append(mode_dipole['total_k_div_rate'])
            avg_G_k_list[i].append(mode_dipole['avg_G_k'])
            f_k_list[i].append(mode_dipole['f_k'])
            spont_emission[i] = mode_dipole['spont_emission']
            interp_bandplot = mode_dipole['interp_bandplot']
            ef = mode_dipole['Ef']
            rad = mode_dipole['radius']
            dip[i] = (mode_dipole['dipole'] == 2) * 'ULS' + (mode_dipole['dipole'] == 1) * 'INJ'
            print(i, regular_integration_1d(np.real(mode_dipole['total_k_div_rate']) / regular_integration_1d(dipole_locations(z_linspace), z_linspace), z_linspace) / spont_emission[i])
            print(i, mode_dipole['Fp_estimation'])
            print(i, fix_purcell(mode_dipole['f_k'], mode_dipole['f_ij'], mode_dipole['Fp_estimation']))
    return z_linspace, spont_emission, total_k_rate_list, total_k_div_rate_list, avg_G_k_list, f_k_list, dip, ef, rad, interp_bandplot

def extract_purcells_set(folder_path, ef, rad, fix_fp=False):
    mode_dicts = load_specific_compared_set(folder_path, ef, rad)
    # mode_dicts = [mode_dicts[0]]
    created_lists = False

    for mode in mode_dicts:
        if not created_lists:
            theory_purcell_list = [[] for i in range(len(mode))]
            estimated_fp_list = [[] for i in range(len(mode))]
            f_k_list = [[] for i in range(len(mode))]
            spont_emission = [[] for i in range(len(mode))]
            dip = [[] for i in range(len(mode))]
            created_lists = True
        for i, mode_dipole in enumerate(mode):
            if fix_fp:  # because i accidently saved 07/11/22 results with f_k instead of f_if
                theory_purcell_list[i].append(
                    fix_purcell(mode_dipole['f_k'], mode_dipole['f_ij'], mode_dipole['theory_purcell']))
                estimated_fp_list[i].append(
                    fix_purcell(mode_dipole['f_k'], mode_dipole['f_ij'], mode_dipole['Fp_estimation']))
            else:
                theory_purcell_list[i].append(mode_dipole['theory_purcell'])
                estimated_fp_list[i].append(mode_dipole['Fp_estimation'])
            f_k_list[i].append(mode_dipole['f_k'])
            spont_emission[i] = mode_dipole['spont_emission']
            ef = mode_dipole['Ef']
            rad = mode_dipole['radius']
            dip[i] = (mode_dipole['dipole'] == 2) * 'ULS' + (mode_dipole['dipole'] == 1) * 'INJ'

    return spont_emission, theory_purcell_list, estimated_fp_list, f_k_list, dip, ef, rad


def fix_purcell(f_k, f_if, purcell):
    return purcell * (((cn.c / f_if) / np.sqrt(np.real(disspersion(freq2energy(f_if))[1]))) ** 3) / (((cn.c / f_k) / np.sqrt(np.real(disspersion(freq2energy(f_k))[1]))) ** 3)


def check_spatial_frequency(Ek):
    points = Ek.points
    #u_z = Ek.e_field[:, 2]
    x, y, z = points.T
    u_z = np.sin(2 * np.pi * z / (z.max() - z.min()))
    magnitude_z = abs(u_z) ** 2
    avg_z = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
    magnitude_z = np.cos(np.pi * 10 * z / (z.max() - z.min()))
    #average_u_z, avg_z = averaging_over_area_non_interp(points, Ek.disk_area, magnitude_z)
    average_u_z = averaging_over_area(points, Ek.disk_area, avg_z, magnitude_z)
    Nz = len(average_u_z)
    Lz = avg_z.max() - avg_z.min()
    srz = Nz/Lz # srz = sampling rate over z
    Dz = 1.0 / srz # period length
    n_z = np.arange(Nz)
    kz = n_z / (Nz * Dz) / (2 * np.pi / Lz)
    U_Z = np.fft.fft(average_u_z)

    print(srz / (2 * np.pi/Lz), Dz/ Lz)
    plt.figure()
    plt.plot(z, magnitude_z)
    plt.figure()
    plt.plot(avg_z, average_u_z)
    plt.figure()
    plt.stem(kz, abs(U_Z), use_line_collection='True')
    plt.show()



