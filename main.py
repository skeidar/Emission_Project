from field_parsing import *
from wavefunc import load_wavefunction, normalize_wv
from matplotlib import pyplot as plt



if __name__ == "__main__":
    file_name = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\test.npy"
    csv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data"
    #wv_path = r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0"
    wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_APL_2019\2-well_APL_2019"
    ULS_path = r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_12_21'
    INJ_path = r'C:\Shaked\Technion\QCL_Project\logs\21_12_21_injector'
    new_logs_path = r'C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\15_12_2021'
    EM_path= r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution'
    sim_files_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\050222"
    divergence_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\11_02_22_divergences"
    f_array = np.linspace(3e12, 4.8e12, 101)
    #f_array = np.linspace(3e12, 4.8e12, 11)
    # f_array = np.linspace(3e12, 4.8e12, 301)
    e_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.2um"

    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.2um")


    #save_divergences(divergence_path)
    #test = np.load(divergence_path + r"\Ef_0.25eV_rad_2.2e-06um_RES_31_dZ.npy")
    compare_Gamma_k_methods(wv_path)

    #generic_scatter_3d(points, np.real(E[0].e_field[:, 2]), False)
    #generic_scatter_3d(grid, np.real(interp_f), False)

    #for Ei in E:
        #energy = freq2energy(Ei.frequency)
        #Ei.normalize(energy)
        #Ei.scatter_plot_3d()
        #Ei.plot_2d_z_rho_slice()
        #Ei.plot_gamma_vs_freq_along_z()

    #generic_scatter_3d(grid, testi, False)
    #num_div = divergence(np.array([fi, fi, fi]))#, steps2)
    #dvgrns(np.array([fi, fi, fi]), grid)
    #generic_scatter_3d(grid, np.ones(len(grid)))
    #generic_scatter_3d(points, f, True)
    #generic_scatter_3d(grid, fi, True)
    #print(np.shape(E_field.T[0]))
    #print(divergence(interpE.T, steps_grid))




    #run_total_check_dipole_rate(wv_path, f_array, sim_files_path,qwgammaplt=True, sumplt=True, zflip=True, nsec=True)

    #x = E[0].points[:, 0]
    #y = E[0].points[:, 1]
    #z = E[1].points[:, 2]
    #u_z = E[1].e_field[:, 2]
    #z_linspace = np.linspace(z.min(), z.max(), 10000)
    #for Ei in E:
        #name = "Ef_" + str(Ei.ef) + "_r_" + str(np.round(Ei.radius * 1e6, 2)) + 'um_f_' + Ei.freq_str + 'THz'
        #Ei.approximations_comparison(wv_path, name, save_log=True)

    #run_approximation_comparison(EM_path, wv_path, save_log=True)


    #show_rate_comparison(r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_12_21')
    #run_emission_spectrum(csv_path, f_array, plot=True, verbose=False)
    #show_rate_for_QW(r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_12_21')
    #new_logs_path = r'C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\15_12_2021'
    #run_save_effective_emission(new_logs_path, f_array)
    #run_emission_spectrum(new_logs_path, f_array, plot=True, verbose=False)
    #fix_eps_r(r'C:\Shaked\Technion\QCL_Project\logs\18_12_21', r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_12_21')
    #show_rate_comparison(r'C:\Shaked\Technion\QCL_Project\logs\21_12_21_injector')
    #show_rate_for_QW(r'C:\Shaked\Technion\QCL_Project\logs\21_12_21_injector')

    #non_approx_emission_enhancement(ULS_path, INJ_path)
    #show_non_approx_enhancement(ULS_path, INJ_path, f_array)
