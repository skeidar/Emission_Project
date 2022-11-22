from field_parsing import *
from playground import *
from cleo_figures import *
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
    e_path2 = r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um"

    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um")
    E2 = load_modes(e_path2)
    #E = load_modes(e_path)

    Ek = E[0]
    points = Ek.points

    div_test = np.load(r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\11_02_22_divergences\Ef_0.25eV_rad_2.2e-06um_f_3.81THz_RES_28_DIV.npy")
    my_grid = np.load(r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\11_02_22_divergences\Ef_0.25eV_rad_2.2e-06um_f_3.81THz_RES_28_grid.npy")
    madafaka = np.zeros([len(div_test), 4])

    #madafaka[:, 3] = np.real(div_test)
    #madafaka[:, :3] = my_grid

    #generic_scattr_3d(my_grid, np.imag(div_test), False)
    Ek = E[0]
    points = Ek.points.T
    field_k = Ek.e_field


    folder_path = r'C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\071122_gamma_compare'
    #compare_Gamma_k_methods(Ek, wv_path, plz_plot=True)
    #print_purcell_per_set(folder_path, ef=0.25, rad=2.2, fix_fp=True)
    #plot_total_set(folder_path, ef=0.25, rad=2.2)
    #plot_every_set(folder_path)
    #print_every_purcell(folder_path, fix_fp=True)
    #plot_total_set(folder_path, ef=0.15, rad=2.2)
    #plot_z_field_component(E, None, None, None)
    #plot_fp_omega_k_behave(folder_path, ef=0.15, rad=2.2)
    #ef_rad_summary()
    #check_spatial_frequency(E[0])
    cleo_plot(E, folder_path, wv_path)
