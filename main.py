from field_parsing import *
from wavefunc import load_wavefunction, normalize_wv
from matplotlib import pyplot as plt


if __name__ == "__main__":
    file_name = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\test.npy"
    csv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data"
    wv_path = r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0"
    wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_APL_2019\2-well_APL_2019"
    f_array = np.linspace(3e12, 4.8e12, 301)
    #f_array = np.linspace(3e12, 4.8e12, 11)


    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um")
    #x = E[0].points[:, 0]
    #y = E[0].points[:, 1]
    z = E[1].points[:, 2]
    u_z = E[1].e_field[:, 2]
    z_linspace = np.linspace(z.min(), z.max(), 10000)
    #for Ei in E:
        #name = "Ef_" + str(Ei.ef) + "_r_" + str(np.round(Ei.radius * 1e6, 2)) + 'um_f_' + Ei.freq_str + 'THz'
        #Ei.approximations_comparison(wv_path, name, save_log=True)
    #show_rate_comparison(r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_11_21')
    run_emission_spectrum(csv_path, f_array, plot=True, verbose=False)
    #show_rate_for_QW(r'C:\Shaked\Technion\QCL_Project\logs\fixed_18_11_21')



