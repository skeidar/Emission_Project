from field_parsing import *

if __name__ == "__main__":
    file_name = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\test.npy"
    path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data"
    f_array = np.linspace(3e12, 4.8e12, 301)

    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.2um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_3.0um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV-2.2um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV-2.6um")
    save_effective_emissions(E, path, f_array)
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um")
    save_effective_emissions(E, path, f_array)


    #save_effective_emissions(E, path, f_array)

    #create_total_emission_graph(E, path, f_array)
    #E[1].scatter_plot_3d()
    #plot_emission_spectrum(E, path, f_array)
    #generate_purcell_heatmap()

