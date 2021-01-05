from field_parsing import *

if __name__ == "__main__":
    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um")
    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV-2.6um")
    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV-2.2um")

    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um")
    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um")

    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.2um")
    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um")
    #E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_3.0um")

    file_name = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\test.npy"
    path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data"

    f_array = np.linspace(3e12, 4.8e12, 301)
    #save_effective_emissions(E, path, f_array)
    #total_emission = create_total_emission_graph(E, path)

    #plt.plot(f_array, total_emission)
    #plt.show()

    #create_total_emission_graph(E, path, f_array)
    plot_emission_spectrum(E, path, f_array)
    #for Ei in E:
        #Ei.normalize()
        #Ei.scatter_plot_3d()

