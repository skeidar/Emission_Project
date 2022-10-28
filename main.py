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


    Ek = E[0]
    points = Ek.points

    div_test = np.load(r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\11_02_22_divergences\Ef_0.25eV_rad_2.2e-06um_f_3.81THz_RES_28_DIV.npy")
    my_grid = np.load(r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\11_02_22_divergences\Ef_0.25eV_rad_2.2e-06um_f_3.81THz_RES_28_grid.npy")
    madafaka = np.zeros([len(div_test), 4])

    #madafaka[:, 3] = np.real(div_test)
    #madafaka[:, :3] = my_grid

    #generic_scattr_3d(my_grid, np.imag(div_test), False)
    Ek = E[0]
    points = Ek.points
    field_k = Ek.e_field
    print(np.shape(points))
    interp_field = []
    x, y, z = points.T
    NP = 5
    test = np.cos(NP * y * np.pi / (y.max() - y.min())) * x

    for i in range(3):
        interp_f, grid = interp3d(points, field_k[:, i], 19)
        #interp_f, grid = interp3d(points, test, 30)
        interp_field.append(interp_f)
    interp_field = np.array(interp_field)

    grad_list = regular_grid_grads(interp_field, grid)
    div = sum(grad_list)
    E[0].e_norms = np.real(div)
    E[0].points = grid
    E[0].Epolar_Ez_plot()
    """
    g = grad_express(interp_field, grid, 0)

    generic_scatter_3d(grid, g, False)


    #E[0].points = my_grid
    #E[0].e_norms = np.real(div_test)

    #E[0].Epolar_Ez_plot()
    """