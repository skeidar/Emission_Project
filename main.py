from field_parsing import *
from wavefunc import load_wavefunction, normalize_wv
from matplotlib import pyplot as plt

if __name__ == "__main__":
    file_name = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\test.npy"
    path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data"
    wv_path = r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0"
    wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_APL_2019\2-well_APL_2019"
    f_array = np.linspace(3e12, 4.8e12, 301)
    f_array = np.linspace(3e12, 4.8e12, 11)

    E = load_modes(r"C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um")
    x = E[1].points[:, 0]
    y = E[1].points[:, 1]
    z = E[1].points[:, 2]
    u_z = E[1].e_field[:, 2]
    z_linspace = np.linspace(z.min(), z.max(), 10000)
    #averaging_comparison(u_z, E[1].points, E[1].disk_area, z_linspace)
    #E[1].approximations_comparison(wv_path)
    randush = np.random.rand(np.shape(u_z)[0]) + 1j * np.random.rand(np.shape(u_z)[0]) * 1e-5
    print("Maximizing...")
    #new_p, new_u_z = maximise_duplicates(E[1].points, u_z)
    print("Done maximizing!")

    new_p = E[1].points
    new_u_z = np.real(u_z)
    u_z_phase = np.angle(u_z)
    u_z_abs = abs(u_z)
    V1, grid = interp3d_test(new_p, new_u_z, 41)




    #V2, _ = interp3d_test(E[1].points, np.imag(u_z), 41)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the values
    ax.scatter(grid[:,0], grid[:,1], grid[:,2], c='b', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()
    """
    plt.plot(grid[:, 2], V1, '.')
    plt.plot(z, new_u_z,'.')

    plt.legend(['interpolated','original'])
    plt.show()

    interp_integral = irregular_integration_delaunay_3d(grid, abs(V1))
    original_integral = irregular_integration_delaunay_3d(new_p, abs(new_u_z))

    print(interp_integral)
    print(original_integral)
    # need to check the percentage diff - maybe 51 points are enough
    print("Error [%]: {}".format((1 - interp_integral / original_integral) * 100)) # error percentage






