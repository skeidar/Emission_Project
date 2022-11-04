from field_parsing import *


def interpolation_SNR_test(points):
    x, y, z = points.T
    Lx = abs(x.max() - x.min())
    Lz = abs(z.max() - z.min())
    K = np.array([2 ** i for i in range(-4, 4)]) * 2 * np.pi / Lx
    #resolutions = np.linspace(20, 60, 9)
    resolutions = np.linspace(20, 50, 7)
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
    print(x_snr)
