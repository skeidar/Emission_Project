import numpy as np
import scipy.spatial as ssp
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from tqdm import tqdm

def complex_3d_norm(x,y,z):
    return np.sqrt(np.dot((x,y,z), np.conj((x,y,z))))

def tetrahedron_volume(a,b,c,d):
    return abs(np.dot(a-d,np.cross(b-d,c-d))) / 6

def triangle_area(a,b,c):
    return abs(np.cross(a-b, c-b)) / 2

def create_func_dict(points, func):
    f = dict()
    z_val_count = 0
    for i,p in enumerate(points):
        x, y, z = p
        if x in f.keys():
            if y in f[x].keys():
                if z in f[x][y].keys():
                    # changed the logic so the func_dict will contain the mean value for each z and not a single value
                    prev_val = f[x][y][z]
                    new_val = ((prev_val * z_val_count) + func[i]) / (z_val_count + 1)
                    z_val_count += 1
                    f[x][y][z] = new_val
                    #pass
                else:
                    f[x][y][z] = func[i]
                    z_val_count = 1
            else:
                f[x][y] = {z: func[i]}
        else:
            f[x] = {y : {z : func[i]}}
    return f   

def create_dict_by_z(points, func):
    points_dict = {}
    for i,p in enumerate(points):
        x, y, z = p
        z_rounded = round_micro_meter(z,4)
        if z_rounded in points_dict.keys():
            points_dict[z_rounded].append((x,y,func[i]))
        else:
            points_dict[z_rounded] = [(x,y,func[i])]
    return points_dict
    

def irregular_integration_delaunay_3d(points, func):
    deb = ssp.Delaunay(points)
    tetrahedrons = deb.points[deb.simplices]

    f_dict = create_func_dict(points, func)

    res = 0
    for tetra in tetrahedrons:
        a = tetra[0,:]
        b = tetra[1,:]
        c = tetra[2,:]
        d = tetra[3,:]
        v = tetrahedron_volume(a, b, c, d)
        l = [a, b, c, d]
        dv = v * sum([f_dict[p[0]][p[1]][p[2]] for p in l]) / 4
        res += dv

    return res

def irregular_integration_delaunay_2d(points_func_tuple):
    points = np.array([np.array([p[0],p[1],0]) for p in points_func_tuple] )
    func = [p[2] for p in points_func_tuple]
    f_dict = create_func_dict(points, func)
    deb = ssp.Delaunay(points[:,:2])
    triangles = deb.points[deb.simplices] 
    
    res = 0
    for tri in triangles:
        a = tri[0,:]
        b = tri[1,:]
        c = tri[2,:]
        area = triangle_area(a, b, c)
        l = [a, b, c]
        res += area * sum([f_dict[p[0]][p[1]][0] for p in l]) / 3
    return res

def regular_integration_1d(samples, z):
    if len(z) > 1:
        delta = z[1] - z[0]
    else:
        delta = 0
    result = sum([samples[i] * delta for i in range(len(samples) -1)]) # -1?
    return result

def interp1d_complex_function(original_range, func, new_range , kind='linear', bounds_error=True, fill_value=0):
    reals = np.real(func)
    imags = np.imag(func)
    real_interp = interpolate.interp1d(original_range, reals, kind=kind, bounds_error=bounds_error, fill_value=fill_value)
    imag_interp = interpolate.interp1d(original_range, imags, kind=kind, bounds_error=bounds_error, fill_value=fill_value)
    interp_func = real_interp(new_range) + 1j * imag_interp(new_range)
    return interp_func

def inner_prodcut_with_field(points, field, z_func, func):
    z_dict = create_dict_by_z(points, field)
    #mean_field = np.array([[zi, np.mean([z_dict[zi][i][2] for i in range(len(z_dict[zi]))])] for zi in z_dict.keys()])
    #interp_field = interp1d_complex_function(np.real(np.array(mean_field[:, 0])), np.array(mean_field[:, 1]), z_func, kind='cubic', bounds_error=False, fill_value=0)
    area = 2.82e-11
    xy_integrations_per_z_slice = np.array(
        [[zi, irregular_integration_delaunay_2d(z_dict[zi]) / np.sqrt(area)] for zi in z_dict.keys()])
    #interp_field = interp1d_complex_function(np.real(xy_integrations_per_z_slice[:, 0]), xy_integrations_per_z_slice[:, 1], z_func,
                                #             kind='cubic', bounds_error=False, fill_value=0)

    # res = regular_integration_1d(interp_field * func, z_func)
    res = regular_integration_1d(func, z_func) / area
    return res

def inner_prodcut_with_field2(p, u, z_func, func, A):
    z_dict = create_dict_by_z(p, u)
    xy_integrations_per_z_slice = np.array([[zi, irregular_integration_delaunay_2d(z_dict[zi]) / A] for zi in z_dict.keys()])
    interp_field = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:, 0]),
                                                     np.array(xy_integrations_per_z_slice[:, 1]), kind='linear',bounds_error=False, fill_value=0)
    return regular_integration_1d(interp_field(z_func) * func, z_func)


def reduce_field_vals(z, field):
    # really not efficient implementation
    used_set = set(z)
    reduced_z = []
    reduced_field = []
    for zi in used_set:
        counter = 0
        sum_field = 0
        for i in range(len(z)):
            if z[i] == zi:
                sum_field = sum_field + field[i]
                counter += 1
        avg_field = sum_field / counter
        reduced_z.append(zi)
        reduced_field.append(avg_field)
    return reduced_z, reduced_field

def round_micro_meter(num, n_digits_after_point):
    return np.round(num * 1e6, n_digits_after_point) * 1e-6

def find_point_idx_on_differnt_z_layer(x, y, z, px, py, layer_z):
        layer_values = []
        for i in range(len(z)):           
            if round_micro_meter(x[i],3) == px and round_micro_meter(y[i],3) == py and z[i] >= layer_z:
                layer_values.append([i, z[i]])
        layer_values = sorted(layer_values, key = lambda x: x[1])
        try:
            idx = layer_values[0][0]
        except IndexError:
            idx = find_point_idx_on_differnt_z_layer(x, y, z, px, py, 0.9999*layer_z)
        return idx

def averaging_comparison(func, points, area, z_func):
    z_dict = create_dict_by_z(points, func)
    ones_dict = create_dict_by_z(points, np.ones(np.shape(func)))
    areas_dict = dict()
    print(np.mean([z_dict[0][i][2] for i in range(len(z_dict[0]))]))
    for zi in z_dict.keys():
        areas_dict[zi] = irregular_integration_delaunay_2d(ones_dict[zi])
    print(irregular_integration_delaunay_2d(z_dict[0]) / areas_dict[0])
    for zi in z_dict.keys():
        for i in range(len(z_dict[zi])):
            plt.plot(z_dict[zi][i][0], z_dict[zi][i][1], '.')
        plt.show()
    mean_field = np.array([[zi, np.mean([z_dict[zi][i][2] for i in range(len(z_dict[zi]))]), irregular_integration_delaunay_2d(z_dict[zi]) / areas_dict[zi]] for zi in z_dict.keys()])
    interp_mean = interp1d_complex_function(np.real(np.array(mean_field[:, 0])), np.array(mean_field[:, 1]), z_func, kind='cubic', bounds_error=False, fill_value=0)
    interp_avg = interp1d_complex_function(np.real(np.array(mean_field[:, 0])), np.array(mean_field[:, 2]), z_func,
                                            kind='cubic', bounds_error=False, fill_value=0)
    plt.plot(abs(mean_field[:, 0]), abs(mean_field[:, 1]), abs(mean_field[:, 0]), abs(mean_field[:, 2]))
    plt.show()

def create_cylinder_grid(x,y,z, resolution=None):
    if resolution is None:
        resolution = 100
    R = max(np.sqrt(x ** 2 + y ** 2))

    zg = np.linspace(z.min(), z.max(), resolution)
    xg = np.linspace(x.min(), x.max(), resolution)
    yg = np.linspace(y.min(), y.max(), resolution)
    xy_grid = list()
    cylinder_grid = list()
    for xi in xg:
        for yi in yg:
            if (xi ** 2 + yi ** 2) <= (R ** 2):

                xy_grid.append(np.array([xi, yi]))
                #plt.plot(xi,yi,'.')
    #plt.show()
    xy_grid = np.array(xy_grid)

    for zi in zg:
        for xi, yi in xy_grid:
            cylinder_grid.append(np.array([xi, yi, zi]))
    cylinder_grid = np.array(cylinder_grid)
    return cylinder_grid



def interp3d_test(points, V, resolution=None):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    grid = create_cylinder_grid(x, y, z, resolution)
    #my_interpolating_function = rgi((x, y, z), V)
    #rbf_interp = interpolate.Rbf(x,y,z,V, function='linear')
    xi = grid[:, 0]
    yi = grid[:, 1]
    zi = grid[:, 2]

    #Vi = my_interpolating_function(array([xi, yi, zi]).T)
    print("Interpolating...")
    Vi = griddata(points, V, grid, method='linear', fill_value=0, rescale=True)
    print("Done interpolating!")
    return Vi, grid

def maximise_duplicates(points, func):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dups_num = 0
    new_points = list()
    new_func = list()
    black_list = list()

    for i in range(len(func)):
        if i not in black_list:
            func_val = func[i]
            for j in range(i+1, len(func)):
                if x[i] == points[j][0] and y[i] == points[j][1] and z[i] == points[j][2]:
                    if round(func[i].real, 3) != round(func[j].real, 3) or round(func[i].imag, 3) != round(func[j].imag, 3):
                        max_idx = np.argmax([abs(func[j]), abs(func[i])])
                        func_val = func[j + max_idx * (i - j)]
                    black_list.append(j)
                    dups_num += 1
            new_points.append(np.array([x[i], y[i], z[i]]))
            new_func.append(func_val)

    if dups_num == 0:
        return points, func
    new_points = np.array(new_points)
    new_func = np.array(new_func)

    return maximise_duplicates(new_points, new_func)