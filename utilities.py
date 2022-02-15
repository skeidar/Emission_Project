import numpy as np
import scipy.spatial as ssp
from scipy import interpolate
from scipy import constants as cn
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import plotly.express as px
import pandas as pd
import time

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
    
def cyclinder_grid_3d_integration(points, func):
    func_z_dict = create_dict_by_z(points,func)
    area_list = list()
    for zi in func_z_dict.keys():
        xy_area = irregular_integration_delaunay_2d(func_z_dict[zi])
        area_list.append([xy_area, zi])
    res = 0
    for i in range(len(area_list)):
        if i == len(area_list) - 1:
            break
        delta_z = area_list[i+1][1] - area_list[i][1]
        res += delta_z * area_list[i][0]
    return res


def irregular_integration_delaunay_3d(points, func, plot_tetras=False):
    deb = ssp.Delaunay(points)
    tetrahedrons = deb.points[deb.simplices]

    f_dict = create_func_dict(points, func)
    zero_volume_count = 0
    res = 0

    for k, tetra in enumerate(tetrahedrons):
        a = tetra[0,:]
        b = tetra[1,:]
        c = tetra[2,:]
        d = tetra[3,:]
        v = tetrahedron_volume(a, b, c, d)
        l = [a, b, c, d]
        dv = v * sum([f_dict[p[0]][p[1]][p[2]] for p in l]) / 4

        res += dv
        if v == 0:
            zero_volume_count +=1
        if plot_tetras:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            # plotting the six edges of the tetrahedron
            for ij in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
               ax.plot3D(tetra[ij, 0], tetra[ij, 1], tetra[ij, 2])
            #print(tetra[:,0],tetra[:,1],tetra[:,2])

            plt.show()



    #print("Zero volumes % = {}".format(zero_volume_count / len(tetrahedrons) * 100))

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

def inner_product_squared(points, field, z_func, func, area):

    Vi, grid = interp3d(points, field, resolution=25)
    f_dict = create_func_dict(grid, Vi)

    # plotting the grid:
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(grid[:,0], grid[:,1], grid[:,2])
    #plt.show()

    z = grid[:, 2]
    squared_xy_terms = []
    for x,y in tqdm(zip(grid[:,0],grid[:,1]), total=len(z)):
        xy_func = np.array([f_dict[x][y][zi] for zi in z])
        u_z_interp = interpolate.interp1d(z, xy_func, kind='linear', bounds_error=False, fill_value=0)
        interp_field = u_z_interp(z_func)
        squared_xy_terms.append((x, y, abs(regular_integration_1d(func * interp_field, z_func)) ** 2))
    res = irregular_integration_delaunay_2d(squared_xy_terms) / area
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



def interp3d(points, V, resolution=None):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    grid = create_cylinder_grid(x, y, z, resolution)
    #grid[:,2] = grid[:,2] * 10
    print("Interpolating...")
    Vi = griddata(points, V, grid, method='linear', fill_value=0, rescale=True)
    print("Done interpolating!")
    grid[:, 2] = grid[:, 2]
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

def generic_scatter_3d(points, func_array=None, as_log=True):
    func_name = 'Func'
    d = {'X': points[:,0], 'Y': points[:,1], 'Z': points[:,2], func_name: func_array}
    df = pd.DataFrame.from_dict(d)
    if not as_log:
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=func_name)
        fig.show()
    else:
        log_field_name = 'log'
        df[log_field_name] = df[func_name].apply(np.log10)
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=log_field_name)
        fig.show()

def freq2energy(frequnecy):
    # energy in eV, frequency in Hz
    return frequnecy * cn.h / cn.e

def energy2freq(energy):
    # energy in eV, frequency in Hz
    return cn.e * energy / cn.h

def round_scaleless(num, order):
    if 0 < abs(num) < 1:
        return np.round(num, int(abs(np.log10(abs(num)))) + order)
    elif num !=0:
        return np.round(num, order)
    else:
        return 0


def multi_run_wrapper(args):
    return grad(*args)


def regular_grid_grads(vec, grid):
    if np.shape(vec.T) == np.shape(grid):
        vec = vec.T
    print(np.shape(vec), np.shape(grid))
    with Pool(processes=3) as p:
        # grad_list = p.map(worker, range(5))
        #grad_list = sum(np.array(p.map(multi_run_wrapper, [(vec[:,ax], grid, ax) for ax in range(3)])))
        #grad_list = sum(np.array(p.starmap(grad, [(vec[:, ax], grid, ax) for ax in range(3)])))
        grad_list = np.array(p.starmap(grad, [(vec[:, ax], grid, ax) for ax in range(3)]))
    return np.array(grad_list)

def grad(scalar_field, points, axis):
    """
    delta = 2022  # a large arbitrary number, points scale is micrometer
        grad_list = []
        dist_set = set()
        for i in range(len(points)):
            dist = abs(points[0,axis] - points[i,axis])
            dist_set.add(dist)
            if delta > dist > 0:
                delta = dist
        print("working on axis {}".format(axis))
        for i in tqdm(range(len(points))):
            p = points[i,:]
            closest_positive_idx = -1
            closest_negative_idx = -1
            adjacent_points_idx = -1
            min_positive_dist = 2022
            for j,c in enumerate(points):
                if j == i:
                    continue
                if round_scaleless(c[(axis+1)%3],3) == round_scaleless(p[(axis+1)%3],3) and round_scaleless(c[(axis+2)%3],3) == round_scaleless(p[(axis+2)%3],3):
                    current_positive_dist = c[axis] - p[axis]
                    if min_dist > current_positive_dist > 0:
                        min_dist = current_positive_dist
                        closest_positive_idx = j
            if closest_positive_idx != -1:
                grad_list.append((scalar_field[closest_positive_idx] - scalar_field[i]) / delta)
            else:
                #probably happens on boundarys
                for j,c in enumerate(points):
                    if j == i:
                        continue
                    if round_scaleless(c[(axis+1)%3],3) == round_scaleless(p[(axis+1)%3],3) and round_scaleless(c[(axis+2)%3],3) == round_scaleless(p[(axis+2)%3],3):
                        closest_negative_idx = j
                        break
                if closest_negative_idx != -1:
                    grad_list.append((scalar_field[i] - scalar_field[closest_negative_idx]) / delta)
                # to add: what happens at the tips? there could be no points before and after
                else:
                    if axis == 2:
                        continue
                    else:
                        for j, c in enumerate(points):
                            if j == i:
                                continue
                            if round_scaleless(c[axis], 3) == round_scaleless(p[axis], 3) and (round_scaleless(
                                    c[(axis + 1) % 2], 3) == round_scaleless(p[(axis + 1) % 2] - delta, 3)
                                    or round_scaleless(c[(axis + 1) % 2], 3) == round_scaleless(
                                    p[(axis + 1) % 2] + delta, 3)) and round_scaleless(c[2], 3) == round_scaleless(p[2], 3):
                                adjacent_points_idx = j
                                break
                        if adjacent_points_idx != -1:
                            grad_list.append(scalar_field[adjacent_points_idx])
                        else:
                            grad_list.append(float('nan'))
        return grad_list

    """
    delta = 2022  # a large arbitrary number, points scale is micrometer
    grad_list = []
    dist_set = set()
    for i in range(len(points)):
        dist = abs(points[0,axis] - points[i,axis])
        dist_set.add(dist)
        if delta > dist > 0:
            delta = dist
    print("working on axis {}".format(axis))
    for i in tqdm(range(len(points))):
        p = points[i,:]
        closest_point_idx = -1
        prev_points_idx = -1
        adjacent_points_idx = -1
        for j,c in enumerate(points):
            if j == i:
                continue
            if round_scaleless(c[axis] ,3) == round_scaleless(p[axis] + delta, 3) and round_scaleless(c[(axis+1)%3],3) == round_scaleless(p[(axis+1)%3],3) and round_scaleless(c[(axis+2)%3],3) == round_scaleless(p[(axis+2)%3],3):
            #if c[axis] == (p[axis] + delta) and c[(axis+1)%3] == p[(axis+1)%3] and c[(axis+2)%3]== p[(axis+2)%3]:
                closest_point_idx = j
                break
        if closest_point_idx != -1:
            grad_list.append((scalar_field[closest_point_idx] - scalar_field[i]) / delta)
        else:
            #probably happens on boundarys
            for j,c in enumerate(points):
                if j == i:
                    continue
                if round_scaleless(c[axis] ,3) == round_scaleless(p[axis] - delta, 3) and round_scaleless(c[(axis+1)%3],3) == round_scaleless(p[(axis+1)%3],3) and round_scaleless(c[(axis+2)%3],3) == round_scaleless(p[(axis+2)%3],3):
                    prev_points_idx = j
                    break
            if prev_points_idx != -1:
                grad_list.append((scalar_field[i] - scalar_field[prev_points_idx]) / delta)
            # to add: what happens at the tips? there could be no points before and after
            else:
                if axis == 2:
                    continue
                else:
                    for j, c in enumerate(points):
                        if j == i:
                            continue
                        if round_scaleless(c[axis], 3) == round_scaleless(p[axis], 3) and (round_scaleless(
                                c[(axis + 1) % 2], 3) == round_scaleless(p[(axis + 1) % 2] - delta, 3)
                                or round_scaleless(c[(axis + 1) % 2], 3) == round_scaleless(
                                p[(axis + 1) % 2] + delta, 3)) and round_scaleless(c[2], 3) == round_scaleless(p[2], 3):
                            adjacent_points_idx = j
                            break
                    if adjacent_points_idx != -1:
                        grad_list.append(scalar_field[adjacent_points_idx])
                    else:
                        grad_list.append(float('nan'))
    return grad_list

