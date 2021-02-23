import numpy as np
import scipy.spatial as ssp

def complex_3d_norm(x,y,z):
    return np.sqrt(np.dot((x,y,z), np.conj((x,y,z))))

def tetrahedron_volume(a,b,c,d):
    return abs(np.dot(a-d,np.cross(b-d,c-d))) / 6

def triangle_area(a,b,c):
    return abs(np.cross(a-b, c-b)) / 2

def create_func_dict(points, func):
    f = dict() 
    for i,p in enumerate(points):
        x, y, z = p
        if x in f.keys():
            if y in f[x].keys():
                if z in f[x][y].keys():
                    pass
                else:
                    f[x][y][z] = func[i]
            else:
                f[x][y] = {z: func[i]}
        else:
            f[x] = {y : {z : func[i]}}
    return f   

def create_dict_by_z(points, func):
    points_dict = {}
    for i,p in enumerate(points):
        x, y, z = p
        if round_micro_meter(z,4) in points_dict.keys():
            points_dict[round_micro_meter(z,4)].append((x,y,func[i]))
        else:
            points_dict[round_micro_meter(z,4)] = [(x,y,func[i])]
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
        res += v * sum([f_dict[p[0]][p[1]][p[2]] for p in l]) / 4

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

