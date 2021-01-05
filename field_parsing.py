import scipy.io as sio
from scipy import interpolate
#import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from utilities import *
import csv
from tqdm import tqdm

class ElectricMode(object):
    def __init__(self, path, ef, radius, frequency, Q):
        self.ef = np.round(float(ef) * 6.24150913* 1e18,3)
        self.radius = float(radius)
        self.frequency = float(frequency)
        frequency_str = str(np.round(float(frequency[:-3]),2))
        if frequency_str[-2] == '.':
            frequency_str += '0'
        elif '.' not in frequency_str:
            frequency_str += '.00'
        self.Q = float(Q)
        self.freq_str = frequency_str
        mat_path = path + frequency_str + "THz-Ecomp.mat"
        self._mat = sio.loadmat(mat_path)
        self.samples = self._mat['E']
        # multiply by 1e-6 becaues the data is given as um
        self.points = np.array([np.real([sample[0], sample[1], sample[2]]) for sample in self.samples]) * 1e-6
        self.e_field = np.array([np.array([sample[3], sample[4], sample[5]]) for sample in self.samples]) * 1e-6
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])
        self.gamma_ratio = None

    def compute_norm(self):
        return irregular_integration_delaunay_3d(self.points, (self.e_norms) ** 2)

    def normalize(self):
        self.e_field = self.e_field / (self.compute_norm()) ** 0.5
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])

    def scatter_plot_3d(self, func_array=None, func_name='$\Gamma_m(r,\omega_m)/\Gamma_0$ [1]', as_log=True):
        if func_array is None:
            func_array = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency)
        d = {'X': self.points[:,0], 'Y': self.points[:,1], 'Z': self.points[:,2], func_name: func_array}
        df = pd.DataFrame.from_dict(d)
        if not as_log:
            fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=func_name,)
            fig.update_layout(title_text="f={}THz".format(self.freq_str), title_x=0.5)
            fig.show()
        else:
            log_field_name = '$log_{10}$'+'('+func_name+')'
            df[log_field_name] = df[func_name].apply(np.log10)
            fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=log_field_name,title="f={}THz".format(self.freq_str))
            fig.update_layout(title_text="f={}THz".format(self.freq_str), title_x=0.5)
            fig.show()

    def plot_2d_z_rho_slice(self, scatter=False):
        """
        plots the ratio of gammas at for z and rho axis and omega = omega_m
        """
        z_values = self.points[:,2]
        rho_values = np.sqrt(self.points[:,0] ** 2 + self.points[:,1] ** 2)
        if self.gamma_ratio is None:
            # add mode's Q 
            gamma = gamma_m(self.e_field[:,2], 10.28, self.frequency, self.frequency)
            self.gamma_ratio = gamma
        else:
            # add mode's Q 
            gamma = self.gamma_ratio
        if scatter:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            plt1 = ax1.scatter(rho_values, z_values, c=np.log10(gamma), cmap=plt.cm.plasma)
            ax1.title.set_text(r'Scatter plot of $\rho$-z slice')
            cbar1 = fig.colorbar(plt1, ax=ax1)
            cbar1.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            ax1.set_xlabel(r"$\rho$ [m]")
            ax1.set_ylabel(r"z [m]")

            ax2 = fig.add_subplot(212)
            plt2= ax2.tripcolor(rho_values,z_values,np.log10(gamma), cmap=plt.cm.plasma, shading='gouraud')
            ax2.title.set_text('Emission Enhancement')
            cbar2 = fig.colorbar(plt2, ax=ax2)
            cbar2.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            ax2.set_xlabel(r"$\rho$ [m]")
            ax2.set_ylabel(r"z [m]")

            plt.subplots_adjust(hspace=0.3)
        else:
            fig = plt.figure(figsize = (10,5))
            plt.tripcolor(rho_values,z_values,np.log10(gamma), cmap=plt.cm.plasma, shading='gouraud')
            plt.title('Emission Enhancement')
            cbar = plt.colorbar()
            cbar.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            plt.xlabel(r"$\rho$ [m]")
            plt.ylabel(r"z [m]")
        plt.show()

    def plot_2d_z_x_slice(self, scatter=False):
        """
        plots the ratio of gammas at for z and rho axis and omega = omega_m
        """
        z_values = self.points[:,2]
        x_values = self.points[:,0]
        if self.gamma_ratio is None:
            # add mode's Q 
            gamma = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency)
            self.gamma_ratio = gamma
        else:
            # add mode's Q 
            gamma = self.gamma_ratio

        if scatter:
            fig = plt.figure(figsize = (7,10))
            ax1 = fig.add_subplot(211)
            plt1 = ax1.scatter(x_values, z_values, c=np.log10(gamma), cmap=plt.cm.plasma)
            ax1.title.set_text(r'Scatter plot of x-z slice')
            cbar1 = fig.colorbar(plt1, ax=ax1)
            cbar1.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            ax1.set_xlabel(r"x [m]")
            ax1.set_ylabel(r"z [m]")

            ax2 = fig.add_subplot(212)
            plt2= ax2.tripcolor(x_values,z_values,np.log10(gamma), cmap=plt.cm.plasma, shading='gouraud')
            ax2.title.set_text('Emission Enhancement')
            cbar2 = fig.colorbar(plt2, ax=ax2)
            cbar2.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            ax2.set_xlabel(r"x [m]")
            ax2.set_ylabel(r"z [m]")

            plt.subplots_adjust(hspace=0.3)
        else:
            fig = plt.figure(figsize = (10,5))
            plt.tripcolor(x_values,z_values,np.log10(gamma), cmap=plt.cm.plasma, shading='gouraud')
            plt.title('Emission Enhancement')
            cbar = plt.colorbar()
            cbar.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
            plt.xlabel(r"x [m]")
            plt.ylabel(r"z [m]")
        plt.show()

    def plot_gamma_vs_freq_along_z(self):
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]
        L = z.max() - z.min()
        rho = np.sqrt(self.points[:,0] ** 2 + self.points[:,1] ** 2)
        f = np.linspace(self.frequency * 0.7, self.frequency * 1.3, 1000)
        gamma_ratio = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency)
        maximum_idx = np.argmax(gamma_ratio)
        x_max = round_micro_meter(x[maximum_idx],3)
        y_max = round_micro_meter(y[maximum_idx],3)

        # some points appear more than once with different values of field. 
        # it also happens with z=0.36000000001 and z=0.36, and their values could differ by a lot.
        # so i plot the maximal value between the two
        """
        if np.round(z[maximum_idx] * 1e1, 3) * 1e-1 == np.round(L * 1e1, 3) * 1e-1:
            idx_1L = maximum_idx
        else:
            idx_1L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, z.max())
        """

        idx_1L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, z.max())
        idx_075L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, 3 * L/4)
        idx_05L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, L/2)
        idx_025L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, L/4)
        idx_0L = find_point_idx_on_differnt_z_layer(x, y, z, x_max, y_max, 0)

        u_z_max = self.e_field[:,2][maximum_idx]
        u_z_1L = self.e_field[:,2][idx_1L]
        u_z_075L = self.e_field[:,2][idx_075L]
        u_z_05L = self.e_field[:,2][idx_05L]
        u_z_025L = self.e_field[:,2][idx_025L]
        u_z_0L = self.e_field[:,2][idx_0L]

        gamma_ratio_max = np.log10(gamma_m(u_z_max, self.Q, f, self.frequency))
        gamma_ratio_1L = np.log10(gamma_m(u_z_1L, self.Q, f, self.frequency))
        gamma_ratio_075L = np.log10(gamma_m(u_z_075L, self.Q, f, self.frequency))
        gamma_ratio_05L = np.log10(gamma_m(u_z_05L, self.Q, f, self.frequency))
        gamma_ratio_025L = np.log10(gamma_m(u_z_025L, self.Q, f, self.frequency))
        gamma_ratio_0L = np.log10(gamma_m(u_z_0L, self.Q, f, self.frequency))

        chosen_rhos = np.array([rho[idx_0L], rho[idx_025L], rho[idx_05L], rho[idx_075L], rho[idx_1L]])
        chosen_zs = np.array([z[idx_0L], z[idx_025L], z[idx_05L], z[idx_075L], z[idx_1L]])

        #plt.rcParams['figure.figsize'] = [10, 5]
        fig = plt.figure(figsize=(20,5))
        ax1 = fig.add_subplot(121)
        plt1 = ax1.plot(f, gamma_ratio_0L, 'b', f, gamma_ratio_025L, 'r', f, gamma_ratio_05L, 'g', f, gamma_ratio_075L, 'k', f, gamma_ratio_1L, 'c', f, gamma_ratio_max, ',c')
        ax1.title.set_text(r'Mode strength along the z-axis at the radius with maximal emission')
        ax1.legend(['z = 0L', 'z = 0.25L', 'z = 0.5L', 'z = 0.75L', 'z = L', 'z = L (maxmimal emission)'])
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel(r"$log_{10}(\Gamma_m(r,\omega)/\Gamma_0)$ [1]")

        ax2 = fig.add_subplot(122)
        plt2= ax2.tripcolor(rho,z,np.log10(gamma_ratio), cmap=plt.cm.plasma, shading='gouraud')
        ax2.scatter(chosen_rhos, chosen_zs, facecolor=['blue','red','green','black','cyan'])
        ax2.title.set_text('Emission Enhancement $\Gamma_m/\Gamma_0$')
        cbar2 = fig.colorbar(plt2, ax=ax2)
        cbar2.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]')
        ax2.set_xlabel(r"$\rho$ [m]")
        ax2.set_ylabel(r"z [m]")

        plt.subplots_adjust(hspace=0.3)
        plt.show()

    def calculate_effective_rate(self, z, u_z, f):
        # step 1 - 2d integration over xy to obtain gamma(z). 
        #        step 1a - collecting every xy values in a certain z (z is rounded)
        #        step 1b - integration for every z using triangulation
        # step 2 - interpolation over z
        # step 3 - 1d integration over gamma(z)*Y(z)dz

        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(),4), 10000)
        dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, self.points)

        gamma_func = gamma_m(u_z,  self.Q, f, self.frequency)
        z_dict = create_dict_by_z(self.points, gamma_func)

        xy_integrations_per_z_slice =  np.array([[z, irregular_integration_delaunay_2d(z_dict[z])] for z in z_dict.keys()])
        interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:,0]), np.array(xy_integrations_per_z_slice[:,1]), kind='cubic')

        interpolated_z_gamma = interpolated_z_gamma_func(z_linspace)
        gamma_effective = regular_integration_1d(dipole_density * interpolated_z_gamma, z_linspace)

        return gamma_effective

    def get_effective_emission(self, file_name, f=None):
        if f is None:
            f = np.linspace(self.frequency * 0.8, self.frequency * 1.2, 41)

        # qe = 1.6e-19
        # d1_strength = qe * 2.74e-9
        # d2_strength = qe * 2.98e-9
        # dipoles = [d1_strength, d2_strength]
        try:
            gammas = np.load(file_name)
        except:
            z = self.points[:, 2]
            u_z = self.e_field[:, 2]
            gammas = np.array([self.calculate_effective_rate(z, u_z, freq) for freq in f])
            print("Didn't find the file - saving it")
            np.save(file_name, gammas)

        plt.figure()
        plt.title('Effective Emission vs Frequency')
        plt.xlabel('Frequency [THz]')
        plt.ylabel('$log_{10}(\Gamma_m(r,\omega_m)/\epsilon_r)$ [1]')
        plt.plot(f / 1e12, np.log10(gammas))
        plt.show()
        #plt.legend(['$d_1 = 2.74nm * q_e$', '$d_2 = 2.98nm * q_e$'])

    def save_mode_emission(self, file_name, f_array=None):
        if f_array is None:
            f = np.linspace(self.frequency * 0.8, self.frequency * 1.2, 41)
        else:
            f = f_array
        z = self.points[:, 2]
        u_z = self.e_field[:, 2]
        gammas = np.array([self.calculate_effective_rate(z, u_z, f[i]) for i in tqdm(range(len(f)))])
        np.save(file_name, gammas)


def load_modes(folder_path, mode_frequeny_THz=None):
    DATA_START = 5
    EF = 0
    RADIUS = 1
    FREQUENCY = 3
    Q = 4

    if folder_path[-1] != '\\':
        folder_path += '\\'
    q_file = open(folder_path + "Q-tot.txt", 'r')
    q_tot = q_file.read().splitlines()
    modes = None

    if mode_frequeny_THz is None:
        # opens a whole folder
        modes = [ElectricMode(folder_path, mode.split()[EF], mode.split()[RADIUS], mode.split()[FREQUENCY], mode.split()[Q]) for mode in q_tot[DATA_START:]]
    else:
        # a single mode frequency was given
        for mode in q_tot[DATA_START:]:
            if np.round(float(mode.split()[FREQUENCY]) / 1e12, 2) == mode_frequeny_THz:
                modes = ElectricMode(folder_path, mode.split()[EF], mode.split()[RADIUS], mode.split()[FREQUENCY], mode.split()[Q])
                break
        try:
            modes
        except:
            raise ValueError("Can't find a mode with a frequency of {}THz".format(mode_frequeny_THz))
    return modes


def dipole_locations(z_linspace):
        LAYER1 = 8.28e-9
        LAYER2 = 1.97e-9
        LAYER3 = 8.7e-9
        dipole_width = 3e-9
        LAYER4 = 5.35e-9
        LAYER5 = 3.38e-9
        # sum * 12 is larger than the device's z-lenght
        TOTAL_LAYERS = LAYER1  + LAYER2 + LAYER3 + LAYER4 + LAYER5 + dipole_width
        dipole_layer_offset =  z_linspace.max() - (LAYER1  + LAYER2 + LAYER3)
        d = np.zeros(np.shape(z_linspace))

        count = int(round((z_linspace.max() - z_linspace.min()) / (TOTAL_LAYERS)))
        for k in range(count):
            # assuming the graphene is at z=0.36um, we know where layer1 begins
            # so the layer stacking starts from there
            for i,zi in enumerate(z_linspace[::-1]):
                dipole_begin = dipole_layer_offset - TOTAL_LAYERS * k
                dipole_end = dipole_begin - dipole_width
                if zi < dipole_begin and zi > dipole_end:
                    d[i] = 1
        d = d[::-1]
        return d


def generate_dipole_density(d, z_linspace, points):
        # integrate to obtain Y(r)
        ones_array = create_dict_by_z(points, [1] * len(points)) # normalization
        xy_areas = [irregular_integration_delaunay_2d(ones_array[z]) for z in ones_array.keys()]
        A = np.mean(xy_areas)
        # the dipole normalized density isn't affected by the dipole strength
        dipole_density = d / (regular_integration_1d(d, z_linspace) * A)
        return dipole_density


def Gamma_m(u, d, Q, f, f_m):
    hbar = 1.0545711818e-34
    eps_0 = 8.85418781762039e-12
    eps_r = 3.5 # typical value
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    res = (2 * (abs(d) ** 2) * (abs(u) ** 2) * 2 * Q * w_m * w) / ((hbar * eps_0 * eps_r * np.pi) * (4 * ((Q * (w - w_m)) ** 2) + w_m ** 2))
    return res


def gamma_m(u, Q, f, f_m):
    # the relation between Gamma_m / Gamma_0
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    c = 2.997e8
    res = 12 * (c ** 3) * Q * w_m * (abs(u) ** 2) / ((w ** 2) * (4 * (Q ** 2) * ((w - w_m) ** 2) + (w_m ** 2)))
    return res


def save_effective_emissions(electric_modes, path, f_array):
    for i, ei in enumerate(electric_modes):
        print("mode number {}".format(i))
        ei.normalize()
        file_name = path + r"\gamma_ratios_ef_{}_r_{}_f_{}.npy".format(str(ei.ef), str(ei.radius), str(ei.freq_str))
        ei.save_mode_emission(file_name, f_array)

def create_total_emission_graph(electric_modes, path, f):
    total_emission = np.array([])
    dipole_path = path + r"\dipole_spectrum.csv"
    for ei in electric_modes:
        file_name = path + r"\gamma_ratios_ef_{}_r_{}_f_{}.npy".format(str(ei.ef), str(ei.radius), str(ei.freq_str))
        try:
            emission = np.load(file_name)
            dipoles = load_dipole_spectrum(dipole_path, f)
            print("Mode - {}THz - Purcell Factor = {}".format(ei.freq_str, calculate_purcell_factor(emission, dipoles, f)))
        except:
            "Couldn't load the emission"
            emission = 0
        if len(total_emission) == 0:
            total_emission = np.zeros(len(emission))
        total_emission += emission
    return total_emission


def load_dipole_spectrum(path, f):
    with open(path) as csvfile:
        dipole = csv.reader(csvfile)
        dipole = np.array(list(dipole))
    dip_f = dipole[:,0]
    dip_f = np.array([float(d) * 1e12 for d in dip_f])
    dip_spec = dipole[:,1]
    dip_spec = np.array([float(d) for d in dip_spec])
    extrapolated_dip_func = interpolate.interp1d(dip_f, dip_spec, bounds_error=False, fill_value=0) #fill_value="extrapolate")
    extrapolated_dip = extrapolated_dip_func(f)
    return extrapolated_dip

def plot_emission_spectrum(electric_modes, path, f):
    dipole_path = path + r"\dipole_spectrum.csv"
    total_emission = create_total_emission_graph(electric_modes, path, f)
    dipoles = load_dipole_spectrum(dipole_path, f)
    central_freqs = np.array([Ei.frequency for Ei in electric_modes])
    plt.figure()
    plt.plot(f / 1e12, total_emission, f / 1e12, dipoles, 'm')
    for cf in central_freqs:
        plt.axvline(x=cf / 1e12, linestyle=':', linewidth=0.5, c='k')
    plt.title('Emission Spectrum for Ef={}eV, R={}m'.format(electric_modes[0].ef, electric_modes[0].radius) +' with $P_F^{eff}$' + '= {}'.format(calculate_purcell_factor(total_emission, dipoles, f)))
    plt.ylabel('Spectrum [a.u]')
    plt.xlabel('Frequnecy [THz]')
    plt.legend(["$\Gamma(\omega)/\Gamma_0(\omega)$ with graphene", "Spectrum without graphene", "Modes frequencies"])
    #plt.figure()

    #plt.title(r'Structure Emission for Ef={}eV, R={}m'.format(electric_modes[0].ef, electric_modes[0].radius))
    #plt.plot(f / 1e12, total_emission)

    #plt.xlabel('Frequency [THz]')
    #lt.ylabel('Spectrum [a.u]')
    #plt.legend(["Total emission with graphene", "Modes frequencies"])
    plt.show()

def calculate_purcell_factor(total_emission, dipole_emission, f):
    normalized_dipole_spect = np.array(dipole_emission / regular_integration_1d(dipole_emission, f))
    return regular_integration_1d(normalized_dipole_spect * total_emission, f)

