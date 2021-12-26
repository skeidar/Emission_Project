import scipy.io as sio
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy import constants as cn
import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from utilities import *
from wavefunc import load_wavefunction, normalize_wv
import csv
import os
from tqdm import tqdm

class ElectricMode(object):
    def __init__(self, path, ef, radius, frequency, Q):
        self.ef = np.round(float(ef) * 6.24150913 * 1e18,3)
        self.radius = float(radius)
        self.disk_area = np.pi * (self.radius ** 2)
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
        self._eps = np.load(r"C:\Shaked\Technion\QCL_Project\Wavefunctions\structure_epsilon.npy", allow_pickle=True)[()]
        self.bulk_eps_r = 1
        self.active_eps_r = 1
        self.d_bulk_eps_r = 1
        self.d_active_eps_r = 1


    def compute_norm(self):
        return irregular_integration_delaunay_3d(self.points, (self.e_norms) ** 2)

    def naive_normalize(self):
        self.e_field = self.e_field / (self.compute_norm()) ** 0.5
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])

    def use_disperssion(self, energy):
        # calcualting the effective epsilon and derivative of epsilon:
        mode_energy = cn.physical_constants['Planck constant in eV/Hz'][0] * self.frequency
        bulk_dispersion = self._eps['bulk_eps']
        active_dispersion = self._eps['total_eps']
        energy = self._eps['omega']
        omega = energy / cn.physical_constants['reduced Planck constant in eV s'][0]
        bulk_interp = interpolate.interp1d(energy, bulk_dispersion)
        active_interp = interpolate.interp1d(energy, active_dispersion)
        bulk_eps_r = bulk_interp(mode_energy)
        active_eps_r = active_interp(mode_energy)
        self.bulk_eps_r = bulk_eps_r
        self.active_eps_r = active_eps_r

        dw = omega[1] - omega[0]
        bulk_eps_derivative = np.gradient(omega * np.sqrt(np.real(bulk_dispersion))) / dw
        active_eps_derivative = np.gradient(omega * np.sqrt(np.real(active_dispersion))) / dw

        d_bulk_interp = interpolate.interp1d(energy, bulk_eps_derivative)
        d_active_interp = interpolate.interp1d(energy, active_eps_derivative)
        d_bulk_eps_r = d_bulk_interp(mode_energy)
        d_active_eps_r = d_active_interp(mode_energy)
        self.d_bulk_eps_r = d_bulk_eps_r
        self.d_active_eps_r = d_active_eps_r

    def normalize(self, energy):
        self.use_disperssion(energy)
        active_eps_r = self.active_eps_r
        bulk_eps_r = self.bulk_eps_r
        d_active_eps_r = self.d_active_eps_r
        d_bulk_eps_r = self.d_bulk_eps_r
        # using the effective epsilon for the normalization
        z = self.points[:, 2]
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(),4), 10000)
        d = dipole_locations(z_linspace)
        effective_eps = active_eps_r
        effective_derivative_eps = d_active_eps_r
        normalization_results = 1 / (((np.real(effective_eps)) ** (-3/2)) * effective_derivative_eps)
        self.e_field = self.e_field * (normalization_results / self.compute_norm()) ** 0.5
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])

    def scatter_plot_3d(self, func_array=None, func_name='$\Gamma_m(r,\omega_m)/\Gamma_0$ [1]', as_log=True):
        # calculate at the mode's central frequency
        eps_r = np.real(self.active_eps_r)
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        if func_array is None:
            func_array = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency, eps_r)
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
        # Calculate at the mode's central frequency
        eps_r = np.real(self.active_eps_r)
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        z_values = self.points[:,2]
        rho_values = np.sqrt(self.points[:,0] ** 2 + self.points[:,1] ** 2)
        if self.gamma_ratio is None:
            # add mode's Q 
            gamma = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency, eps_r)
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
        # calculate at the mode's central frequnecy
        eps_r = np.real(self.active_eps_r)
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        z_values = self.points[:,2]
        x_values = self.points[:,0]
        if self.gamma_ratio is None:
            # add mode's Q 
            gamma = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency, eps_r)
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
        # calculate the mode's central frequency
        eps_r = np.real(self.active_eps_r)
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        gamma_ratio = gamma_m(self.e_field[:,2], self.Q, self.frequency, self.frequency, eps_r)
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

        gamma_ratio_max = np.log10(gamma_m(u_z_max, self.Q, f, self.frequency, eps_r))
        gamma_ratio_1L = np.log10(gamma_m(u_z_1L, self.Q, f, self.frequency, eps_r))
        gamma_ratio_075L = np.log10(gamma_m(u_z_075L, self.Q, f, self.frequency, eps_r))
        gamma_ratio_05L = np.log10(gamma_m(u_z_05L, self.Q, f, self.frequency, eps_r))
        gamma_ratio_025L = np.log10(gamma_m(u_z_025L, self.Q, f, self.frequency, eps_r))
        gamma_ratio_0L = np.log10(gamma_m(u_z_0L, self.Q, f, self.frequency, eps_r))

        chosen_rhos = np.array([rho[idx_0L], rho[idx_025L], rho[idx_05L], rho[idx_075L], rho[idx_1L]])
        chosen_zs = np.array([z[idx_0L], z[idx_025L], z[idx_05L], z[idx_075L], z[idx_1L]])

        #plt.rcParams['figure.figsize'] = [10, 5]
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        #fig = plt.figure(figsize=(20,5))
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        f_thz = f/1e12
        plt1 = ax1.plot(f_thz, gamma_ratio_0L, 'b', f_thz, gamma_ratio_025L, 'r', f_thz, gamma_ratio_05L, 'g', f_thz, gamma_ratio_075L, 'k', f_thz, gamma_ratio_1L, 'c', f_thz, gamma_ratio_max, ',c')
        ax1.set_title(r'Mode strength along the z-axis at the radius with maximal emission', fontsize=8)
        ax1.legend(['z = 0L', 'z = 0.25L', 'z = 0.5L', 'z = 0.75L', 'z = L', 'z = L (Max)'], loc='upper right', fontsize=6)
        ax1.set_xlabel("Frequency [THz]", fontsize=8)
        ax1.set_ylabel(r"$log_{10}(\Gamma_m(r,\omega)/\Gamma_0)$ [1]", fontsize=8)

        ax2 = fig.add_subplot(212)
        plt2 = ax2.tripcolor(rho /1e-6, z/1e-6 , np.log10(gamma_ratio), cmap=plt.cm.plasma, shading='gouraud')
        ax2.scatter(chosen_rhos/1e-6 , chosen_zs/1e-6 , facecolor=['blue','red','green','black','cyan'])
        ax2.set_title('Emission Enhancement $\Gamma_m/\Gamma_0$', fontsize=8)
        cbar2 = fig.colorbar(plt2, ax=ax2)
        cbar2.set_label('$log_{10}(\Gamma_m(r,\omega_m)/\Gamma_0)$ [1]', fontsize=8)
        ax2.set_xlabel(r"$\rho$ [$\mu$m]", fontsize=8)
        ax2.set_ylabel(r"z [$\mu$m]", fontsize=8)
        #plt.subplots_adjust(hspace=0.3)
        plt.subplots_adjust(top=0.928, bottom=0.148, left=0.088, right=0.964, hspace=0.5)
        #fig.tight_layout()
        plt.show()

    def calculate_effective_rate(self, z, u_z, f, gamma_func=None):
        # step 1 - 2d integration over xy to obtain gamma(z). 
        #        step 1a - collecting every xy values in a certain z (z is rounded)
        #        step 1b - integration for every z using triangulation
        # step 2 - interpolation over z
        # step 3 - 1d integration over gamma(z)*Y(z)dz
        energy = f * cn.h / cn.e
        eps_r = np.real(disspersion(energy)[1])
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(),4), 10000)
        dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, self.points)
        if gamma_func is None:
            gamma_func = gamma_m(u_z,  self.Q, f, self.frequency, eps_r)
        """
        z_dict = create_dict_by_z(self.points, gamma_func)

        xy_integrations_per_z_slice = np.array([[zi, irregular_integration_delaunay_2d(z_dict[zi]) / self.disk_area] for zi in z_dict.keys()])
        interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:,0]), np.array(xy_integrations_per_z_slice[:,1]), kind='cubic')
        interpolated_z_gamma = interpolated_z_gamma_func(z_linspace)
        """
        avg_z_gamma = averaging_over_area(self.points, self.disk_area, z_linspace, gamma_func)
        gamma_effective = regular_integration_1d(dipole_density * avg_z_gamma, z_linspace)

        return gamma_effective

    def calculate_effective_rate2(self, z, u_z, f, gamma_func=None):
        # step 1 - 2d integration over xy to obtain gamma(z).
        #        step 1a - calculating the mean value at a certain z
        #        step 1b - integration for every z using triangulation
        # step 2 - interpolation over z
        # step 3 - 1d integration over gamma(z)*Y(z)dz
        energy = f * cn.h / cn.e
        eps_r = np.real(disspersion(energy)[1])
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(),4), 10000)
        dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, self.points)
        if gamma_func is None:
            gamma_func = gamma_m(u_z,  self.Q, f, self.frequency, eps_r)
        z_dict = create_dict_by_z(self.points, gamma_func)

        xy_integrations_per_z_slice = np.array([[zi, np.mean(np.array(z_dict[zi])[:, 2])] for zi in z_dict.keys()])
        interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:,0]), np.array(xy_integrations_per_z_slice[:,1]), kind='linear')

        interpolated_z_gamma = interpolated_z_gamma_func(z_linspace)
        gamma_effective = regular_integration_1d(dipole_density * interpolated_z_gamma, z_linspace)
        gamma_effective = regular_integration_1d(1 * interpolated_z_gamma, z_linspace)
        return gamma_effective

    def calculate_AdotP_effective_rate(self, z, u_z, f, f_ij, psi_i, psi_f, z_wv):
        # for the A dot P hamiltonian
        energy = f * cn.h / cn.e
        eps_r = np.real(disspersion(energy)[1])
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
        dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, self.points)
        if type(f_ij) is list:
            pass
        else:
            gamma_func = non_approximated_gamma_m(u_z, self.Q, f_ij, f, psi_i, psi_f, z_wv, eps_r)
        z_dict = create_dict_by_z(self.points, gamma_func)

        xy_integrations_per_z_slice = np.array([[z, irregular_integration_delaunay_2d(z_dict[z])] for z in z_dict.keys()])
        interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:, 0]), np.array(xy_integrations_per_z_slice[:, 1]), kind='cubic')

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
        mode_energy = self.frequency * cn.h / cn.e
        self.normalize(mode_energy)
        if f_array is None:
            f = np.linspace(self.frequency * 0.8, self.frequency * 1.2, 41)
        else:
            f = f_array
        z = self.points[:, 2]
        u_z = self.e_field[:, 2]
        gammas = np.array([self.calculate_effective_rate(z, u_z, f[i]) for i in tqdm(range(len(f)))])
        np.save(file_name, gammas)

    def try_AdotP(self, z_wv, psi_i, psi_f):
        z = self.points[:, 2]
        u_z = self.e_field[:, 2]
        gamma_max = self.calculate_effective_rate(z, u_z, self.frequency)
        adotp_max = self.calculate_AdotP_effective_rate(z, u_z, self.frequency, psi_i, psi_f, z_wv)
        print(gamma_max / adotp_max, gamma_max, adotp_max)

    def save_AdotP_emission(self, wv_path, file_name=None, f_array=None):
        z = self.points[:, 2]
        u_z = self.e_field[:, 2]

        wavetot, z_wv, levelstot = load_wavefunction(wv_path)
        z_wv = z_wv * 1e-9
        PER = 6
        """
        50.50.50.50
        init_states = [0, PER + 2]  # states 0, 8
        FINAL_STATE = PER + 1  # state 7
        """
        init_states = [1,2]  # states 0, 8
        FINAL_STATE = 0  # state 7
        gamma_rates = []

        for INIT_STATE in init_states:
            psi_i = normalize_wv(wavetot[:, INIT_STATE], z_wv)
            psi_f = normalize_wv(wavetot[:, FINAL_STATE], z_wv)
            energy_i = levelstot[INIT_STATE]
            energy_f = levelstot[FINAL_STATE]

            #plt.plot(0.3 * wavetot[:,INIT_STATE] ** 2 + levelstot[INIT_STATE])
            #plt.plot(0.3 * wavetot[:, FINAL_STATE] ** 2 + levelstot[FINAL_STATE])
            #plt.show()
            f_ij = abs(energy_i - energy_f) * cn.e / cn.h
            print("d = {}e*m".format(regular_integration_1d(psi_i * z_wv * psi_f, z_wv)))
            print("E = {}meV".format(abs(energy_f - energy_i) / 1e-3))
            print("f = {}THz".format(f_ij / 1e12))
            if f_array is None:
                f = [f_ij]
            else:
                f = f_array
            adotp_gamma = [self.calculate_AdotP_effective_rate(z, u_z, f_ij, f[i], psi_i, psi_f, z_wv) for i in tqdm(range(len(f)))]
            gammas = np.array([self.calculate_effective_rate(z, u_z, f[i]) for i in tqdm(range(len(f)))])
            gamma_rates.append(adotp_gamma)
        print("A = {}".format(gamma_rates))
        print("B = {}".format(gammas))

    def average_field(self):
        points = self.points
        z = points[:,2]
        E_z = self.e_field[:,2]

        ones_array = create_dict_by_z(points, [1] * len(points)) # normalization
        xy_areas = [irregular_integration_delaunay_2d(ones_array[zi]) for zi in ones_array.keys()]
        A = np.mean(xy_areas)
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
        z_dict = create_dict_by_z(points, E_z)
        xy_integrations_per_z_slice = np.array([[zi, irregular_integration_delaunay_2d(z_dict[zi]) / A] for zi in z_dict.keys()])
        interpolated_z_field_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:, 0]), np.array(xy_integrations_per_z_slice[:, 1]), kind='linear')
        averaged_z_field = interpolated_z_field_func(z_linspace)

        return averaged_z_field, z_linspace


    def approximations_comparison(self, wv_path, file_name=None, f=None, save_log=False):
        if save_log:
            log_file = open(r"C:\Shaked\Technion\QCL_Project\logs\21_12_21_injector\{}.txt".format(file_name), 'w')
        # control flags
        PLOT_Z_RATE = False
        TOTAL_Z_RATE = True
        SANITY = False
        PLOT_DIPOLE = False
        RUN_PERIODS = True

        # setting parameters
        init_states = [1]  # states 0, 8
        FINAL_STATE = 0  # state 7
        f_m = self.frequency
        Q = self.Q

        k = 2 * np.pi * f_m * np.sqrt(self.active_eps_r) / cn.c
        e = cn.e
        PERIODS = range(12)
        PER_LEN = 30.68e-9

        # loading device parameters
        z = self.points[:, 2]

        wavetot, z_wv, levelstot = load_wavefunction(wv_path)
        z_wv = z_wv * 1e-9
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(),4), 10000)

        dipole_z_density = generate_dipole_along_z(dipole_locations(z_linspace), z_linspace)
        d_interp = interpolate.interp1d(z_linspace, dipole_z_density, kind='linear',bounds_error=False, fill_value=0)

        #for test purposes:
        if f == None:
            f = np.array([f_m])

        for INIT_STATE in init_states:
            psi_i = wavetot[:, INIT_STATE] * np.sqrt(1e9)
            psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
            energy_i = levelstot[INIT_STATE]
            energy_f = levelstot[FINAL_STATE]
            delta_energy = abs(energy_i - energy_f)
            self.use_disperssion(delta_energy)
            eps_r = np.real(self.active_eps_r)
            self.normalize(delta_energy)
            u_z = self.e_field[:, 2]
            f_ij = delta_energy * cn.e / cn.h
            print("f_ij = {}THz".format(f_ij / 1e12))
            print("Delta E = {}eV".format(abs(energy_i - energy_f)))
            d = e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
            print("d = {}".format(d / e * 1e9))
            if save_log:
                log_file.write("f_ij = {}THz\n".format(f_ij / 1e12))
                log_file.write("Delta E = {}eV\n".format(abs(energy_i - energy_f)))
                log_file.write("d = {}\n".format(d / e * 1e9))
            only_momentum_product = regular_integration_1d(np.conj(psi_f) * np.gradient(psi_i, z_wv), z_wv)
            with_exponent_product = regular_integration_1d(np.conj(psi_f) * np.exp(1j * k * z_wv) * np.gradient(psi_i, z_wv), z_wv)
            #complete_non_approx_prod = inner_prodcut_with_field(z, u_z, z_wv, np.conj(psi_f) * np.gradient(psi_i, z_wv))
            #complete_exponent_prod = inner_prodcut_with_field(z, u_z, z_wv, np.conj(psi_f) * np.gradient(psi_i * np.exp(1j * k * z_wv)))

            if PLOT_DIPOLE:
                for per in PERIODS:
                    #plt.plot(z_wv + per * PER_LEN, psi_i,'r', z_wv + per * PER_LEN, psi_f, 'g')
                    psi_prod = np.gradient(psi_i) * psi_f
                    plt.plot(z_wv + per * PER_LEN, psi_prod, 'g')
                    ddd = d_interp(z_wv + per * PER_LEN)
                    plt.plot(z_wv + per * PER_LEN, ddd / max(ddd) * max(psi_prod) * 2, 'b')
                plt.show()

            if PLOT_Z_RATE:
                plt.plot(z, np.log(Gamma_m(u_z,d, Q, f_ij, f_m)))
                plt.plot(z, np.log(non_dipole_Gamma_m(u_z, only_momentum_product, Q, f_ij, f_m)))
                plt.plot(z, np.log(non_dipole_Gamma_m(u_z, with_exponent_product, Q, f_ij, f_m)))
                plt.legend(['Ed', 'Ap', 'Ape^(ikz)'])
                plt.show()
            if TOTAL_Z_RATE:
                electric_dipole_result = self.calculate_effective_rate(z, u_z, f, Gamma_m(u_z, d, Q, f_ij, f_m, eps_r))
                original_potential_result = self.calculate_effective_rate(z, u_z, f, non_dipole_Gamma_m(u_z, only_momentum_product, Q, f_ij, f_m, eps_r))

                #with_exponent_result = self.calculate_effective_rate(z, u_z, f, non_dipole_Gamma_m(u_z, with_exponent_product, Q, fi, f_m))
                #electric_dipole_result = regular_integration_1d(Gamma_m(uz_interp * dipole_density,d, Q, f_ij, f_m), z_linspace)
                #non_dipole_result = regular_integration_1d(non_dipole_Gamma_m(uz_interp * dipole_density, only_momentum_product, Q, f_ij, f_m), z_linspace)
                OFFSET = 8.73e-9

                if RUN_PERIODS:
                    p = self.points
                    periods_list = []
                    for per in PERIODS:
                        psi_func = np.conj(psi_f) * np.gradient(psi_i, z_wv) * (np.heaviside(z_wv - OFFSET,1) - np.heaviside(z_wv - OFFSET - 3e-9,1))
                        product = inner_product_squared(p, u_z, z_wv + per * PER_LEN, psi_func, self.disk_area)
                        ipfg = inner_product_field_gamma(product, Q, f_ij, f_m, eps_r)
                        periods_list.append(ipfg)
                        print(per, ipfg)
                        if save_log:
                            log_file.write('period: ' + str(per) + '   ipfg: ' + str(ipfg) + '   ipfg [sec]: ' + str(1/ipfg) + '\n')
                    non_approximated_rate = sum(periods_list)

                    print("Non approximated = {}  --->  {} [sec],       [{}]".format(non_approximated_rate, 1/non_approximated_rate, (non_approximated_rate / electric_dipole_result - 1) * 100))
                    if save_log:
                        log_file.write("Non approximated = {}  --->  {} [sec],       [{}]\n".format(non_approximated_rate, 1/non_approximated_rate, (non_approximated_rate / electric_dipole_result - 1) * 100))

                print("E dot d = {}  --->  {} [sec],      [{}]".format(electric_dipole_result, 1/electric_dipole_result ,  (electric_dipole_result / electric_dipole_result - 1) * 100))

                print("A dot p = {}  --->  {} [sec],      [{}]".format(original_potential_result, 1/original_potential_result, (original_potential_result / electric_dipole_result - 1) * 100))

                if save_log:
                    log_file.write("E dot d = {}  --->  {} [sec],      [{}]\n".format(electric_dipole_result, 1/electric_dipole_result ,  (electric_dipole_result / electric_dipole_result - 1) * 100))
                    log_file.write("A dot p = {}  --->  {} [sec],     [{}]\n".format(original_potential_result,
                                                                                 1 / original_potential_result, (
                                                                                             original_potential_result / electric_dipole_result - 1) * 100))
                    log_file.close()
            if SANITY:
                #sanity check:
                print("d^2 / overlaps^2 = {}".format((d / only_momentum_product)**2))
                w_m = 2 * np.pi * f_m
                #print("E dot d * (e^2*pi^2*hbar^2/2m^2w^2)  / A dot p = {}".format(electric_dipole_result * ((cn.e ** 2 * np.pi * cn.hbar **2) / (4 * (cn.m_e * 0.067) ** 2 * (2 * np.pi * f_ij) ** 2)) / non_dipole_result))
                mass = 0.067 * cn.m_e
                #Mt_squared = (abs(d) * 2 * mass / cn.e) ** 2 * 2 * f_ij / cn.epsilon_0
                #print("Transition Matrix (?): {}".format(Mt_squared * 2 / mass * 6.2414e18 * 1e3))

    def plot_gamma(self, z, gamma_func):
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
        interpolated_z_gamma = averaging_over_area(self.points, self.disk_area, z_linspace, gamma_func)
        plt.plot(z_linspace * 1e9, 1/interpolated_z_gamma * 1e6)
        plt.title("Emission rate vs z")
        plt.xlabel('z [nm]')
        plt.ylabel('Rate [us]')




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
        # sum * 12 is larger than the device's z-length
        TOTAL_LAYERS = LAYER1 + LAYER2 + LAYER3 + LAYER4 + LAYER5 + dipole_width
        dipole_layer_offset = z_linspace.max() - (LAYER1 + LAYER2 + LAYER3)
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
        #ones_array = create_dict_by_z(points, [1] * len(points)) # normalization

        A = 1
        # the dipole normalized density isn't affected by the dipole strength
        dipole_density = d / (regular_integration_1d(d, z_linspace) * A)
        return dipole_density

def generate_dipole_along_z(d, z_linspace):
    # the dipole normalized density isn't affected by the dipole strength
    dipole_density = d / (regular_integration_1d(d, z_linspace))
    return dipole_density

def spontaneous_emission(d, f, eps_r=1):
    hbar = cn.hbar #1.0545711818e-34
    eps_0 = cn.epsilon_0 #8.85418781762039e-12
    c = cn.speed_of_light
    w = 2 * np.pi * f
    res = ((w ** 3) * (abs(d) ** 2) * (eps_r ** 0.5)) / (3 * np.pi * eps_0 * hbar * (c ** 3))
    return res

def Gamma_m(u, d, Q, f, f_m, eps_r):
    hbar = cn.hbar #1.0545711818e-34
    eps_0 = cn.epsilon_0 #8.85418781762039e-12
    #eps_r = 1
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    res = ((abs(d) ** 2) * (abs(u) ** 2) * 4 * Q * w_m * w) / ((hbar * eps_0 * (eps_r ** 2) * np.pi) * (4 * ((Q * (w - w_m)) ** 2) + w_m ** 2))
    return res

def non_dipole_Gamma_m(u, prod, Q, f, f_m, eps_r):
    e = cn.e
    m = cn.m_e * 0.067
    hbar = cn.hbar
    eps_0 = cn.epsilon_0
    #eps_r = 1
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    res = ((e / m) ** 2) * (Q * hbar / (w * eps_0 * (eps_r ** 2))) * (w_m / (4 * ((Q * (w - w_m)) ** 2) + w_m ** 2)) * (abs(u) ** 2) * (abs(prod) ** 2)
    return res

def inner_product_field_gamma(prod, Q, f, f_m, eps_r):
    e = cn.e
    m = cn.m_e * 0.067
    hbar = cn.hbar
    eps_0 = cn.epsilon_0
    #eps_r = 1
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    # assuming prod is already abs-squared
    res = ((e / m) ** 2) * (Q * hbar / (w * eps_0 * (eps_r ** 2))) * (w_m / (4 * ((Q * (w - w_m)) ** 2) + w_m ** 2)) * prod
    return res

def gamma_m(u, Q, f, f_m, eps_r):
    # the relation between Gamma_m / Gamma_0
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    c = 2.997e8
    n = np.sqrt(eps_r)
    res = 12 * (c ** 3) * Q * w_m * (abs(u) ** 2) / ((w ** 2) * (4 * (Q ** 2) * ((w - w_m) ** 2) + (w_m ** 2)) * (n ** 5))
    return res

def non_approximated_gamma_m(u, Q, f, f_m, psi_i, psi_f, z_wv, eps_r):
    # the relation between Gamma_m / Gamma_0 - for the A dot P hamiltonian
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    c = cn.speed_of_light
    c = 3e8
    hbar = cn.hbar
    hbar = 1.054e-34
    e = cn.e
    e = 1.6e-19
    m_e = cn.electron_mass
    m_e = 9.1e-31
    d = e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
    n = np.sqrt(eps_r)
    inner_product_1 = regular_integration_1d(np.conj(psi_i) * np.gradient(psi_f, z_wv), z_wv)
    inner_product_2 = regular_integration_1d(np.conj(psi_f) * np.gradient(psi_i, z_wv), z_wv)

    #plt.plot(z_wv, psi_i * np.gradient(psi_f, z_wv))
    #plt.plot(z_wv, psi_f * np.gradient(psi_i, z_wv))
    #plt.show()

    wavefunctions_term = abs(inner_product_2) ** 2 #inner_product_1 * inner_product_2
    consts_term = ((e / m_e) ** 2) * (6 * np.pi * Q * (c ** 3) * (hbar ** 2)) / ((abs(d) ** 2) * (w ** 4) * (n ** 5)) / 2
    lorentzian_term = w_m * (abs(u) ** 2) / ((4 * (Q ** 2) * ((w - w_m) ** 2) + (w_m ** 2)))
    res = consts_term * lorentzian_term * wavefunctions_term

    return res

def save_effective_emissions(electric_modes, path, f_array):
    for i, ei in enumerate(electric_modes):
        print("mode number {}".format(i))
        #ei.normalize()
        file_name = path + r"\gamma_ratios_ef_{}_r_{}_f_{}.npy".format(str(ei.ef), str(ei.radius), str(ei.freq_str))
        ei.save_mode_emission(file_name, f_array)

def create_total_emission_graph(electric_modes, path, f, verbose=True):
    total_emission = np.array([])
    dipole_path = path + r"\dipole_spectrum.csv"
    factors = []
    for ei in electric_modes:
        file_name = path + r"\gamma_ratios_ef_{}_r_{}_f_{}.npy".format(str(ei.ef), str(ei.radius), str(ei.freq_str))
        try:
            emission = np.load(file_name)
            dipoles = load_dipole_spectrum(dipole_path, f)
        except:

            "Couldn't load the emission"
            emission = 0
        factors.append((ei.freq_str, calculate_purcell_factor(emission, dipoles, f)))
        if len(total_emission) == 0:
            total_emission = np.zeros(len(emission))
        total_emission += emission
        total_purcell = calculate_purcell_factor(total_emission, dipoles, f)
    if verbose:
        for fact in factors:
            print("Mode {} - Fp = {} - Relative {}".format(fact[0], fact[1], np.round(fact[1] / total_purcell * 100, 2)))
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

def get_emission_spectrum(electric_modes, path, f, plot=True, verbose=True, ax=None):
    dipole_path = path + r"\dipole_spectrum.csv"
    total_emission = create_total_emission_graph(electric_modes, path, f, verbose)
    dipoles = load_dipole_spectrum(dipole_path, f)
    central_freqs = np.array([Ei.frequency for Ei in electric_modes])
    Fp = calculate_purcell_factor(total_emission, dipoles, f)

    if plot:
        if ax is None:
            plt.figure()
            plt.plot(f / 1e12, total_emission, f / 1e12, dipoles, 'm')
            for cf in central_freqs:
                plt.axvline(x=cf / 1e12, linestyle=':', linewidth=0.5, c='k')

            plt.title('Emission Spectrum for Ef={}eV, R={}m'.format(electric_modes[0].ef, electric_modes[0].radius) +' with $F_P^{eff}$' + '= {}'.format(Fp))
            plt.ylabel('Spectrum [a.u]')
            plt.xlabel('Frequnecy [THz]')
            #plt.legend(["$\Gamma(\omega)/\Gamma_0(\omega)$ with graphene", "Spectrum without graphene", "Modes frequencies"])
            plt.legend(['Total spectral enhancement', 'Emitters taken from APL paper', 'Modes frequencies'])
            #plt.figure()

            #plt.title(r'Structure Emission for Ef={}eV, R={}m'.format(electric_modes[0].ef, electric_modes[0].radius))
            #plt.plot(f / 1e12, total_emission)

            #plt.xlabel('Frequency [THz]')
            #plt.ylabel('Spectrum [a.u]')
            #plt.legend(["Total emission with graphene", "Modes frequencies"])
            plt.show()
        else:
            dipoles = dipoles / max(dipoles) * 1.5 * max(total_emission) # arranging nicer plots
            ax.plot(f / 1e12, total_emission, f / 1e12, dipoles, 'm', linewidth=0.7)
            for cf in central_freqs:
                ax.axvline(x=cf / 1e12, linestyle=':', linewidth=0.5, c='k')
                ax.set_title('Ef={}eV, R={}m'.format(electric_modes[0].ef, electric_modes[
                    0].radius) + ' with $F_P^{eff}$' + '= {}'.format(Fp), fontsize=6)
                ax.set_ylabel('Spectrum [a.u]', fontsize=6)
                ax.set_xlabel('Frequnecy [THz]', fontsize=6)
                plt.sca(ax)
                plt.yticks(fontsize=6, rotation=90)
                plt.xticks(fontsize=6)
                ax.legend(['Total spectral enhancement', 'Emitters taken from APL paper', 'Modes frequencies'], fontsize=4)

        if verbose:
            print('Ef={}eV    ----    rad={}um    ----    Fp={}'.format(electric_modes[0].ef, electric_modes[0].radius, Fp))
    return Fp

def calculate_purcell_factor(total_emission, dipole_emission, f):
    normalized_dipole_spect = np.array(dipole_emission / regular_integration_1d(dipole_emission, f))
    return int(np.round(regular_integration_1d(normalized_dipole_spect * total_emission, f),3))

def generate_purcell_heatmap(data=[[4206, None, 3399], [2926, 2009, 1872], [2743, 2124, 2167]]):
    import seaborn as sns

    Index = ['0.15', '0.2', '0.25']

    Cols = ['2.2', '2.6', '3.0']

    #data = [[17341, None, 13968],[42250, 29139, 27090],[39525, 30757, 31335]]
    #data = [[4206, None, 3399], [2926, 2009, 1872], [2743, 2124, 2167]]
    df = pd.DataFrame(data, index=Index, columns=Cols)

    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.0f')
    plt.title("Effective Purcell Enhancement Summary")
    plt.xlabel('Radius [$\mu$m]')
    plt.ylabel('Graphene $E_f$ [eV]')
    plt.show()

def averaging_over_area(points, area, z_linspace, gamma_func, kind=None):
    if kind is None:
        kind = 'cubic'
    z_dict = create_dict_by_z(points, gamma_func)
    xy_integrations_per_z_slice = np.array(
        [[zi, irregular_integration_delaunay_2d(z_dict[zi])] for zi in z_dict.keys()])
    interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:, 0]),
                                                     np.array(xy_integrations_per_z_slice[:, 1]), kind=kind)
    interpolated_z_gamma = interpolated_z_gamma_func(z_linspace)
    return interpolated_z_gamma / area

def run_emission_spectrum(device_spectrum_path, f_array=np.linspace(3e12, 4.8e12, 301), plot=False, verbose=True):
    paths = [
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um'
    ]
    fp_list = []
    fig = None
    axs = None
    ax = None
    if plot:
        fig, axs = plt.subplots(3, 3)

        pass
    for idx, pth in enumerate(paths):
        if '0.15' in pth and '2.6' in pth:
            print("--- Bad data folder eV=0.15 rad=2.6um")
            fp_list.append(float("NaN"))
            continue
        E = load_modes(pth)
        if plot:
            ax = axs[int(idx / 3), idx % 3]
            pass
        fp_res = get_emission_spectrum(E, device_spectrum_path, f_array, plot, verbose, ax)
        fp_list.append(fp_res)

    if plot:
        fig.tight_layout()
        fig.show()

    fp_array=np.array(fp_list).reshape([3,3])
    fig = plt.figure()
    generate_purcell_heatmap(fp_array)


def run_approximation_comparison(folder_path, wv_path, save_log=False):
    rads = ['2.2', '2.6', '3.0']
    efs = ['0.15', '0.2', '0.25']
    for ef in efs:
        for rad in rads:
            if ef == '0.25':
                E = load_modes(
                    folder_path + r"\Ef-" + ef + "eV_" + rad + "um")
            else:
                if ef =='0.15' and rad == '2.6':
                    continue
                E = load_modes(
                    folder_path + r"\E_f-" + ef + "eV\Ef-" + ef +"eV_" + rad + "um")
            for Ei in E:
                name = "Ef_" + str(Ei.ef) + "_r_" + str(np.round(Ei.radius * 1e6,2)) + 'um_f_' + Ei.freq_str +'THz'
                Ei.approximations_comparison(wv_path, name, save_log=save_log)

def run_save_effective_emission(dest_path, f_array):
    paths = [
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.2eV\Ef-0.2eV_3.0um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.2um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_2.6um',
        r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\Ef-0.25eV_3.0um'
    ]
    for path in paths:
        if '0.15' in path and '2.6' in path:
            print("--- Bad data folder eV=0.15 rad=2.6um")
            continue
        E = load_modes(path)
        save_effective_emissions(E, dest_path, f_array)

def disspersion(mode_energy):
    # calcualting the effective epsilon and derivative of epsilon:
    eps = np.load(r"C:\Shaked\Technion\QCL_Project\Wavefunctions\structure_epsilon.npy", allow_pickle=True)[()]
    bulk_dispersion = eps['bulk_eps']
    active_dispersion = eps['total_eps']

    energy = eps['omega']
    omega = energy / cn.physical_constants['reduced Planck constant in eV s'][0]
    bulk_interp = interpolate.interp1d(energy, bulk_dispersion)
    active_interp = interpolate.interp1d(energy, active_dispersion)

    bulk_eps_r = bulk_interp(mode_energy)
    active_eps_r = active_interp(mode_energy)


    dw = omega[1] - omega[0]
    bulk_eps_derivative = np.gradient(omega * np.sqrt(np.real(bulk_dispersion))) / dw
    active_eps_derivative = np.gradient(omega * np.sqrt(np.real(active_dispersion))) / dw

    d_bulk_interp = interpolate.interp1d(energy, bulk_eps_derivative)
    d_active_interp = interpolate.interp1d(energy, active_eps_derivative)
    d_bulk_eps_r = d_bulk_interp(mode_energy)
    d_active_eps_r = d_active_interp(mode_energy)
    return bulk_eps_r, active_eps_r



def parse_comparison_log(log_path):
    """

    """
    log_file = open(log_path, 'r')
    log = log_file.read()
    log_file.close()
    loglines = log.splitlines()
    f_ij = float(loglines[0][loglines[0].find('=') + 2:-3]) * 1e12
    delta_E = float(loglines[1][loglines[1].find('=') + 2:-2])
    dipole = float(loglines[2][loglines[2].find('=') + 2:])
    non_approximated_result = float(loglines[-3][loglines[-3].find('>') + 2 : loglines[-3].find('[')])
    dipole_approximated_result = float(loglines[-2][loglines[-2].find('>') + 2: loglines[-2].find('[')])
    vector_potential_result = float(loglines[-1][loglines[-1].find('>') + 2: loglines[-1].find('[')])
    periods_non_approximated_res = [float(loglines[i][loglines[i].find('[') + 6:]) for i in range(3,15)]
    return [f_ij, delta_E, dipole, non_approximated_result, dipole_approximated_result, vector_potential_result,periods_non_approximated_res]

def show_rate_comparison(log_path):
    fig, axs = plt.subplots(3,3)
    rads = ['2.2', '2.6', '3.0']
    efs = ['0.15', '0.2', '0.25']
    plot_idx = dict()
    for efi in efs:
        for radi in rads:
            if float(efi) in plot_idx.keys():
                plot_idx[float(efi)][float(radi)] = dict()
            else:
                plot_idx[float(efi)] = dict()
                plot_idx[float(efi)][float(radi)] = dict()

    for filename in os.listdir(log_path):
        edotd_list = []
        non_approx_list = []
        label_list = []
        full_log_path = os.path.join(log_path, filename)
        ef = float(filename[3:filename.find('_r_')])
        radius_um = float(filename[filename.find('_r_') + 3:filename.find('um_')])
        f_thz = float(filename[filename.find('_f_') + 3:filename.find('THz')])
        parsed_log = parse_comparison_log(full_log_path)

        if f_thz in plot_idx[ef][radius_um].keys():
            #lot_idx[ef][radius_um][f_thz].append([parsed_log[4], parsed_log[3]])
            pass
        else:
            plot_idx[ef][radius_um][f_thz] = [parsed_log[4], parsed_log[3]]
        #plot_idx[ef][radius_um][f_thz].append([edotd_list, non_approx_list])
        #edotd_list.append(parsed_log[4])
        #non_approx_list.append(parsed_log[3])
        #plot_idx[ef][radius_um][f_thz].append([parsed_log[4], parsed_log[3]])

    ax_counter = 0
    for efi in efs:
        for radi in rads:
            x_axis = np.array(range(len(plot_idx[float(efi)][float(radi)].keys())))
            edotd_list = list()
            non_app_list = list()
            freqs = list()
            for f in plot_idx[float(efi)][float(radi)].keys():
                edotd, non_approx = plot_idx[float(efi)][float(radi)][f]
                edotd_list.append(edotd / 1e-6)
                non_app_list.append(non_approx / 1e-6)
                freqs.append(f)
            #if len(edotd_list) == 0:
                #print(efi, radi)
            #print('row {}'.format(int(ax_counter / 4)))
            #print('col {}'.format(ax_counter % 3))
            axs[int(ax_counter / 3), ax_counter % 3].set_title('{}eV {}um'.format(efi,radi),fontsize=6)
            axs[int(ax_counter / 3), ax_counter % 3].bar(x_axis - 0.2, edotd_list, 0.4, label='Dipole Approx.')
            axs[int(ax_counter / 3), ax_counter % 3].bar(x_axis + 0.2, non_app_list, 0.4, label='Non Approx.')
            plt.sca(axs[int(ax_counter / 3), ax_counter % 3])
            plt.xticks(x_axis, freqs, fontsize=6)
            plt.yticks(fontsize=6)

            axs[int(ax_counter / 3), ax_counter % 3].set_xlabel('Mode [THz]', fontsize=6)
            axs[int(ax_counter / 3), ax_counter % 3].set_ylabel('Rate [$\mu$sec]', fontsize=6)
            #axs[int(ax_counter / 3), ax_counter % 3].legend()
            ax_counter += 1

    plt.legend(loc='lower right', bbox_to_anchor=(1,-0.0), fontsize=6, ncol=1, bbox_transform=fig.transFigure)

    fig.tight_layout()
    plt.show()

def show_rate_for_QW(log_path):
    fig, axs = plt.subplots(3, 3)
    rads = ['2.2', '2.6', '3.0']
    efs = ['0.15', '0.2', '0.25']
    plot_idx = dict()
    for efi in efs:
        for radi in rads:
            if float(efi) in plot_idx.keys():
                plot_idx[float(efi)][float(radi)] = dict()
            else:
                plot_idx[float(efi)] = dict()
                plot_idx[float(efi)][float(radi)] = dict()

    for filename in os.listdir(log_path):
        edotd_list = []
        non_approx_list = []
        label_list = []
        full_log_path = os.path.join(log_path, filename)
        ef = float(filename[3:filename.find('_r_')])
        radius_um = float(filename[filename.find('_r_') + 3:filename.find('um_')])
        f_thz = float(filename[filename.find('_f_') + 3:filename.find('THz')])
        parsed_log = parse_comparison_log(full_log_path)

        if f_thz in plot_idx[ef][radius_um].keys():
            # lot_idx[ef][radius_um][f_thz].append([parsed_log[4], parsed_log[3]])
            pass
        else:
            plot_idx[ef][radius_um][f_thz] = parsed_log[-1]
        # plot_idx[ef][radius_um][f_thz].append([edotd_list, non_approx_list])
        # edotd_list.append(parsed_log[4])
        # non_approx_list.append(parsed_log[3])
        # plot_idx[ef][radius_um][f_thz].append([parsed_log[4], parsed_log[3]])

    ax_counter = 0
    for efi in efs:
        for radi in rads:
            x_axis = np.array(range(12))
            qw_ipfgs = list()
            freqs = list()
            for f in plot_idx[float(efi)][float(radi)].keys():
                qw_rates = np.array(plot_idx[float(efi)][float(radi)][f]) / 1e-6
                axs[int(ax_counter / 3), ax_counter % 3].plot(x_axis+1, qw_rates, '+', label='{}THz'.format(f))
            plt.sca(axs[int(ax_counter / 3), ax_counter % 3])
            plt.yticks(fontsize=6)
            plt.xticks(x_axis+1,fontsize=6)
            axs[int(ax_counter / 3), ax_counter % 3].set_xlabel('QW [#]', fontsize=6)
            axs[int(ax_counter / 3), ax_counter % 3].set_ylabel('Rate [usec]', fontsize=6)
            axs[int(ax_counter / 3), ax_counter % 3].legend(fontsize=4)
            # axs[int(ax_counter / 3), ax_counter % 3].legend()
            ax_counter += 1

    # plt.legend(loc='lower right', bbox_to_anchor=(1, -0.0), ncol=1, bbox_transform=fig.transFigure, fontsize=6)
    fig.tight_layout()
    plt.show()


def fix_eps_r(folder_path, dest_path):
    fix_electric = True

    for filename in os.listdir(folder_path):
        log_path = os.path.join(folder_path, filename)
        f = open(log_path, 'r')
        log = f.read()
        f.close()
        parsed_log = parse_comparison_log(log_path)
        delta_E = parsed_log[1]
        electric_dipole_result = 1 / parsed_log[4]
        vector_potential_result = 1 / parsed_log[5]
        eps_r = np.real(disspersion(delta_E)[1])
        flines = log.splitlines()
        new_flines = list(flines)
        rates = []
        for i in range(3,15):
            lfln = flines[i]
            rate = float(lfln[lfln.find('ipfg: ') + 6: lfln.find('  ipfg [')])
            new_rate = rate / eps_r
            rates.append(new_rate)
            nlfln = new_flines[i]
            new_flines[i] = nlfln[:nlfln.find('ipfg: ') + 6] + str(new_rate) + nlfln[nlfln.find('  ipfg [') : nlfln.find(']:') + 3] + str(1 / new_rate)
        nlfln15 = new_flines[15]
        new_total_rate = sum(rates)

        if fix_electric:
            new_electric_dipole_result = electric_dipole_result / eps_r
            new_vector_potential = vector_potential_result / eps_r
            nlfln17 = new_flines[17]
            nlfln16 = new_flines[16]
            elec_percentage = (new_electric_dipole_result / new_electric_dipole_result - 1) * 100
            vec_percentage = (new_vector_potential / new_electric_dipole_result - 1) * 100
            new_flines[16] = nlfln16[:nlfln16.find('=') + 2] + str(new_electric_dipole_result) + nlfln16[nlfln16.find(' --'): nlfln16.find('>') + 3] + str(1 / new_electric_dipole_result) + nlfln16[nlfln16.find(' [s'): nlfln16.find('   [') + 4] + str(elec_percentage) + ']'
            new_flines[17] = nlfln17[:nlfln17.find('=') + 2] + str(new_vector_potential) + nlfln17[nlfln17.find(' --'): nlfln17.find('>') + 3] + str(1 / new_vector_potential) + nlfln17[nlfln17.find(' [s'): nlfln17.find('   [') + 4] + str(vec_percentage) + ']'
            percentage = (new_total_rate / new_electric_dipole_result - 1) * 100
        else:
            percentage = (new_total_rate / electric_dipole_result - 1) * 100
        new_flines[15] = nlfln15[:nlfln15.find('=') + 2] + str(new_total_rate) + nlfln15[nlfln15.find(' --') : nlfln15.find('>') + 3] + str(1 / new_total_rate) + nlfln15[nlfln15.find(' [s') : nlfln15.find('   [') + 4] + str(percentage) + ']'
        new_log = '\n'.join([new_flines[i] for i in range(len(new_flines))])
        new_log_path = os.path.join(dest_path, filename)
        f = open(new_log_path, 'w')
        f.write(new_log + '\n')
        f.close()

def non_approx_total_results(ULS_path, INJ_path):
    uls_parse_dict = dict()
    for filename in os.listdir(ULS_path):
        log_path = os.path.join(ULS_path, filename)
        parse_uls = parse_comparison_log(log_path)
        uls_parse_dict[filename] = parse_uls

    total_res_dict = dict()
    total_qw_list_dict = dict()
    total_spont_dict = dict()

    for filename in os.listdir(INJ_path):
        log_path = os.path.join(INJ_path, filename)
        parse_inj = parse_comparison_log(log_path)
        corresponding_uls = uls_parse_dict[filename]
        # ~ The return value of the 'parse_comparison_log' is:
        # ~ [f_ij, delta_E, dipole, non_approximated_result, dipole_approximated_result,
        # ~    vector_potential_result, periods_non_approximated_res]
        uls_qw_list = np.array(corresponding_uls[-1])
        uls_non_app_res = corresponding_uls[3]
        uls_dipole = corresponding_uls[2] * cn.e * 1e-9
        uls_fij = corresponding_uls[0]
        uls_eps_r = np.real(disspersion(corresponding_uls[1])[1])
        inj_qw_list = np.array(parse_inj[-1])
        inj_non_app_res = parse_inj[3]
        inj_dipole = parse_inj[2] * cn.e * 1e-9
        inj_fij = parse_inj[0]
        inj_eps_r = np.real(disspersion(parse_inj[1])[1])
        total_result = 1 / (1 / uls_non_app_res + 1 / inj_non_app_res)
        total_qw_list = 1 / (1 / uls_qw_list + 1 / inj_qw_list)
        total_spontaneous = 1 / (spontaneous_emission(uls_dipole, uls_fij, uls_eps_r) + spontaneous_emission(inj_dipole, inj_fij, inj_eps_r))
        total_res_dict[filename] = total_result
        total_qw_list_dict[filename] = total_qw_list
        total_spont_dict[filename] = total_spontaneous
    return total_res_dict, total_qw_list_dict, total_spont_dict

def non_approx_emission_enhancement(ULS_path, INJ_path):
    total_res_dict, total_qw_list_dict, total_spont_dict = non_approx_total_results(ULS_path, INJ_path)

    fp_list = list()
    titles_list = list()
    classifier_dict = dict()
    for key in total_res_dict.keys():
        ef = key[key.find('Ef_') + 3 : key.find('_r')]
        r = key[key.find('r_') + 2 : key.find('um')]
        f = key[key.find('_f_') + 3 : key.find('THz')]
        if ef in classifier_dict.keys():
            if r in classifier_dict[ef].keys():
                rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
                classifier_dict[ef][r] += rate_factor
            else:
                rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
                classifier_dict[ef][r] = rate_factor
        else:
            rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
            classifier_dict[ef] = dict()
            classifier_dict[ef][r] = rate_factor

    for key_e in classifier_dict.keys():
        for key_r in classifier_dict[key_e]:
            enhancement_factor = np.round(classifier_dict[key_e][key_r])
            fp_list.append(enhancement_factor)
            titles_list.append('Ef={} r={}um'.format(key_e, key_r))
    if np.size(fp_list) == 8:
        fp_list.insert(1, float("NaN"))
        titles_list.insert(1, 'Ef=0.15 r=2.6um')

    sort_idx = np.array(titles_list).argsort()
    fp_list = np.array(fp_list)[sort_idx]
    titles_list = np.array(titles_list)[sort_idx]
    fp_array = np.array(fp_list).reshape([3, 3])
    plt.figure()
    generate_purcell_heatmap(fp_array)


def show_non_approx_enhancement(ULS_path, INJ_path, f_array):
    total_res_dict, total_qw_list_dict, total_spont_dict = non_approx_total_results(ULS_path, INJ_path)
    uls_freq = parse_comparison_log(os.path.join(ULS_path, os.listdir(ULS_path)[0]))[0]
    inj_freq = parse_comparison_log(os.path.join(INJ_path, os.listdir(INJ_path)[0]))[0]
    freqs = [uls_freq, inj_freq]
    classified_list_dict = dict()
    for key in total_res_dict.keys():
        ef = key[key.find('Ef_') + 3 : key.find('_r')]
        r = key[key.find('r_') + 2 : key.find('um')]
        f = key[key.find('_f_') + 3 : key.find('THz')]
        if ef in classified_list_dict.keys():
            if r in classified_list_dict[ef].keys():
                rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
                classified_list_dict[ef][r].append((rate_factor, f))
            else:
                rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
                classified_list_dict[ef][r] = list()
                classified_list_dict[ef][r].append((rate_factor, f))
        else:
            rate_factor = ((1 / total_res_dict[key]) / (1 / total_spont_dict[key]))
            classified_list_dict[ef] = dict()
            classified_list_dict[ef][r] = list()
            classified_list_dict[ef][r].append((rate_factor, f))

    idx_counter = 0
    fig, axs = plt.subplots(3, 3)
    for ef in sorted(classified_list_dict.keys()):
        for rad in sorted(classified_list_dict[ef]):
            ax = axs[int(idx_counter / 3), idx_counter % 3]
            ax.set_title('Ef={} r={}um'.format(ef, rad), fontsize=6)
            ax.set_xlabel('Frequnecy [THz]', fontsize=6)
            ax.set_ylabel('Enhancement [1]', fontsize=6)
            ax.set_xlim([f_array.min() / 1e12, f_array.max() / 1e12])
            rates = [res_tuple[0] for res_tuple in classified_list_dict[ef][rad]]
            modes = [float(res_tuple[1]) for res_tuple in classified_list_dict[ef][rad]]
            ax.set_ylim([min(rates) * 0.8, max(rates) * 1.2])
            plt.sca(ax)
            plt.yticks(fontsize=6, rotation=90)
            plt.xticks(np.linspace(f_array.min() / 1e12, f_array.max() / 1e12, 7), fontsize=6)
            ax.plot(modes, rates, 'x', label='Mode enhancement')
            for i, cf in enumerate(freqs):
                if i == 0 :
                    label='ULS'
                    color = 'r'
                elif i==1:
                    label='Injector'
                    color = 'm'
                ax.axvline(x=cf / 1e12, linestyle=':', linewidth=0.8, c=color, label=label)
            idx_counter += 1
            if ef == '0.15' and rad == '2.2':
                ax = axs[int(idx_counter / 3), idx_counter % 3]
                ax.set_title('Ef={} r={}um'.format(ef, rad), fontsize=6)
                ax.set_xlabel('Mode [THz]', fontsize=6)
                ax.set_ylabel('Enhancement [1]', fontsize=6)
                ax.set_xlim([f_array.min() / 1e12, f_array.max() / 1e12])
                plt.sca(ax)
                plt.yticks(fontsize=6, rotation=90)
                plt.xticks(np.linspace(f_array.min() / 1e12, f_array.max() / 1e12, 7), fontsize=6)
                idx_counter += 1
    #bbox_to_anchor=(1,-0.0), fontsize=6, ncol=1, bbox_transform=fig.transFigure
    fig.legend(['Mode enhancement', 'ULS', 'Injector'], loc='lower center', ncol=3, fontsize=6, bbox_transform=fig.transFigure)
    fig.tight_layout()
    plt.show()
