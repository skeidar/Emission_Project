import scipy.io as sio
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy import constants as cn
import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
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
        self.use_disperssion(0)
        return irregular_integration_delaunay_3d(self.points, (self.e_norms) ** 2)

    def naive_normalize(self):
        self.e_field = self.e_field / (self.compute_norm() / self.active_eps_r) ** 0.5
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])

    def use_disperssion(self, energy):
        # calcualting the effective epsilon and derivative of epsilon:
        mode_energy = cn.physical_constants['Planck constant in eV/Hz'][0] * self.frequency
        bulk_dispersion = self._eps['bulk_eps']
        active_dispersion = self._eps['total_eps']
        energy = self._eps['omega'] # why there is energy as input?
        omega = energy / cn.physical_constants['reduced Planck constant in eV s'][0]
        bulk_interp = interpolate.interp1d(energy, bulk_dispersion)
        active_interp = interpolate.interp1d(energy, active_dispersion)
        bulk_eps_r = bulk_interp(mode_energy)
        active_eps_r = active_interp(mode_energy)
        self.bulk_eps_r = bulk_eps_r
        self.active_eps_r = active_eps_r

        dw = omega[1] - omega[0]
        bulk_eps_derivative = np.gradient(omega * np.sqrt(np.real(bulk_dispersion))) / dw # why did I use dw? - beacuse Martin did, and its better that way
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

    def return_polar(self, z_val=None, Np=1000):
        e_norms = self.e_norms
        e_field = self.e_field[:,2]
        print(self.freq_str)
        x, y, z = self.points.T
        # finding the polar image in z values
        polar_norms_max = []
        if z_val is None:
            z_val = z.min()
        for i, zi in enumerate(z):
            # z=360nm (the z axis is inverted) # changed from zi==0
            if round_num(zi, 2) == round_num(z_val, 2):
                polar_norms_max.append([x[i], y[i], abs(e_field[i]) ** 2])
        polar_norms_max = np.array(polar_norms_max).T
        x_norms, y_norms, norms_func_max = polar_norms_max

        # create x-y points to be used in heatmap
        xa = np.linspace(x_norms.min(), x_norms.max(), Np)
        ya = np.linspace(y_norms.min(), y_norms.max(), Np)

        # Interpolate for plotting
        zg_max = griddata((x_norms, y_norms), norms_func_max, (xa[None, :], ya[:, None]), method='cubic')
        return xa, ya, zg_max

    def Epolar_plot(self, fig=None):
        """
        assuming the field is E(z)*E(r,theta)
        plotting |E(r,theta)|
        """
        e_norms = self.e_norms
        x, y, z = self.points.T
        # finding the polar image in z values
        polar_norms_min = []
        polar_norms_max = []

        for i, zi in enumerate(z):
            # z=0 (the z axis is inverted)
            if round_num(zi, 2) == round_num(z.max(), 2):
                polar_norms_min.append([x[i],y[i], e_norms[i]])
            # z=360nm (the z axis is inverted) # changed from zi==0
            if round_num(zi, 2) == round_num(z.min(), 2):
                polar_norms_max.append([x[i], y[i], e_norms[i]])
        polar_norms_min = np.array(polar_norms_min).T
        polar_norms_max = np.array(polar_norms_max).T
        x_norms_min, y_norms_min, norms_func_min = polar_norms_min
        x_norms_max, y_norms_max, norms_func_max = polar_norms_max

        if fig is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
        # create x-y points to be used in heatmap
        xi = np.linspace(x_norms_min.min(), x_norms_min.max(), 1000)
        yi = np.linspace(y_norms_min.min(), y_norms_min.max(), 1000)
        xa = np.linspace(x_norms_min.min(), x_norms_min.max(), 1000)
        ya = np.linspace(y_norms_min.min(), y_norms_min.max(), 1000)

        # Interpolate for plotting
        zg_min = griddata((x_norms_min, y_norms_min), norms_func_min, (xi[None, :], yi[:, None]), method='cubic')
        zg_max = griddata((x_norms_max, y_norms_max), norms_func_max, (xa[None, :], ya[:, None]), method='cubic')
        im1 = ax1.contourf(xi / 1e-6, yi / 1e-6, zg_min, 50, cmap=plt.cm.plasma)
        im2 = ax2.contourf(xa / 1e-6, ya / 1e-6, zg_max, 50, cmap=plt.cm.plasma)
        #ax1.scatter(x_norms_min / 1e-6, y_norms_min / 1e-6)
        #ax2.scatter(x_norms_max / 1e-6, y_norms_max / 1e-6)
        ax1.set_xlabel('x [$\mu$m]')
        ax1.set_ylabel('y [$\mu$m]')
        ax1.set_title(r'Magnitude |E(r,$\theta$)| @ z=0nm')
        ax2.set_xlabel('x [$\mu$m]')
        ax2.set_ylabel('y [$\mu$m]')
        ax2.set_title(r'Magnitude |E(r,$\theta$)| @ z=360nm')

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        if fig is None:
            plt.show()
        return fig

    def Ez_plot(self, fig=None):
        """
        assuming the field is E(z)*E(r,theta)
        plotting |E(z)|
        """
        e_norms = self.e_norms
        x, y, z = self.points.T
        # finding the polar image in z values

        zi = np.linspace(z.min(), round_micro_meter(z.max(), 4), 1000)
        Ez_interp_res = averaging_over_area(self.points, self.disk_area, zi, e_norms)
        if fig is None:
            fig = plt.figure()
            # reversing z is mandatory because graphene is at z=0
            plt.plot(zi[::-1] / 1e-9, Ez_interp_res)
            plt.xlabel('z [nm]')
            plt.ylabel('Mean |E(z)|')
            plt.title('Mean magnitude |E(z)|')
            plt.show()
        else:
            ax3 = fig.add_subplot(212)
            ax3.plot(zi[::-1] / 1e-9, Ez_interp_res)
            ax3.set_xlabel('z [nm]')
            ax3.set_ylabel('Mean |E(z)|')
            ax3.set_title('Mean magnitude |E(z)|')
        return fig

    def return_z_field(self):
        # returning the average |E_z(z)|^2 along z
        e_norms = self.e_norms
        x, y, z = self.points.T
        # finding the polar image in z values
        Ez = self.e_field[:,2]
        zi = np.linspace(z.min(), round_micro_meter(z.max(), 4), 1000)
        #Ez_interp_res = averaging_over_area(self.points, self.disk_area, zi, e_norms)
        Ez_interp_res = averaging_over_area(self.points, self.disk_area, zi, abs(Ez) ** 2, kind='cubic')
        return zi, Ez_interp_res

    def Epolar_Ez_plot(self):
        fig = plt.figure()
        fig = self.Epolar_plot(fig)
        fig = self.Ez_plot(fig)
        fig.tight_layout()
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

    def calculate_rate_over_z(self, z_linspace, u_z, f_ij, d):
        energy = freq2energy(f_ij)
        eps_r = np.real(disspersion(energy)[1])
        if eps_r == 1:
            print("Warning : eps_r = 1  [might be unnormalized]")
        Q = self.Q
        f_k = self.frequency
        gamma_rate = Gamma_k(u_z, d, Q, f_ij, f_k)
        #non_interp_gamma_0 = averaging_over_area_non_interp(self.points, self.disk_area, gamma_rate)
        gamma_avg = averaging_over_area(self.points, self.disk_area, z_linspace, gamma_rate)
        return gamma_avg

    def check_rate_over_dipole(self, z_linspace, f_ij, d, normalize=False):
        f_k = self.frequency
        if normalize:
            self.normalize(freq2energy(f_k))
        u_z = self.e_field[:, 2]
        gamma_avg = self.calculate_rate_over_z(z_linspace, u_z, f_ij, d)
        return gamma_avg

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

        wavetot, z_wv, levelstot, _ = load_wavefunction(wv_path)
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

class CylinderGridField(ElectricMode):
    def __init__(self, path, ef, radius, frequency, Q, points, e_field):
        super().__init__(path, ef, radius, frequency, Q)
        self.e_field = e_field
        self.points = points
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])

    def compute_norm(self):
        return cyclinder_grid_3d_integration(self.points, (self.e_norms) ** 2)

    def normalize(self, energy):
        self.use_disperssion(energy)
        active_eps_r = self.active_eps_r
        bulk_eps_r = self.bulk_eps_r
        d_active_eps_r = self.d_active_eps_r
        d_bulk_eps_r = self.d_bulk_eps_r
        # using the effective epsilon for the normalization
        z = self.points[:, 2]
        z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
        d = dipole_locations(z_linspace)
        effective_eps = active_eps_r
        effective_derivative_eps = d_active_eps_r
        normalization_results = 1 / (((np.real(effective_eps)) ** (-3 / 2)) * effective_derivative_eps)
        self.e_field = self.e_field * (normalization_results / self.compute_norm()) ** 0.5
        self.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in self.e_field])




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
        #dipole_layer_offset = z_linspace.max() - (LAYER1 + LAYER2 + LAYER3)
        dipole_layer_offset = (LAYER1 + LAYER2 + LAYER3)
        d = np.zeros(np.shape(z_linspace))

        count = int(round((z_linspace.max() - z_linspace.min()) / (TOTAL_LAYERS)))
        for k in range(count):
            # assuming the graphene is at z=0.36um, we know where layer1 begins
            # so the layer stacking starts from there
            for i,zi in enumerate(z_linspace):
                dipole_begin = dipole_layer_offset + TOTAL_LAYERS * k
                dipole_end = dipole_begin + dipole_width
                if zi >= dipole_begin and zi < dipole_end:
                    d[i] = 1
        #d = d[::-1]
        return d


def generate_dipole_density(d, z_linspace, points):
        # integrate to obtain Y(r)
        #ones_array = create_dict_by_z(points, [1] * len(points)) # normalization

        A = 1
        # the dipole normalized density isn't affected by the dipole strength
        Lwv = 30.68e-9
        Nqw = round((z_linspace[-1] - z_linspace[0]) / Lwv)
        dipole_density = Nqw * d / (regular_integration_1d(d, z_linspace) * A)
        return dipole_density

def generate_dipole_along_z(d, z_linspace):
    # the dipole normalized density isn't affected by the dipole strength
    dipole_density = d / regular_integration_1d(d, z_linspace)
    return dipole_density

def average_over_dipoles(func, dipoles, z):
    if len(func) != len(dipoles) or len(func) != len(z):
        raise ValueError("Incorrect sizes")
    averaged = iregular_integration_1d(func * dipoles, z) / abs(z.max() - z.min())
    return averaged


def spontaneous_emission(d, f, disp=True):
    hbar = cn.hbar #1.0545711818e-34
    eps_0 = cn.epsilon_0 #8.85418781762039e-12
    if disp:
        eps_r = np.real(disspersion(freq2energy(f))[1])
    else:
        eps_r = 1
    c = cn.speed_of_light
    w = 2 * np.pi * f
    res = ((w ** 3) * (abs(d) ** 2) * (eps_r ** 0.5)) / (3 * np.pi * eps_0 * hbar * (c ** 3))
    return res

def Gamma_m(u, d, Q, f, f_m, eps_r=None):
    hbar = cn.hbar #1.0545711818e-34
    eps_0 = cn.epsilon_0 #8.85418781762039e-12
    if eps_r is None:
        eps_r = np.real(disspersion(freq2energy(f_m))[1])
    w = 2 * np.pi * f
    w_m = 2 * np.pi * f_m
    res = ((abs(d) ** 2) * (abs(u) ** 2) * 4 * Q * w_m * w) / ((hbar * eps_0 * (eps_r ** 2) * np.pi) * (4 * ((Q * (w - w_m)) ** 2) + w_m ** 2))
    return res

def Gamma_k(u, d, Q, f, f_k, eps_r=None):
    hbar = cn.hbar #1.0545711818e-34 [J * s] = [kg * m^2 / s]
    eps_0 = cn.epsilon_0 #8.85418781762039e-12 [F / m] =  [s^4 * A^2 / (kg * m^3)]
    if eps_r is None:
        eps_r = np.real(disspersion(freq2energy(f_k))[1])
    w = 2 * np.pi * f
    w_k = 2 * np.pi * f_k
    res = ((abs(d) ** 2) * (abs(u) ** 2) * 4 * Q * w_k * w_k) / ((hbar * eps_0 * (eps_r ** 2) * np.pi) * (4 * ((Q * (w - w_k)) ** 2) + w_k ** 2))
    return res

def Gamma_k_q(u_z_q_dict, div_u_dict, points_q, psi_i, psi_f, z_wv, Q, f, f_k, area, eps_r=None):
    e = cn.e
    m = cn.m_e * 0.067
    hbar = cn.hbar
    eps_0 = cn.epsilon_0
    if eps_r is None:
        eps_r = np.real(disspersion(freq2energy(f_k))[1])
    w = 2 * np.pi * f
    w_k = 2 * np.pi * f_k

    u_prod = u_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv)
    zeros_term = np.zeros(np.shape(u_prod))  # div_term_product(u_z_interp, psi_i, psi_f, z_wv)
    div_prod = div_term_product(div_u_dict, points_q, psi_i, psi_f, z_wv)

    res_zero = np.empty(np.shape(u_prod), dtype='complex_')
    res_zero[:, :2] = u_prod[:, :2] # XY coordiantes are the same
    abs_sqrd_term_zero = np.abs(u_prod[:, 2] + zeros_term[:, 2]) ** 2

    res_div = np.empty(np.shape(u_prod), dtype='complex_')
    res_div[:, :2] = u_prod[:, :2] # XY coordiantes are the same
    abs_sqrd_term_div = np.abs(u_prod[:, 2] + div_prod[:, 2]) ** 2

    res_zero[:, 2] = ((hbar * e ** 2) / (2 * m ** 2 * eps_0 * eps_r ** 2)) * (
                Q / (4 * ((Q * (w - w_k)) ** 2) + w_k ** 2)) * abs_sqrd_term_zero
    res_div[:, 2] = ((hbar * e ** 2) / (2 * m ** 2 * eps_0 * eps_r ** 2)) * (
                Q / (4 * ((Q * (w - w_k)) ** 2) + w_k ** 2)) * abs_sqrd_term_div
    return np.real(res_zero), np.real(res_div)

def u_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv):
    psi_term = np.conj(psi_f) * np.gradient(psi_i, z_wv)
    x, y, z = points_q.T
    xy_terms = []
    xy_set = set()
    for xi, yi in zip(x, y):
        xy_set.add((xi, yi))
    for xi, yi in tqdm(xy_set, total=len(xy_set)):
        xy_func = np.array([u_z_q_dict[xi][yi][zi] for zi in z])
        u_z_interp = interpolate.interp1d(z, xy_func, kind='linear', bounds_error=False, fill_value=0)
        interp_field = u_z_interp(z_wv)
        xy_terms.append([xi, yi, regular_integration_1d(2 * psi_term * interp_field, z_wv)])
    return np.array(xy_terms)


def div_term_product(div_u_dict, points_q, psi_i, psi_f, z_wv):
    x, y, z = points_q.T
    xy_terms = []
    xy_set = set()
    psi_term = np.conj(psi_f) * psi_i
    for xi, yi in zip(x, y):
        xy_set.add((xi, yi))
    for xi, yi in tqdm(xy_set, total=len(xy_set)):
        xy_func = np.array([div_u_dict[xi][yi][zi] for zi in z])
        div_u_interp = interpolate.interp1d(z, xy_func, kind='linear', bounds_error=False, fill_value=0)
        interp_div = div_u_interp(z_wv)
        xy_terms.append([xi, yi, regular_integration_1d(psi_term * interp_div, z_wv)])
    return np.array(xy_terms)


def OLD_Gamma_k_q(u_z_q_dict, points_q, psi_i, psi_f, z_wv, Q, f, f_k, area, eps_r=None):
    e = cn.e
    m = cn.m_e * 0.067
    hbar = cn.hbar
    eps_0 = cn.epsilon_0
    if eps_r is None:
        eps_r = np.real(disspersion(freq2energy(f_k))[1])
    w = 2 * np.pi * f
    w_k = 2 * np.pi * f_k
    u_prod = u_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv)
    zeros_term = np.zeros(np.shape(u_prod)) #div_term_product(u_z_interp, psi_i, psi_f, z_wv)
    print("div")
    div_prod = div_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv)
    print("done div?")
    avg_abs_sqrd_term_zero = average_sum_of_prods(u_prod, zeros_term, area)
    avg_abs_sqrd_term_div = average_sum_of_prods(u_prod, div_prod, area)
    res_zero = ((hbar * e ** 2) / (2 * m ** 2 * eps_0 * eps_r ** 2)) * (Q / (4 * ((Q * (w - w_k)) ** 2) + w_k ** 2)) * avg_abs_sqrd_term_zero
    res_div = ((hbar * e ** 2) / (2 * m ** 2 * eps_0 * eps_r ** 2)) * (Q / (4 * ((Q * (w - w_k)) ** 2) + w_k ** 2)) * avg_abs_sqrd_term_div
    return res_zero, res_div


def OLD_u_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv):
    func = np.conj(psi_f) * np.gradient(psi_i, z_wv)
    return non_avg_inner_product_calculation_over_reg_grid(points_q, u_z_q_dict, z_wv, func)

def OLD_div_term_product(u_z_q_dict, points_q, psi_i, psi_f, z_wv):
    z = points_q[:, 2]
    xy_terms = []
    xy_set = set()
    func = np.conj(psi_f) * psi_i
    for x, y in zip(points_q[:, 0], points_q[:, 1]):
        xy_set.add((x,y))
    for x,y in tqdm(xy_set, total=len(xy_set)):
        xy_func = np.array([u_z_q_dict[x][y][zi] for zi in z])
        print(len(xy_func), len(z))
        u_z_interp = interpolate.interp1d(z, xy_func, kind='linear', bounds_error=False, fill_value=0)
        interp_field = np.gradient(u_z_interp(z_wv), z_wv)
        xy_terms.append([x, y, regular_integration_1d(func * interp_field, z_wv)])
    return np.array(xy_terms)



def average_sum_of_prods(prod1, prod2, area):
    ## assuming [[xi, yi, prod],...]
    ## assuming same x, y for both
    xy_terms = []
    print("Averaging XY")
    for i,(x, y) in enumerate(tqdm(zip(prod1[:, 0], prod1[:, 1]), total=len(prod1[:, 0]))):
        xy_terms.append((x, y, abs(2 * prod1[i,2]+prod2[i,2])**2))
    #res = irregular_integration_delaunay_2d(xy_terms) / area
    res = abs(2 * prod1[0,2]+prod2[0,2])**2 #3
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

def gamma_m(u, Q, f, f_m, eps_r=None):
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

def averaging_over_area(points, area, z_linspace, gamma_func, kind='cubic'):
    z_dict = create_dict_by_z(points, gamma_func)
    xy_integrations_per_z_slice = np.array(
        [[zi, irregular_integration_delaunay_2d(z_dict[zi])] for zi in z_dict.keys()])
    interpolated_z_gamma_func = interpolate.interp1d(np.array(xy_integrations_per_z_slice[:, 0]),
                                                     np.array(xy_integrations_per_z_slice[:, 1]), kind=kind)
    interpolated_z_gamma = interpolated_z_gamma_func(z_linspace)
    return interpolated_z_gamma / area

def averaging_over_area_non_interp(points, area, gamma_func):
    z_dict = create_dict_by_z(points, gamma_func)
    xy_integrations_per_z_slice = np.array(
        [irregular_integration_delaunay_2d(z_dict[zi]) for zi in z_dict.keys()])
    z_values = np.array([zi for zi in z_dict.keys()])
    return xy_integrations_per_z_slice / area, z_values

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
        total_spontaneous = 1 / (spontaneous_emission(uls_dipole, uls_fij, disp=True) + spontaneous_emission(inj_dipole, inj_fij, disp=True))
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


def get_total_check_dipole_rate(e_path, wv_path, f_array):
    files_path =r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\110122_updated_calculation"
    E = load_modes(e_path)
    points = E[0].points
    z = points[:, 2] # same for all
    z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)

    wavetot, z_wv, levelstot, _ = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7

    total_spectrum = np.zeros(np.shape(f_array))
    total_z_rate = np.zeros(np.shape(z_linspace))
    spontaneous_rate = 0
    total_rate = 0
    dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, points)
    for INIT_STATE in init_states:
        psi_i = wavetot[:, INIT_STATE] * np.sqrt(1e9)
        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        energy_i = levelstot[INIT_STATE]
        energy_f = levelstot[FINAL_STATE]
        delta_energy = abs(energy_i - energy_f)
        e = cn.e
        f_ij = energy2freq(delta_energy)
        d = e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
        #print("f_ij = {}THz".format(f_ij / 1e12))
        #print("Delta E = {}eV".format(delta_energy))
        #print("d = {}".format(d / e * 1e9))
        print("f_ij = {}THz;    Delta E = {}eV;     d = {}".format(f_ij / 1e12, delta_energy, d / e * 1e9))

        total_rate_for_z = np.ufunc.reduce(np.add, [np.array(Ei.check_rate_over_dipole(z_linspace, f_ij,d)) for Ei in E])

        current_total_rate = regular_integration_1d(total_rate_for_z * dipole_density, z_linspace)
        current_spontaneous_rate = spontaneous_emission(d, f_ij)
        plt.figure(1)
        current_spectrum = np.zeros(np.shape(total_spectrum))
        for Ei in E:
            mode_spectrum = []
            print("f_k = {}THz".format(Ei.freq_str))
            for i in tqdm(range(len(f_array))):
                f = f_array[i]
                spectrum_for_z = Ei.check_rate_over_dipole(z_linspace, f, d)
                mode_spectrum.append(regular_integration_1d(spectrum_for_z * dipole_density, z_linspace))
            current_spectrum += np.array(mode_spectrum)
        total_spectrum += current_spectrum
        total_z_rate += total_rate_for_z
        total_rate += current_total_rate
        spontaneous_rate += current_spontaneous_rate
    #plt.plot(z_linspace, dipole_density / max(dipole_density) * max(1/total_rate_for_z), 'gray' )
    #plt.plot(z_linspace, 1/total_rate_for_z)
    #plt.show()

    return total_z_rate, spontaneous_rate, total_z_rate, z_linspace, dipole_density, total_spectrum

def run_total_check_dipole_rate(wv_path, f_array, sim_files_path, zgammaplt=False, qwgammaplt=False, factorplt=False, spectrumplt=False, sumplt=False, zflip=False, nsec=True):
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
    nQW = 12
    enhancement_list = []
    total_vs_z_list = []
    spectrum_list = []
    total_rate_list = []
    spont_list = []
    total_max=0
    period_max = np.zeros(nQW)
    ndip = 0
    perLen = 30.68e-9
    # Code repeatition, wth...
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    bandplot[:, 0] = bandplot[:, 0] * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7
    Z_LEN = 360e-9
    if spectrumplt:
        fig, axs = plt.subplots(3, 3)



    for ax, e_path in enumerate(paths):
        label = e_path[e_path.rfind(r'Ef'):]
        print(label)
        if '0.15' in e_path and '2.6' in e_path:
            print("--- Bad data folder eV=0.15 rad=2.6um")
            enhancement_list.append(np.array([float("NaN")] * len(enhancement_list[0])))
            total_rate_list.append(np.array([float("NaN")] * len(total_rate_list[0])))
            spont_list.append(np.array([float("NaN")] * len(spont_list[0])))
            continue
        total_rate, spontaneous_rate, total_rate_for_z, z_linspace, dipole_density, spectrum, used_f_array, gamma_k_list, spectrum_k_list = load_updated_dipole_rate(e_path, wv_path, f_array, sim_files_path)

        print('Rate    [1] = {}       Rate    [2] = {}'.format(total_rate[0], total_rate[1]))
        print('Spont   [1] = {}       Spont   [2] = {}'.format(spontaneous_rate[0], spontaneous_rate[1]))
        print('Enhance [1] = {}       Enhance [2] = {}'.format(total_rate[0]/spontaneous_rate[0], total_rate[1]/spontaneous_rate[1]))
        ndip = len(spontaneous_rate)
        # calculating Fp for each dipole
        enhancement_list.append(np.array(total_rate) / np.array(spontaneous_rate))
        total_rate_list.append(np.array(total_rate))
        spont_list.append(np.array(spontaneous_rate))

        total_vs_z_list.append(total_rate_for_z)
        spectrum_list.append(spectrum)

        nsec_rates = (1 / np.array(total_rate_for_z)) / 1e-9

        if zgammaplt:
            zgammaplt_titles = ["Dipole 1: ULS-LLS Transition", "Dipole 2: Injector-LLS Transition"]
            if sumplt:
                z_total = sum(np.array([total_rate_for_z[dip, :] for dip in range(ndip)]))
                nsec_rate = (1 / z_total) / 1e-9
                if zflip:
                    if nsec:
                        plt.plot(z_linspace[::-1] / 1e-9, nsec_rate, label=label, zorder=2)
                    else:
                        plt.plot(z_linspace[::-1] / 1e-9, z_total /1e6, label=label, zorder=2) # flipping the z axis
                else:
                    if nsec:
                        plt.plot(z_linspace / 1e-9, nsec_rate, label=label, zorder=2)
                    else:
                        plt.plot(z_linspace / 1e-9, z_total /1e6, label=label, zorder=2)  # flipping the z axis
                if total_max < max(nsec_rate):
                    total_max = max(nsec_rate)
            else:
                for dip in range(ndip):
                    nsec_rate_for_z = nsec_rates[dip,:]
                    rate_for_z = total_rate_for_z[dip,:]
                    plt.figure(dip)
                    if zflip:
                        if nsec:
                            plt.plot(z_linspace[::-1] / 1e-9, nsec_rate_for_z, label=label, zorder=10)
                        else:
                            plt.plot(z_linspace[::-1] / 1e-9, rate_for_z / 1e6, label=label, zorder=10) # flipping the z axis
                    else:
                        if nsec:
                            plt.plot(z_linspace / 1e-9, nsec_rate_for_z, label=label, zorder=10)
                        else:
                            plt.plot(z_linspace / 1e-9, rate_for_z / 1e6, label=label, zorder=10)  # flipping the z axis
                    # Dipole graph normalization factor
                    if total_max < max(nsec_rate_for_z):
                        total_max = max(nsec_rate_for_z)
                    plt.title(zgammaplt_titles[dip])

        if qwgammaplt:

            # plotting each QW
            zper = int(len(z_linspace) / nQW)

            if sumplt:
                z_total = sum(np.array([total_rate_for_z[dip, :] for dip in range(ndip)]))
                nsec_rate = (1 / z_total) / 1e-9



                for i in range(nQW):
                    if i == 0 or i == (nQW - 1):
                        plt.figure(i)
                        #plt.subplot(2,1,1)
                        qw_z = z_linspace[i * zper: (i + 1) * zper]
                        if zflip:
                            flip_z_total = z_total[::-1]
                            per_z_total = flip_z_total[i * zper: (i + 1) * zper]
                            per_z_total = per_z_total[::-1]
                            flip_nsec_rate = nsec_rate[::-1]
                            per_nsec_rate = flip_nsec_rate[i * zper: (i + 1) * zper]
                            per_nsec_rate = per_nsec_rate[::-1]
                            if nsec:
                                plt.plot(qw_z[::-1] / 1e-9, per_nsec_rate, label=label)
                            else:
                                plt.plot(qw_z[::-1] / 1e-9, per_z_total / 1e6, label=label)
                        else:
                            if nsec:
                                plt.plot(qw_z / 1e-9, nsec_rate[i * zper: (i + 1) * zper], label=label)
                            else:
                                plt.plot(qw_z / 1e-9, z_total[i * zper: (i + 1) * zper] / 1e6, label=label)
                        plt.title('Period #{}'.format(i + 1))
                    # QW graph normalization factor
                        if period_max[i] < max(nsec_rate[i * zper: (i + 1) * zper]):
                            period_max[i] = max(nsec_rate[i * zper: (i + 1) * zper])

            else:
                for dip in range(ndip):
                    nsec_rate_for_z = nsec_rates[dip, :]
                    rate_for_z = total_rate_for_z[dip, :]
                    for i in range(nQW):
                        if i == 0 or i == (nQW - 1):
                            plt.figure(2 * i + dip)
                            plt.title('Period #{}, dip #{}'.format(i + 1, dip + 1))
                            #ax1 = plt.subplot(2, 1, 1) ##HERE

                            qw_z = z_linspace[i * zper: (i+1) * zper]
                            #ax1.plot(qw_z / 1e-9, nsec_rate_for_z[i * zper: (i+1) * zper], label=label) ##HERE
                            if zflip:
                                flip_rate_for_z = rate_for_z[::-1]
                                per_rate_for_z = flip_rate_for_z[i * zper: (i + 1) * zper]
                                per_rate_for_z = per_rate_for_z[::-1]
                                flip_nsec_for_z = nsec_rate_for_z[::-1]
                                per_nsec_for_z = flip_nsec_for_z[i * zper: (i + 1) * zper]
                                per_nsec_for_z = per_nsec_for_z[::-1]
                                if nsec:
                                    plt.plot(qw_z[::-1] / 1e-9, per_nsec_for_z, label=label) # flipping the z-axis
                                else:
                                    plt.plot(qw_z[::-1] / 1e-9, per_rate_for_z / 1e6, label=label)  # flipping the z-axis
                            else:
                                if nsec:
                                    plt.plot(qw_z / 1e-9, nsec_rate_for_z[i * zper: (i + 1) * zper], label=label)
                                else:
                                    plt.plot(qw_z / 1e-9, rate_for_z[i * zper: (i + 1) * zper] / 1e6, label=label)  # flipping the z-axis
                            # QW graph normalization factor
                            if period_max[i] < max(nsec_rate_for_z[i * zper : (i+1) * zper]):
                                period_max[i] = max(nsec_rate_for_z[i * zper : (i+1) * zper])
                        #ax1.legend(fontsize=6)
                        #plt.legend(fontsize=6) ##HERE
        if spectrumplt:
            # plotting the spectrum
            hard_coded_fij = [4.27, 3.58]
            axi = axs[int(ax / 3), ax % 3]
            if sumplt:
                total_spect = sum(np.array([spectrum[dip, :] for dip in range(ndip)]))
                axs[int(ax / 3), ax % 3].plot(used_f_array / 1e12, total_spect / 1e6, 'b')
                for dip in range(ndip):
                    axs[int(ax / 3), ax % 3].axvline(hard_coded_fij[dip], linestyle=':', linewidth=0.5, c='b')
                axi.set_title(label, fontsize=6)
                axi.set_xlabel('Frequency [THz]', fontsize=6)
                axi.set_ylabel('$\Gamma$ [$\mu$s$^-$$^1$]', fontsize=6)
                plt.sca(axi)
                plt.yticks(fontsize=6, rotation=90)
                plt.xticks(np.linspace(used_f_array.min() / 1e12, used_f_array.max() / 1e12, 7), fontsize=6)

            else:
                hard_coded_color = ['b','r']
                for dip in range(ndip):
                    axs[int(ax / 3), ax % 3].plot(used_f_array / 1e12, spectrum[dip, :] , hard_coded_color[dip], label="Dip. #{}".format(dip+1))
                    for S in spectrum_k_list[dip]:
                        axs[int(ax / 3), ax % 3].plot(used_f_array / 1e12, S, linestyle='--', linewidth=0.4, c=hard_coded_color[dip])
                    #axs[int(ax / 3), ax % 3].axvline(hard_coded_fij[dip], linestyle=':', linewidth=0.5, c=hard_coded_color[dip])
                axi.set_title(label, fontsize=6)
                axi.set_xlabel('Frequency [THz]', fontsize=6)
                tx = axi.yaxis.get_offset_text()
                tx.set_fontsize(6)
                #axi.set_ylabel('$\Gamma$ [$\mu$s$^-$$^1$]', fontsize=6)
                axi.set_ylabel('Spectrum [1]', fontsize=6)
                plt.sca(axi)
                plt.yticks(fontsize=6, rotation=90)
                plt.xticks(np.linspace(used_f_array.min() / 1e12, used_f_array.max() / 1e12, 7), fontsize=6)

    if zgammaplt:
        # interpolating the wavefunctions - HARDCODED
        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
        psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
        psi_i = [psi_uls, psi_inj]
        i_colour = ['#a63636' ,'#a68436']
        MILI_TO_EV = 1000
        BEAUTY_FACTOR = 2e6
        band_energy_diff = (levelstot[0] - levelstot[7]) * MILI_TO_EV

        if sumplt:
            #plt.plot(z_linspace / 1e-9, dipole_density / max(dipole_density) * total_max, 'k--', zorder=1, linewidth=0.5)
            plt.legend(fontsize=6, loc='upper left', framealpha=0.7, bbox_to_anchor=(0, -0.2), ncol=2)
            if nsec:
                plt.ylabel('$\Gamma$$^-$$^1$(z) [nsec]')
            else:
                plt.ylabel('$\Gamma$(z) [$\mu$sec$^-$$^1$]')
            plt.xlabel('z [nm]')
            plt.title('Both Transitions Included')
            NICE_BLUE = '#365ba6'
            BROWNISH_YELLOW = '#a68436'
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)
            ax2.set_ylabel('Wavefunctions [a.u.]', color='b')
            ax2.spines['right'].set_color('b')
            ax2.tick_params(axis='y', colors='b')
            ax2.yaxis.set_ticklabels([])
            for i in range(nQW):
                OFFSET = 0e-9
                ALPHA = 0.5
                interp_psi_f_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_f) ** 2, kind='cubic', fill_value=0, bounds_error=False)
                interp_psi_uls_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[0]) ** 2, kind='cubic', fill_value=0, bounds_error=False)
                interp_psi_inj_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[1]) ** 2, kind='cubic', fill_value=0, bounds_error=False)
                interp_bandplot_func = interpolate.interp1d(bandplot[:, 0] + perLen * i + OFFSET, bandplot[:, 1],
                                                            kind='linear',
                                                            fill_value=0, bounds_error=False)
                periods_args = np.logical_and(z_linspace >= (z_wv + perLen * i + OFFSET).min(),
                                              z_linspace <= (z_wv + perLen * i + OFFSET).max())
                interp_psi_f = interp_psi_f_func(z_linspace) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[FINAL_STATE] * MILI_TO_EV
                interp_psi_uls = interp_psi_uls_func(z_linspace) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[init_states[0]] * MILI_TO_EV
                interp_psi_inj = interp_psi_inj_func(z_linspace) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[init_states[1]] * MILI_TO_EV
                interp_bandplot = interp_bandplot_func(z_linspace) - band_energy_diff * i  # + levelstot[0] * 10
                #ax2.plot(z_linspace * 1e9, interp_psi_f, color='b', linestyle='--',  linewidth=0.6, zorder=1, alpha=ALPHA)
                ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_f[periods_args], color='b', linestyle='--', linewidth=0.6, zorder=1, alpha=ALPHA)
                #ax2.plot(z_linspace * 1e9, interp_psi_i, color=i_colour[dip], linestyle='--', linewidth=0.3, zorder=1, alpha=ALPHA)
                #ax2.plot(z_linspace * 1e9, interp_psi_i, color='b', linewidth=0.6, zorder=1, alpha=ALPHA)
                ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_uls[periods_args], color='b', linewidth=0.6,
                         zorder=1, alpha=ALPHA)
                ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_inj[periods_args], color='b', linestyle=':', linewidth=0.6, zorder=1, alpha=ALPHA)
                ax2.plot(z_linspace[periods_args] * 1e9, interp_bandplot[periods_args], color='black', linewidth=0.3, zorder=1, alpha=ALPHA)

                #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_f[periods_args], color='#365ba6', linestyle='--', linewidth=0.5, zorder=1, alpha=ALPHA)
                #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_i[periods_args], color=i_colour[dip], linestyle='--', linewidth=0.5, zorder=1, alpha=ALPHA)
                #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_i[periods_args], color='#365ba6', linewidth=0.5, zorder=1, alpha=ALPHA)

                #plt.plot(z_linspace / 1e-9, dipole_density / max(dipole_density) * total_max, 'k--', zorder=1,  linewidth=0.5)
            legend2 = plt.legend(['|$\psi$$_f$|$^2$', '|$\psi$$_i$|$^2$-ULS', '|$\psi$$_i$|$^2$-INJ', 'Band Structure'], fontsize=7, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, -0.2))
            for line in legend2.get_lines():
                line.set_linewidth(1.0)
            ax2.add_artist(legend2)
            plt.tight_layout(rect=[0,0.02,1,1])
            plt.show()

        else:
            for dip in range(ndip):
                plt.figure(dip)
                plt.legend(fontsize=6, loc='upper left', framealpha=0.7, bbox_to_anchor=(0, -0.2), ncol=2)
                if nsec:
                    plt.ylabel('$\Gamma$$^-$$^1$(z) [nsec]')
                else:
                    plt.ylabel('$\Gamma$(z) [$\mu$sec$^-$$^1$]')
                plt.xlabel('z [nm]')
                NICE_BLUE = '#365ba6'
                BROWNISH_YELLOW = '#a68436'
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax1.set_zorder(ax2.get_zorder() + 1)
                ax1.patch.set_visible(False)
                ax2.set_ylabel('Wavefunctions [a.u.]', color='b')
                ax2.spines['right'].set_color('b')
                ax2.tick_params(axis='y', colors='b')
                ax2.yaxis.set_ticklabels([])
                # ax2.set_yticks([0,1])
                for i in range(nQW):
                    OFFSET = 0e-9
                    ALPHA = 0.5
                    interp_psi_f_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_f) ** 2, kind='cubic',
                                                             fill_value=0, bounds_error=False)
                    interp_psi_i_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[dip]) ** 2,
                                                             kind='cubic',
                                                             fill_value=0, bounds_error=False)
                    interp_bandplot_func = interpolate.interp1d(bandplot[:, 0] + perLen * i + OFFSET, bandplot[:, 1],
                                                                kind='linear',
                                                                fill_value=0, bounds_error=False)
                    periods_args = np.logical_and(z_linspace >= (z_wv + perLen * i + OFFSET).min(),
                                                  z_linspace <= (z_wv + perLen * i + OFFSET).max())
                    interp_psi_f = interp_psi_f_func(z_linspace) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[FINAL_STATE] * MILI_TO_EV
                    interp_psi_i = interp_psi_i_func(z_linspace) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[init_states[dip]] * MILI_TO_EV
                    interp_bandplot = interp_bandplot_func(z_linspace) - band_energy_diff * i  # + levelstot[0] * 10
                    #ax2.plot(z_linspace * 1e9, interp_psi_f, color='b', linestyle='--',  linewidth=0.6, zorder=1, alpha=ALPHA)
                    ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_f[periods_args], color='b', linestyle='--', linewidth=0.6, zorder=1, alpha=ALPHA)
                    #ax2.plot(z_linspace * 1e9, interp_psi_i, color=i_colour[dip], linestyle='--', linewidth=0.3, zorder=1, alpha=ALPHA)
                    #ax2.plot(z_linspace * 1e9, interp_psi_i, color='b', linewidth=0.6, zorder=1, alpha=ALPHA)
                    ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_i[periods_args], color='b', linewidth=0.6, zorder=1, alpha=ALPHA)
                    ax2.plot(z_linspace[periods_args] * 1e9, interp_bandplot[periods_args], color='black', linewidth=0.3, zorder=1, alpha=ALPHA)

                    #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_f[periods_args], color='#365ba6', linestyle='--', linewidth=0.5, zorder=1, alpha=ALPHA)
                    #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_i[periods_args], color=i_colour[dip], linestyle='--', linewidth=0.5, zorder=1, alpha=ALPHA)
                    #ax2.plot(z_linspace[periods_args] * 1e9, interp_psi_i[periods_args], color='#365ba6', linewidth=0.5, zorder=1, alpha=ALPHA)

                #plt.plot(z_linspace / 1e-9, dipole_density / max(dipole_density) * total_max, 'k--', zorder=1,  linewidth=0.5)
                legend2 = plt.legend(['|$\psi$$_f$|$^2$','|$\psi$$_i$|$^2$','Band Structure'], fontsize=8, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, -0.2))
                for line in legend2.get_lines():
                    line.set_linewidth(1.0)
                ax2.add_artist(legend2)
                plt.tight_layout(rect=[0,0.02,1,1])
            plt.show()

    if qwgammaplt:
        # interpolating the wavefunctions - HARDCODED

        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
        psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
        psi_i = [psi_uls, psi_inj]
        total_psi_uls = np.zeros(np.shape(z_linspace))
        total_psi_inj = np.zeros(np.shape(z_linspace))
        total_psi_f = np.zeros(np.shape(z_linspace))

        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
        psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
        psi_i = [psi_uls, psi_inj]
        i_colour = ['#a63636' ,'#a68436']
        MILI_TO_EV = 1000
        BEAUTY_FACTOR = 1e6
        band_energy_diff = (levelstot[0] - levelstot[7]) * MILI_TO_EV
        if sumplt:
            for i in range(nQW):
                if i == 0 or i == (nQW - 1):
                    plt.figure(i)
                    plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2)
                    if nsec:
                        plt.ylabel('$\Gamma$$^-$$^1$(z) [nsec]')
                    else:
                        plt.ylabel('$\Gamma$(z) [$\mu$sec$^-$$^1$]')
                    plt.xlabel('z [nm]')
                    NICE_BLUE = '#365ba6'
                    BROWNISH_YELLOW = '#a68436'
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    ax2.set_zorder(ax1.get_zorder() + 1)
                    # ax1.patch.set_visible(False)
                    ax2.set_ylabel('Wavefunctions [a.u.]', color='b')
                    ax2.spines['right'].set_color('b')
                    ax2.tick_params(axis='y', colors='b')
                    ax2.yaxis.set_ticklabels([])
                    OFFSET = 0e-9
                    ALPHA = 0.5
                    qw_z = z_linspace[i * zper: (i + 1) * zper]
                    interp_psi_f_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_f) ** 2,
                                                             kind='cubic',
                                                             fill_value=0, bounds_error=False)
                    interp_psi_uls_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[0]) ** 2,
                                                             kind='cubic',
                                                             fill_value=0, bounds_error=False)
                    interp_psi_inj_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[1]) ** 2,
                                                             kind='cubic',
                                                             fill_value=0, bounds_error=False)
                    interp_bandplot_func = interpolate.interp1d(bandplot[:, 0] + perLen * i + OFFSET,
                                                                bandplot[:, 1],
                                                                kind='linear',
                                                                fill_value=0, bounds_error=False)
                    periods_args = np.logical_and(z_linspace >= (z_wv + perLen * i + OFFSET).min(),
                                                  z_linspace <= (z_wv + perLen * i + OFFSET).max())
                    interp_psi_f = interp_psi_f_func(qw_z) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[
                        FINAL_STATE] * MILI_TO_EV
                    interp_psi_uls = interp_psi_uls_func(qw_z) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[
                        init_states[0]] * MILI_TO_EV
                    interp_psi_inj = interp_psi_inj_func(qw_z) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[
                        init_states[1]] * MILI_TO_EV
                    interp_bandplot = interp_bandplot_func(qw_z) - band_energy_diff * i  # + levelstot[0] * 10
                    ax2.plot(qw_z * 1e9, interp_psi_f, color='b', linestyle='--', linewidth=0.6, zorder=1,
                             alpha=ALPHA)
                    # ax2.plot(z_linspace * 1e9, interp_psi_i, color=i_colour[dip], linestyle='--', linewidth=0.3, zorder=1, alpha=ALPHA)
                    ax2.plot(qw_z * 1e9, interp_psi_uls, color='b', linewidth=0.6, zorder=1, alpha=ALPHA)
                    ax2.plot(qw_z * 1e9, interp_psi_inj, color='b', linestyle=':', linewidth=0.6, zorder=1, alpha=ALPHA)
                    ax2.plot(qw_z * 1e9, interp_bandplot, color='black',
                             linestyle='--', linewidth=0.5, zorder=1, alpha=ALPHA)

                    legend2 = plt.legend(['|$\psi$$_f$|$^2$', '|$\psi$$_i$|$^2$-ULS', '|$\psi$$_i$|$^2$-INJ', 'Band Structure'], fontsize=8, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, -0.2))
                    for line in legend2.get_lines():
                        line.set_linewidth(1.0)
                    ax2.add_artist(legend2)
                    plt.tight_layout(rect=[0,0.05,1,1])
            plt.show()
        else:
            for dip in range(ndip):
                for i in range(nQW):
                    if i == 0 or i == (nQW - 1):
                        plt.figure(2 * i + dip)
                        plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2)
                        if nsec:
                            plt.ylabel('$\Gamma$$^-$$^1$(z) [nsec]')
                        else:
                            plt.ylabel('$\Gamma$(z) [$\mu$sec$^-$$^1$]')
                        plt.xlabel('z [nm]')
                        NICE_BLUE = '#365ba6'
                        BROWNISH_YELLOW = '#a68436'
                        ax1 = plt.gca()
                        ax2 = ax1.twinx()
                        ax2.set_zorder(ax1.get_zorder() + 1)
                        #ax1.patch.set_visible(False)
                        ax2.set_ylabel('Wavefunctions [a.u.]', color='b')
                        ax2.spines['right'].set_color('b')
                        ax2.tick_params(axis='y', colors='b')
                        ax2.yaxis.set_ticklabels([])
                        OFFSET = 0e-9
                        ALPHA = 0.5
                        qw_z = z_linspace[i * zper: (i + 1) * zper]
                        interp_psi_f_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_f) ** 2,
                                                                 kind='cubic',
                                                                 fill_value=0, bounds_error=False)
                        interp_psi_i_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[dip]) ** 2,
                                                                 kind='cubic',
                                                                 fill_value=0, bounds_error=False)
                        interp_bandplot_func = interpolate.interp1d(bandplot[:, 0] + perLen * i + OFFSET,
                                                                    bandplot[:, 1],
                                                                    kind='linear',
                                                                    fill_value=0, bounds_error=False)
                        periods_args = np.logical_and(z_linspace >= (z_wv + perLen * i + OFFSET).min(),
                                                      z_linspace <= (z_wv + perLen * i + OFFSET).max())
                        interp_psi_f = interp_psi_f_func(qw_z) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[
                            FINAL_STATE] * MILI_TO_EV
                        interp_psi_i = interp_psi_i_func(qw_z) / BEAUTY_FACTOR - band_energy_diff * i + levelstot[
                            init_states[dip]] * MILI_TO_EV
                        interp_bandplot = interp_bandplot_func(qw_z) - band_energy_diff * i  # + levelstot[0] * 10
                        ax2.plot(qw_z * 1e9, interp_psi_f, color='b', linestyle='--', linewidth=0.8, zorder=1,
                                 alpha=ALPHA)
                        # ax2.plot(z_linspace * 1e9, interp_psi_i, color=i_colour[dip], linestyle='--', linewidth=0.3, zorder=1, alpha=ALPHA)
                        ax2.plot(qw_z * 1e9, interp_psi_i, color='b', linewidth=0.8, zorder=1, alpha=ALPHA)
                        ax2.plot(qw_z * 1e9, interp_bandplot, color='black',
                                 linestyle='--', linewidth=0.6, zorder=1, alpha=ALPHA)
                        d = dipole_density[::-1][i * zper: (i + 1) * zper]
                        #d = d * max(interp_bandplot) / max(d)

                        legend2 = plt.legend(['|$\psi$$_f$|$^2$','|$\psi$$_i$|$^2$','Band Structure'], fontsize=8, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, -0.2))
                        for line in legend2.get_lines():
                            line.set_linewidth(1.0)
                        ax2.add_artist(legend2)
                        #ax3 = ax1.twinx()
                        #ax3.plot(qw_z * 1e9, d / max(d), linestyle='--', linewidth=0.8, color='red', alpha=ALPHA)
                        #ax3.yaxis.set_visible(False)
                        #legend3 = plt.legend(['Active layer'], fontsize=8, loc=4)
                        #ax3.add_artist(legend3)
                        plt.tight_layout(rect=[0,0.05,1,1])
            plt.show()
    if spectrumplt:
        if sumplt:
            fig.tight_layout()
            plt.show()
        else:
            fig.tight_layout()
            plt.show()
    if factorplt:
        if sumplt:
            total_rate_arr = np.array([sum(total_rate_list[i]) for i in range(len(total_rate_list))])
            spont_rate_arr = np.array([sum(spont_list[i]) for i in range(len(spont_list))])
            enhance_array = total_rate_arr / spont_rate_arr
            fp_array = enhance_array.reshape([3, 3])
            generate_purcell_heatmap(fp_array)
            plt.show()

        else:
            for dip in range(ndip):
                enhance_array = np.array(enhancement_list)[:, dip]
                fp_array = enhance_array.reshape([3, 3])
                generate_purcell_heatmap(fp_array)
                plt.show()


    # 1. Plot the graph nicely, perhaps zoom in?
    # 2. Purcell factor table
    # 3. Plot the emission spectrum, when the transition frequency is mentioned with a vertical line
    # 4. Does the enhancement is the sum of emissions?



def save_updated_dipole_rate(e_path, wv_path, f_array, files_path):
    # files_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\110122_updated_calculation"
    E = load_modes(e_path)
    ef = E[0].ef
    rad = E[0].radius
    points = E[0].points

    z = points[:, 2]  # same for all
    z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 10000)
    dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, points)

    wavetot, z_wv, levelstot, _ = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7

    spectrum_per_dipole = np.zeros([len(init_states), len(f_array)])
    avg_rate_for_z_per_dipole = np.zeros([len(init_states), len(z_linspace)])
    avg_rate_per_dipole = np.zeros([len(init_states)])
    gamma_k_list = [[] for i in range(len(init_states))]
    spectrum_k_list = [[] for i in range(len(init_states))]
    spontaneous_rate_per_dipole = np.zeros([len(init_states)])
    header = r"\updated_dipole_rate_Ef_{}eV_rad_{}um".format(ef, rad)
    file_name_header = files_path + header


    for Ei in E:
        f_k = Ei.frequency
        Ei.normalize(freq2energy(f_k))
        u_z = Ei.e_field[:, 2]
        print("---  f_k = {}THz".format(Ei.freq_str))
        for dip_idx, INIT_STATE in enumerate(init_states):
            psi_i = wavetot[:, INIT_STATE] * np.sqrt(1e9)
            psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
            energy_i = levelstot[INIT_STATE]
            energy_f = levelstot[FINAL_STATE]
            delta_energy = abs(energy_i - energy_f)
            e = cn.e
            f_ij = energy2freq(delta_energy)
            d = e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
            print("------   f_ij = {}THz\n------   Delta E = {}eV\n------   d = {}".format(f_ij / 1e12, delta_energy, d / e * 1e9))


            avg_rate_for_z = Ei.calculate_rate_over_z(z_linspace, u_z, f_ij, d)
            #current_avg_rate = regular_integration_1d(avg_rate_for_z * dipole_density, z_linspace)
            current_spontaneous_rate = spontaneous_emission(d, f_ij)

            ### old way for obtaining a spectrum
            #mode_spectrum = []
            #for i in tqdm(range(len(f_array))):
                #f = f_array[i]
                #arbitrary_freq_for_z = Ei.calculate_rate_over_z(z_linspace, u_z, f, d)
                #mode_spectrum.append(regular_integration_1d(arbitrary_freq_for_z * dipole_density, z_linspace))

            omega = 2 * np.pi * f_array
            omega_k = 2 * np.pi * f_k
            Q = Ei.Q
            S_k = (2 * Q * omega_k) / (np.pi * (4 * (Q ** 2) * (omega - omega_k) ** 2 + omega_k ** 2))
            G_k = regular_integration_1d(avg_rate_for_z * dipole_density, z_linspace)
            mode_spectrum = G_k * S_k
            gamma_k_list[dip_idx].append(G_k)
            spectrum_k_list[dip_idx].append(mode_spectrum)

            spectrum_per_dipole[dip_idx, :] += np.array(mode_spectrum)
            avg_rate_for_z_per_dipole[dip_idx, :] += avg_rate_for_z
            #avg_rate_per_dipole[dip_idx] += current_avg_rate
            spontaneous_rate_per_dipole[dip_idx] = current_spontaneous_rate

    np.save(file_name_header + ".spectrum", spectrum_per_dipole)
    np.save(file_name_header + ".zrate", avg_rate_for_z_per_dipole)
    np.save(file_name_header + ".spont", spontaneous_rate_per_dipole)
    np.save(file_name_header + ".zlinspce", z_linspace)
    np.save(file_name_header + ".farray", f_array)
    np.save(file_name_header + ".dipole", dipole_density)
    np.save(file_name_header + ".Gk", gamma_k_list)
    np.save(file_name_header + ".Sk", spectrum_k_list)
    print("Saved {}\n\n.".format(header))

def load_updated_dipole_rate(e_path, wv_path, f_array, files_path):
    spectrum_per_dipole = None
    avg_rate_for_z_per_dipole = None
    spontaneous_rate_per_dipole = None
    z_linspace = None
    used_f_array = None
    dipole_density = None
    E = load_modes(e_path)
    ef = E[0].ef
    rad = E[0].radius
    header = r"\updated_dipole_rate_Ef_{}eV_rad_{}um".format(ef, rad)
    file_name_header = files_path + header
    try:
        spectrum_per_dipole = np.load(file_name_header + ".spectrum.npy")
        avg_rate_for_z_per_dipole = np.load(file_name_header + ".zrate.npy")
        spontaneous_rate_per_dipole = np.load(file_name_header + ".spont.npy") #/ len(E) # Important fix for the old files
        z_linspace = np.load(file_name_header + ".zlinspce.npy")
        used_f_array = np.load(file_name_header + ".farray.npy")
        dipole_density = np.load(file_name_header + ".dipole.npy")
        gamma_k_list = np.load(file_name_header + ".Gk.npy")
        spectrum_k_list = np.load(file_name_header + ".Sk.npy")
    except:
        print("Couldn't load. Calc & save..... ")
        save_updated_dipole_rate(e_path, wv_path, f_array, files_path)
        spectrum_per_dipole = np.load(file_name_header + ".spectrum.npy")
        avg_rate_for_z_per_dipole = np.load(file_name_header + ".zrate.npy")
        spontaneous_rate_per_dipole = np.load(file_name_header + ".spont.npy")
        z_linspace = np.load(file_name_header + ".zlinspce.npy")
        used_f_array = np.load(file_name_header + ".farray.npy")
        dipole_density = np.load(file_name_header + ".dipole.npy")
        gamma_k_list = np.load(file_name_header + ".Gk.npy")
        spectrum_k_list = np.load(file_name_header + ".Sk.npy")
    print("Integrating for a total rate")
    avg_rate_per_dipole = np.zeros(np.shape(spontaneous_rate_per_dipole))
    for i in range(len(avg_rate_per_dipole)):
        avg_rate_per_dipole[i] = regular_integration_1d(avg_rate_for_z_per_dipole[i,:] * dipole_density, z_linspace)
    return avg_rate_per_dipole, spontaneous_rate_per_dipole, avg_rate_for_z_per_dipole, z_linspace, dipole_density, spectrum_per_dipole, used_f_array, gamma_k_list, spectrum_k_list

def save_divergences(folder_path):
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

    RESOLUTION = 25
    derivative_name = ['dX', 'dY', 'dZ', 'DIV']
    for path in paths:
        if '0.15' in path and '2.6' in path:
            print("--- Bad data folder eV=0.15 rad=2.6um")
            continue
        #path = paths[6]
        E = load_modes(path)
        for Ek in E:
            print('Ef_{}eV_rad_{}um_f_{}THz'.format(Ek.ef, Ek.radius, Ek.freq_str))
            delta_E = freq2energy(Ek.frequency)
            Ek.normalize(delta_E)
            points = Ek.points
            field_k = Ek.e_field
            interp_field = []
            for i in range(3):
                interp_f, grid = interp3d(points, field_k[:, i], RESOLUTION)
                interp_field.append(interp_f)
            interp_field = np.array(interp_field)
            grad_list = regular_grid_grads(interp_field, grid)

            for i,g in enumerate(grad_list):
                file_name = r"\Ef_{}eV_rad_{}um_f_{}THz_RES_{}_{}".format(Ek.ef, Ek.radius, Ek.freq_str, RESOLUTION, derivative_name[i])
                file_path = folder_path + file_name
                #generic_scatter_3d(grid, np.real(g), False)
                np.save(file_path, g)
            file_name = r"\Ef_{}eV_rad_{}um_f_{}THz_RES_{}_{}".format(Ek.ef, Ek.radius, Ek.freq_str, RESOLUTION, derivative_name[-1])
            file_path = folder_path + file_name
            div = sum(grad_list)
            #generic_scatter_3d(grid, np.real(div), False)
            np.save(file_path, div)
            file_name = r"\Ef_{}eV_rad_{}um_f_{}THz_RES_{}_{}".format(Ek.ef, Ek.radius, Ek.freq_str, RESOLUTION, 'grid')
            file_path = folder_path + file_name
            np.save(file_path, grid)

def generate_slow_varying_fields(wavelength):
    INTERP_RESOLUTION = 19
    LAMBDA = 20e-2
    E0 = 2
    # loading the points from the an arbitrary field
    dummy_path = r'C:\Shaked\Technion\QCL_Project\Electrical Field\EM_spatial-distribution\E_f-0.15eV\Ef-0.15eV_2.2um\\'
    dummy_fields = load_modes(dummy_path)
    dummy_E = dummy_fields[0]
    dummy_ef = str(dummy_E.ef)
    dummy_radius = str(dummy_E.radius)
    dummy_frequency = str(dummy_E.frequency / 1e12) + 'e12'
    dummy_Q = str(dummy_E.Q)
    dummy_points = dummy_E.points
    dummy_e_field = dummy_E.e_field
    slow_field = np.array([E0 * np.ones(np.shape(p[2])) + E0 * np.cos(2 * np.pi * p[2] / wavelength) for p in dummy_points])
    vec_slow_field = np.zeros(np.shape(dummy_e_field), dtype='complex_')
    vec_slow_field[:, 2] = slow_field
    E = deepcopy(dummy_E)
    E.e_field = vec_slow_field
    E.e_norms = np.real([complex_3d_norm(e[0], e[1], e[2]) for e in E.e_field])

    # creating a grid and a field to work with
    field_interp, grid = interp3d(dummy_points, vec_slow_field, INTERP_RESOLUTION, super_res_z=True)
    print("creating slow field")
    slow_interp_field = np.array([E0 * np.ones(np.shape(p_grid[2])) + E0 * np.cos(2 * np.pi * p_grid[2] / wavelength) for p_grid in grid])
    print("done slow field")
    vec_slow_interp_field = np.zeros(np.shape(grid), dtype='complex_')
    vec_slow_interp_field[:, 2] = slow_interp_field
    print("Creating cylinder")
    E_interp = CylinderGridField(dummy_path, dummy_ef, dummy_radius, dummy_frequency, dummy_Q, grid, vec_slow_interp_field)
    print("Done cylinder")
    return E, E_interp


def OLD_compare_Gamma_k_methods(wv_path, wavelength):
    E, E_interp = generate_slow_varying_fields(wavelength)
    #E.normalize(freq2energy(E.frequency)) #5
    #E_interp.normalize(freq2energy(E_interp.frequency))
    u_z = E_interp.e_field[:, 2]
    z = E_interp.points[:, 2]
    z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 100000)
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    bandplot[:, 0] = bandplot[:, 0] * 1e-9
    init_states = [2]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7
    dipole_density = generate_dipole_density(dipole_locations(z_linspace), z_linspace, E_interp.points)
    periods = range(12)
    PER_LEN = 30.68e-9

    field_dict = create_func_dict(E_interp.points, u_z)
    #plt.plot(E_interp.points[:,2], abs(u_z))
    #plt.show()

    for INIT_STATE in init_states:
        psi_i = wavetot[:, INIT_STATE] * np.sqrt(1e9)
        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        d = cn.e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
        energy_i = levelstot[INIT_STATE]
        energy_f = levelstot[FINAL_STATE]
        delta_energy = abs(energy_i - energy_f)
        f_ij = energy2freq(delta_energy)
        print(f_ij, E_interp.frequency)

        # Gamma_k part (approximated treatment)

        gamma_k_result = Gamma_k(u=u_z, d=d, Q=E_interp.Q, f=f_ij, f_k=E_interp.frequency, eps_r=None)
        averaged_gamma_k = averaging_over_area(E_interp.points, E_interp.disk_area, z_linspace, gamma_k_result)
        G_k = regular_integration_1d(averaged_gamma_k * dipole_density, z_linspace)

        print("done gamma_k")
        #1
        E_interp.points[:, 0] = np.zeros(np.shape(E_interp.points[:, 0]))
        E_interp.points[:, 1] = np.zeros(np.shape(E_interp.points[:, 1]))

        #extend_periodic_wavefunction(z_wv, psi_f * np.gradient(psi_i,z_wv), z_linspace)
        # plt.plot(z, gamma_k_result)
        # plt.plot(z_linspace, averaged_gamma_k)
        # plt.show()

        # Gamma_k_q part (non approximated)
        # z_wv + per * PER_LEN
        # Gamma_k_q(u_z_q_dict, points_q, psi_i, psi_f, z_wv, Q, f, f_k, area, eps_r=None):
        gamma_per_period_col_gauge = []
        gamma_per_period_with_div = []

        for per in periods:
            print(wavelength, per)
            #plt.plot(z_wv + per * PER_LEN, per + abs(np.gradient(psi_i,z_wv + per * PER_LEN) * psi_f) / max(abs(np.gradient(psi_i,z_wv + 0 * PER_LEN) * psi_f)))
            #print("p_intg / z_intg = {}".format(momentum_int / space_int))
            # probably can become more efficient by working on each period separately
            gamma_res = Gamma_k_q(u_z_q_dict=field_dict, points_q=E_interp.points, psi_f=psi_f, psi_i=psi_i, z_wv=z_wv + per * PER_LEN, Q=E_interp.Q, f=f_ij, f_k=E_interp.frequency, area=E_interp.disk_area)
            gamma_per_period_col_gauge.append(gamma_res[0])
            gamma_per_period_with_div.append(gamma_res[1])
        total_gamma_k_q_coloumb = sum(gamma_per_period_col_gauge)
        total_gamma_k_q_div = sum(gamma_per_period_with_div)
        print(G_k, total_gamma_k_q_coloumb)
        print("col Gamma1/Gamma2 =           {}".format((total_gamma_k_q_coloumb / G_k)))
        print("col Gamma1/Gamma2 * 2/pi =    {}".format((total_gamma_k_q_coloumb / G_k) * (2 / np.pi)))
        return total_gamma_k_q_coloumb * (2 / np.pi), total_gamma_k_q_div * (2 / np.pi), G_k
        #plt.show()


def compare_Gamma_k_methods(Ek, wv_path, plz_plot=False):
    RESOLUTION = 33
    Ek.normalize(freq2energy(Ek.frequency))
    u_x, grid = interp3d(Ek.points, Ek.e_field[:, 0], RESOLUTION, ignore_nan=False)
    u_y, _ = interp3d(Ek.points, Ek.e_field[:, 1], RESOLUTION, ignore_nan=False)
    u_z, _ = interp3d(Ek.points, Ek.e_field[:, 2], RESOLUTION, ignore_nan=False)
    u_z = u_z[::-1]
    u_k = np.array([u_x, u_y, u_z])
    #print("--- generating divergence")
    div_u_k = regular_grid_div(np.conjugate(u_k), grid)

    x, y, z = grid.T
    z_linspace = np.linspace(z.min(), round_micro_meter(z.max(), 4), 100000)
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    nQW = 12

    bandplot[:, 0] = bandplot[:, 0] * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7

    periods = range(nQW)
    PER_LEN = 30.68e-9
    zper = int(len(z_linspace) / nQW)
    MILI_TO_EV = 1000
    band_energy_diff = (levelstot[0] - levelstot[7]) * MILI_TO_EV

    interp_bandplot = np.zeros(np.shape(z_linspace))

    u_z_dict = create_func_dict(grid, u_z)
    div_u_dict = create_func_dict(grid, div_u_k)
    #print("--- iterating over init states")
    dicts_return_list = []
    for INIT_STATE in init_states:
        psi_i = wavetot[:, INIT_STATE] * np.sqrt(1e9)
        psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
        d = cn.e * regular_integration_1d(psi_i * z_wv * psi_f, z_wv)
        energy_i = levelstot[INIT_STATE]
        energy_f = levelstot[FINAL_STATE]
        delta_energy = abs(energy_i - energy_f)
        f_ij = energy2freq(delta_energy)
        # dipole approximation averaged Gamma_k(z)
        gamma_k_result = Gamma_k(u=u_z, d=d, Q=Ek.Q, f=f_ij, f_k=Ek.frequency, eps_r=None)
        averaged_gamma_k = averaging_over_area(grid, Ek.disk_area, z_linspace, gamma_k_result)
        avg_G_k = averaged_gamma_k * dipole_locations(z_linspace)
        #avg_G_k = averaged_gamma_k * generate_dipole_along_z(dipole_locations(z_linspace), z_linspace)

        total_q_rate = np.empty((nQW, len(z_linspace)), dtype='complex_')
        total_q_div_rate = np.empty((nQW, len(z_linspace)), dtype='complex_')
        for per in periods:
            print("Init: {},    Period: {}".format(INIT_STATE, per))
            z_wv_q = z_wv + per * PER_LEN
            qw_z = z_linspace[per * zper : (per + 1) * zper]

            gamma_res = Gamma_k_q(u_z_dict, div_u_dict, grid, psi_i, psi_f, z_wv_q, Ek.Q, f_ij, Ek.frequency, Ek.disk_area, eps_r=None)
            avg_G_k_q = irregular_integration_delaunay_2d(gamma_res[0]) / Ek.disk_area
            avg_G_k_q_with_div = irregular_integration_delaunay_2d(gamma_res[1]) / Ek.disk_area
            z_linspace_slice = (z_linspace <= qw_z.max()) & (z_linspace >= qw_z.min())
            avg_G_k_q_slice = z_linspace_slice * avg_G_k_q * dipole_locations(z_linspace)
            #avg_G_k_q_slice = z_linspace_slice * avg_G_k_q * generate_dipole_along_z(dipole_locations(z_linspace), z_linspace)
            total_q_rate[per, :] = avg_G_k_q_slice
            avg_G_k_q_with_div_slice = z_linspace_slice * avg_G_k_q_with_div * dipole_locations(z_linspace)
            #avg_G_k_q_with_div_slice = z_linspace_slice * avg_G_k_q_with_div * generate_dipole_along_z(dipole_locations(z_linspace), z_linspace)
            total_q_div_rate[per, :] = avg_G_k_q_with_div_slice
            interp_bandplot_func = interpolate.interp1d(bandplot[:, 0] + PER_LEN * per,
                                                        bandplot[:, 1] - band_energy_diff * per,
                                                        kind='linear',
                                                        fill_value=0, bounds_error=False)
            interp_bandplot[per * zper : (per + 1) * zper] = (interp_bandplot_func(qw_z))
        total_k_rate = np.sum(total_q_rate, axis=0) # equivalent to integration, so we need to normalize
        total_k_div_rate = np.sum(total_q_div_rate, axis=0) # equivalent to integration, so we need to normalize
        spont_emission = spontaneous_emission(d, f_ij)
        interp_bandplot = interp_bandplot - min(interp_bandplot)
        d_normalization = regular_integration_1d(dipole_locations(z_linspace), z_linspace)
        avg_G_k_Yz = avg_G_k / d_normalization
        total_k_div_rate_Yz = total_k_div_rate / d_normalization
        #print("Calculated Fp", np.round(average_over_dipoles(avg_G_k, dipole_locations(z_linspace), z_linspace) / spont_emission, 2))
        #print("Calculated Fp", np.round(regular_integration_1d(avg_G_k, z_linspace) / spont_emission, 2))
        # print("Non-approx Fp", average_over_dipoles(total_k_div_rate, dipole_locations(z_linspace), z_linspace) / spont_emission)

        #theory_purcell = np.round((3 / (4 * np.pi ** 2) * ((cn.c / Ek.frequency) / np.sqrt(np.real(Ek.active_eps_r))) ** 3 * Ek.Q / (Ek.disk_area * (z.max() - z.min()))) , 2)
        theory_purcell = np.round((3 / (4 * np.pi ** 2) * (
                    (cn.c / f_ij) / np.sqrt(np.real(disspersion(freq2energy(f_ij))[1]))) ** 3 * Ek.Q / (
                                               Ek.disk_area * (z.max() - z.min()))), 2)
        Gk_dipole_Fp = np.round(regular_integration_1d(avg_G_k_Yz, z_linspace) / spont_emission, 2)
        Gk_divergence_Fp = np.round(np.real(regular_integration_1d(total_k_div_rate_Yz, z_linspace) / spont_emission), 2)
        print("Calculated Fp", Gk_dipole_Fp)
        print("Non-approx Fp", Gk_divergence_Fp)
        print("Theory Fp", theory_purcell)

        w_k = 2 * np.pi * Ek.frequency
        w_if = 2 * np.pi * f_ij
        rho_if = 2 * Ek.Q / np.pi * w_k / ((2 * Ek.Q * (w_if - w_k)) ** 2 + w_k ** 2)
        rho_resonance = 2 * Ek.Q / np.pi * w_k / (w_k ** 2)
        rho_ratio = rho_if / rho_resonance
        Fp_estimation = np.round(theory_purcell * rho_ratio, 2)
        #print("--- Estimation",theory_purcell / rho_resonance * rho_if * 3 / 30.68)
        print("--- Estimation", Fp_estimation)
        print("------- Rho ratio", rho_ratio)
        if plz_plot:
            plt.plot(z_linspace * 1e9, np.real(total_k_rate) / spont_emission, z_linspace * 1e9,
                     np.real(total_k_div_rate) / spont_emission, z_linspace * 1e9, np.real(avg_G_k) / spont_emission)
            plt.plot(z_linspace * 1e9, averaged_gamma_k / spont_emission, '--r', linewidth=0.8)
            #plt.ylabel(r'$\Gamma_k$ [sec$^-$$^1$]')
            plt.ylabel(r'$F_p$ enhancement [1]')
            plt.xlabel('z [nm]')
            plt.plot(z_linspace[:-10] *1e9, interp_bandplot[:-10] / abs(max(interp_bandplot[:-10])) * max(np.real(total_k_rate) /spont_emission ) / 0.95 , 'k--', linewidth=0.3)
            plt.legend(['Coloumn gauge', 'With DIV', 'Dipole approx. QW', 'Dipole approx. interpolated', 'Energy band [a.u]'])
            plt.show()

        return_dict = {'dipole': INIT_STATE, 'f_k': Ek.frequency, 'Ef': Ek.ef, 'radius': Ek.radius, 'resolution': RESOLUTION,'Gk_dipole_Fp': Gk_dipole_Fp,
                       'Gk_divergence_Fp': Gk_divergence_Fp, 'theory_purcell': theory_purcell,
                       'Fp_estimation': Fp_estimation, 'spont_emission': spont_emission, 'avg_G_k': avg_G_k, 'averaged_gamma_k': averaged_gamma_k,
                       'total_k_div_rate': total_k_div_rate, 'total_k_rate': total_k_rate, 'z_linspace': z_linspace, 'interp_bandplot': interp_bandplot, 'd': d, 'f_ij': f_ij}
        dicts_return_list.append(return_dict)

    return dicts_return_list






