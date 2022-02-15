import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from utilities import *
import seaborn as sns
import pandas as pd
from scipy import constants as cn

mat_path_2020 = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\PRAppl_2020\PRAppl_2020\wavefun_2020.mat"
bands_2020 = sio.loadmat(mat_path_2020)
dat_path_2020 = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\PRAppl_2020\PRAppl_2020\bandplot.dat"
wavefunctions_2020 = np.loadtxt(dat_path_2020)




class Wavefunction(object):
    def __init__(self, folder_name, full_path=False):
        self.name = folder_name
        mat_year = folder_name[-4:]
        mat_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\{}\{}\wavefun_{}.mat".format(folder_name, folder_name, mat_year)
        dat_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\Scripts\data\{}\{}\bandplot.dat".format(folder_name, folder_name)
        self.mat = sio.loadmat(mat_path)
        self.dat = np.loadtxt(dat_path)
        self.full_path = full_path

    def plot_mat(self):
        key = None
        if self.name == "APL_2019":
            key = 'w_fun_Bosco2019'
        if self.name == "PRAppl_2020":
            key = 'w_fun_PRApp2020'
        wave_data = self.mat[key]
        z = wave_data[:,0]
        energy_band = wave_data[:,1]
        ULS = wave_data[:,2]
        LLS = wave_data[:,3]
        plt.plot(z, energy_band, z, ULS, z, LLS)
        plt.title("{} Wavefunctions - Simplied".format(self.name))
        plt.xlabel("z [nm]")
        #plt.ylabel("E [meV]")
        plt.legend(['Structure bands', "Upper", "Lower"])
        plt.show()

    def plot_dat(self, simplify=False):
        Ntot = 21
        Nper = 7
        z = self.dat[:,0]
        energy_band = self.dat[:, 1]
        wavefuncs = np.array([self.dat[:,i] for i in range(2,Ntot + 2)]).T

        ULS_idx = 2
        injector_idx = 1
        LLS_idx = 0
        if simplify:
            plt.figure()
            plt.plot(z, energy_band)
            plt.plot(z, wavefuncs[:, ULS_idx], label='ULS')
            plt.plot(z, wavefuncs[:, LLS_idx], label='LLS')
            plt.plot(z, wavefuncs[:, injector_idx], label='Injector')
            plt.legend()
        else:
            plt.figure()
            plt.plot(z, energy_band)
            for i in range(Nper * 3):
                plt.plot(z, wavefuncs[:,i])
        plt.title("{} Wavefunctions".format(self.name))
        plt.xlabel('z [nm]')
        #plt.ylabel('E [meV]')
        plt.show()

    def dipole_calc(self):
        Ntot = 21
        Nper = 7
        z = self.dat[:,0] * 1e-9
        wavefuncs = np.array([self.dat[:, i] for i in range(2, Ntot + 2)]).T

        print("~~~~~{}".format(self.name))
        for i in range(Nper):
            for j in range(i+1, Nper):

                dip = regular_integration_1d(wavefuncs[:, i] * z * wavefuncs[:, j], z)
                j_eff = j % Nper
                j_per = int(j / Nper)
                print("{} to {}/{} ---> {}".format(i,j_eff,j_per,dip))

    def load_raw_wave(self):
        wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_{}\2-well_{}".format(self.name, self.name)
        #wv_path = r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0"
        wavetot, z_wv, levelstot = load_wavefunction(wv_path)



def normalize_wv(wavefunction, z):
    norm = regular_integration_1d(abs(wavefunction) ** 2, z)
    return wavefunction / np.sqrt(norm)

def load_wavefunction(wv_path):

    wavec = np.loadtxt(wv_path + r"\wsfuncrediag.dat", skiprows=4)
    bandplot_dat = np.loadtxt(wv_path + r"\bandplot.dat", skiprows=4)
    energy_band = bandplot_dat[:,1]
    band_z = bandplot_dat[:,0]
    bandplot = np.zeros([len(energy_band), 2])
    bandplot[:, 0] = band_z
    bandplot[:, 1] = energy_band
    z = wavec[:, 0]
    wavefuncs = wavec[:, 1:]
    N_WV = len(wavefuncs[0, :])
    Zper = 30.68
    #eFd = 0.050 # 50.50.50.50 file
    eFd = 0.0545 # APL-2019
    Nper = int(Zper / (z[1] - z[0]))
    L_wv = len(wavefuncs[:, 0])

    Npp = Nper
    Nnu = N_WV

    #Nper = 6
    Nper = int((L_wv / Npp - 1))

    wsdata = np.loadtxt(wv_path + r"\wslevelsRediag.dat", skiprows=4, max_rows=Nnu + 2)
    levels = wsdata[:, 2]

    wavetot2 = np.zeros((len(z), 3 * Nnu))
    levelstot = np.zeros((3 * Nnu))
    for iper in range(-1, 2):
        istart = max(0, -iper * Npp)
        iend = min((Nper + 1 - iper) * Npp, (Nper + 1) * Npp)
        istart_p = max(0, iper * Npp)
        iend_p = istart_p + (iend - istart)
        wcper = np.zeros(np.shape(wavefuncs))
        #print(istart, iend, istart_p, iend_p, Npp, L_wv / Npp - 1 )

        for inu in range(Nnu):
            wcper[istart_p:iend_p, inu] = wavefuncs[istart:iend, inu]
            levelstot[(iper + 1) * Nnu + inu] = levels[inu] - iper * eFd
        wavetot2[:, (iper + 1) * Nnu:(iper + 2) * Nnu] = wcper

    return wavetot2, z, levelstot, bandplot

def dipole_and_shifts_500_100():

    martin_path = r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0\bandplot.dat"
    wv_martin = np.loadtxt(martin_path)

    wavec = np.loadtxt(r"C:\Shaked\Technion\QCL_Project\Wavefunctions\50.0_10.0_0.0\50.0_10.0_0.0\wsfuncrediag.dat", skiprows=4)
    z = wavec[:, 0]
    wavefuncs = wavec[:, 1:]
    N_WV = len(wavefuncs[0,:])
    Zper = 30.68
    Nper = int(Zper / (z[1]-z[0]))
    L_wv = len(wavefuncs[:,0])
    """
    left_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
    left_shift_wavefuncs[:L_wv-Nper,:] = wavefuncs[Nper:,:]
    left_shift_wavefuncs[L_wv-3*Nper:,:] = 0 #wavefuncs[-1,:]
    
    right_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
    right_shift_wavefuncs[Nper:,:] = wavefuncs[:L_wv-Nper,:]
    right_shift_wavefuncs[:Nper,:] = wavefuncs[0,:]
    right_shift_wavefuncs[L_wv-2*Nper:,:] = 0
    
    wavefuncs[L_wv-2*Nper:,:] = 0
    """
    plt.plot(wavefuncs)
    plt.show()
    left_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
    left_shift_wavefuncs[:L_wv-Nper,:] = wavefuncs[Nper:,:]
    left_shift_wavefuncs[L_wv-1*Nper:,:] = 0 #wavefuncs[-1,:]

    right_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
    right_shift_wavefuncs[Nper:,:] = wavefuncs[:L_wv-Nper,:]
    right_shift_wavefuncs[:Nper,:] = 0 #wavefuncs[0,:]
    print(Nper, N_WV, L_wv)
    Npp = Nper
    Nnu= N_WV
    Nper=6
    wavetot2 = np.zeros((len(z), 3 * Nnu))
    for iper in range(-1, 2):
        istart = max(0, -iper * Npp)
        # iend = min((Nper-iper)*Npp,Nper*Npp)
        #iend = min((2 * Nper + 1 - iper) * Npp, (Nper + 1) * Npp)
        iend = min((Nper + 1 - iper) * Npp, (Nper + 1) * Npp)
        istart_p = max(0, iper * Npp)
        iend_p = istart_p + (iend - istart)
        wcper = np.zeros(np.shape(wavefuncs))
        print(istart, iend, istart_p, iend_p)
        for inu in range(Nnu):
            wcper[istart_p:iend_p, inu] = wavefuncs[istart:iend, inu]
        wavetot2[:, (iper + 1) * Nnu:(iper + 2) * Nnu] = wcper

    zij = np.zeros((3 * N_WV, 3 *N_WV))

    wavetot = np.zeros((np.shape(wavefuncs)[0], 3 * np.shape(wavefuncs)[1]))
    wavetot[:,:N_WV] = left_shift_wavefuncs
    wavetot[:,N_WV:2* N_WV] = wavefuncs
    wavetot[:, 2 *N_WV:] = right_shift_wavefuncs


    """
    plt.plot(z, wavetot2[:, 0:6], 'b', z, wavetot[:,:N_WV], '--r')
    plt.show()
    plt.plot(z, wavetot2[:, 6:12], 'b',z, wavetot[:,N_WV:2* N_WV], '--r')
    plt.show()
    plt.plot(z, wavetot2[:, 12:18], 'b', z, wavetot[:, 2 *N_WV:], '--r')
    plt.show()
    """
    for i in range(3 * N_WV):
        for j in range(3 * N_WV):
            zij[i][j] = None
            if i is not j:
                wv_i = wavetot[:, i]  # units of 1/sqrt(nm)
                wv_j = wavetot[:, j]  # units of 1/sqrt(nm)
                zij[i][j] = regular_integration_1d(wv_i * wv_j * z, z)
    Index = [f'w={i % 6}, p={int(i / 6) - 1}' for i in range(3*N_WV)]
    Cols = [f'w={i % 6}, p={int(i / 6) - 1}' for i in range(3*N_WV)]
    df = pd.DataFrame(zij, index=Index, columns=Cols)
    plt.figure()
    sns.set(font_scale=0.6)
    sns.heatmap(df, annot=True,annot_kws={"fontsize":5},fmt='.3f')
    plt.title("Zij [nm]")
    plt.ylabel("Wavefunc, Period")
    plt.xlabel("Wavefunc, Period")
    plt.show()
    plt.figure()

    plt.plot(z,wavetot[:,:])
    plt.show()

def testing_shifts_APL():
    wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_APL_2019\2-well_APL_2019"
    wavetot, z_wv, levelstot = load_wavefunction(wv_path)
    wvtot = wavetot[:,:7]
    lvlstot = levelstot[:7]
    # sanity check
    plt.plot(z_wv, abs(wvtot)**2 * 0.3 + lvlstot)
    plt.show()

    N_WV = len(wvtot[0,:])
    Zper = 30.68
    Nper = int(Zper / (z_wv[1] - z_wv[0]))
    L_wv = len(wvtot[:, 0])

    left_shift_wavefuncs = np.zeros(np.shape(wvtot))
    left_shift_wavefuncs[:L_wv - Nper, :] = wvtot[Nper:, :]
    left_shift_wavefuncs[L_wv - 1 * Nper:, :] = wvtot[-1,:]

    right_shift_wavefuncs = np.zeros(np.shape(wvtot))
    right_shift_wavefuncs[Nper:, :] = wvtot[:L_wv - Nper, :]
    right_shift_wavefuncs[:Nper, :] = wvtot[0, :]

    full_range_wvtot = np.zeros((np.shape(wvtot)[0], 3 * np.shape(wvtot)[1]))
    full_range_wvtot[:, :N_WV] = left_shift_wavefuncs
    full_range_wvtot[:, N_WV:(2 * N_WV)] = wvtot
    full_range_wvtot[:, (2 * N_WV):] = right_shift_wavefuncs

    zij = np.zeros((3 * N_WV, 3 * N_WV))
    for i in range(3 * N_WV):
        for j in range(3 * N_WV):
            zij[i][j] = None
            if i is not j:
                wv_i = full_range_wvtot[:, i]  # units of 1/sqrt(nm)
                wv_j = full_range_wvtot[:, j]  # units of 1/sqrt(nm)
                zij[i][j] = regular_integration_1d(wv_i * wv_j * z_wv, z_wv)
    Index = [f'w={i % N_WV}, p={int(i / N_WV) - 1}' for i in range(3 * N_WV)]
    Cols = [f'w={i % N_WV}, p={int(i / N_WV) - 1}' for i in range(3 * N_WV)]
    df = pd.DataFrame(zij, index=Index, columns=Cols)
    plt.figure()
    sns.set(font_scale=0.6)
    sns.heatmap(df, annot=True, annot_kws={"fontsize": 5}, fmt='.3f')
    plt.title("Zij [nm]")
    plt.ylabel("Wavefunc, Period")
    plt.xlabel("Wavefunc, Period")
    plt.show()
    plt.figure()
    print(regular_integration_1d(full_range_wvtot[:, 7] * full_range_wvtot[:, 8] * z_wv, z_wv))

def plot_wavefunctions_along_z(wv_path, zdush):
    wavetot, z_wv, levelstot, bandplot = load_wavefunction(wv_path)
    z_linspace = np.linspace(zdush.min(), round_micro_meter(zdush.max(), 4), 10000)
    print(z_linspace[0], z_linspace[-1])
    z_wv = z_wv * 1e-9
    bandplot[:,0] = bandplot[:,0] * 1e-9
    init_states = [2, 1]  # states 0, 8 -> [2 (ULS), 1 (Injector)]
    FINAL_STATE = 0  # state 7
    nQW = 12
    perLen = 30.68e-9
    psi_f = wavetot[:, FINAL_STATE] * np.sqrt(1e9)
    psi_uls = wavetot[:, init_states[0]] * np.sqrt(1e9)
    psi_inj = wavetot[:, init_states[1]] * np.sqrt(1e9)
    psi_i = [psi_uls, psi_inj]
    total_psi_uls = np.zeros(np.shape(z_linspace))
    total_psi_inj = np.zeros(np.shape(z_linspace))
    total_psi_f = np.zeros(np.shape(z_linspace))
    band_energy_diff = (levelstot[0] - levelstot[7]) * 1000
    f_energy_diff = (levelstot[0] - levelstot[7]) * 1000
    print(f_energy_diff)
    i_energy_diff = [(levelstot[2] - levelstot[9]) * 1000 , levelstot[1] - levelstot[8]]
    print(i_energy_diff)
    for i in range(nQW):
        OFFSET = 0e-9
        interp_psi_f_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_f) ** 2, kind='cubic',
                                                 fill_value=0, bounds_error=False)
        interp_psi_i_func = interpolate.interp1d(z_wv + perLen * i + OFFSET, abs(psi_i[0]) ** 2, kind='cubic',
                                                 fill_value=0, bounds_error=False)
        interp_bandplot_func = interpolate.interp1d(bandplot[:,0] + perLen * i + OFFSET, bandplot[:,1], kind='linear',
                                                 fill_value=0, bounds_error=False)
        periods_args = np.logical_and(z_linspace >= (z_wv + perLen * i + OFFSET).min(), z_linspace <= (z_wv + perLen * i + OFFSET).max())
        interp_psi_f = interp_psi_f_func(z_linspace) / 2e6 - f_energy_diff * i + levelstot[0] * 1000
        #bold_interp_psi_f = interp_psi_f_func(z_linspace[]) / np.sqrt(1e9) - f_energy_diff * i + levelstot[0] * 10
        interp_psi_i = interp_psi_i_func(z_linspace) / 2e6 - i_energy_diff[0] * i + levelstot[2] * 1000
        interp_bandplot = interp_bandplot_func(z_linspace) - band_energy_diff * i #+ levelstot[0] * 10
        plt.plot(z_linspace, interp_psi_f, color='#365ba6', linestyle='--', linewidth=0.5)
        plt.plot(z_linspace[periods_args], interp_psi_f[periods_args], color='#365ba6', linewidth=0.7)
        plt.plot(z_linspace, interp_psi_i, color='#a63636', linestyle='--', linewidth=0.5)
        plt.plot(z_linspace[periods_args], interp_psi_i[periods_args], color='#a63636', linewidth=0.7)
        plt.plot(z_linspace[periods_args], interp_bandplot[periods_args], 'k', linewidth=0.5)
    plt.show()



if __name__ == "__main__":
    wv = Wavefunction("APL_2019")
    wv_path = r"C:\Shaked\Technion\QCL_Project\Electrical Field\2-well_APL_2019\2-well_APL_2019"
    wavetot, z_wv, levelstot = load_wavefunction(wv_path)
    z_wv = z_wv * 1e-9
    ULS_index = 2
    injector_index = 1
    LLS_index = 0


    testing_shifts_APL()


