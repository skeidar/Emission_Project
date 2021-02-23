import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from utilities import *
import seaborn as sns
import pandas as pd

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

    def plot_dat(self):
        Ntot = 21
        Nper = 7
        z = self.dat[:,0]
        energy_band = self.dat[:, 1]
        wavefuncs = np.array([self.dat[:,i] for i in range(2,Ntot + 2)]).T

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

left_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
left_shift_wavefuncs[:L_wv-Nper,:] = wavefuncs[Nper:,:]
left_shift_wavefuncs[L_wv-1*Nper:,:] = 0 #wavefuncs[-1,:]

right_shift_wavefuncs = np.zeros(np.shape(wavefuncs))
right_shift_wavefuncs[Nper:,:] = wavefuncs[:L_wv-Nper,:]
right_shift_wavefuncs[:Nper,:] = 0 #wavefuncs[0,:]

Npp = Nper
Nnu= N_WV
Nper=6
wavetot2 = np.zeros((len(z), 3 * Nnu))
for iper in range(-1, 2):
    istart = max(0, -iper * Npp)
    # iend = min((Nper-iper)*Npp,Nper*Npp)
    # iend = min((2 * Nper + 1 - iper) * Npp, (Nper + 1) * Npp)
    iend = min((Nper + 1 - iper) * Npp, (Nper + 1) * Npp)
    istart_p = max(0, iper * Npp)
    iend_p = istart_p + (iend - istart)
    wcper = np.zeros(np.shape(wavefuncs))
    print(iper, istart, iend, istart_p, iend_p)

    for inu in range(Nnu):
        wcper[istart_p:iend_p, inu] = wavefuncs[istart:iend, inu]
    wavetot2[:, (iper + 1) * Nnu:(iper + 2) * Nnu] = wcper

zij = np.zeros((3 * N_WV, 3 *N_WV))
wavetot = np.zeros((np.shape(wavefuncs)[0], 3 * np.shape(wavefuncs)[1]))
wavetot[:,:N_WV] = left_shift_wavefuncs
wavetot[:,N_WV:2* N_WV] = wavefuncs
wavetot[:, 2 *N_WV:] = right_shift_wavefuncs



plt.plot(z, wavetot2[:, 0:6], 'b', z, wavetot[:,:N_WV], '--r')
plt.show()
plt.plot(z, wavetot2[:, 6:12], 'b',z, wavetot[:,N_WV:2* N_WV], '--r')
plt.show()
plt.plot(z, wavetot2[:, 12:18], 'b', z, wavetot[:, 2 *N_WV:], '--r')
plt.show()




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


