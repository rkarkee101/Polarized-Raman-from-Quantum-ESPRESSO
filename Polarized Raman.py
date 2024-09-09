
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

cons = 0.033539
CmToEv = 1.2398419e-4  # eV
Kb = 8.6173383e-05  # eV/K
SpeedOfLight = 299792458
AMU = 1.6605402e-27
THzToCm = 1.0e12 / (SpeedOfLight * 100)
raman_xsection = 1.0e24 * 1.054571817e-34 / (2.0 * SpeedOfLight**4 * AMU * THzToCm**3)

###############   INPUTS  ###########
total_modes = 36  # Total number of atoms times 3
frequency_laser = 532  # laser frequency in nm
##########################################

def nm_to_cm(value: float):
    return 10**7 / value

def boson_number(frequency: float, temperature: float):
    return 1.0 / (1.0 - np.exp(-CmToEv * frequency / (temperature * Kb)))

# Define the list of temperatures
temperature= 300

# Create a figure for plotting
plt.figure(figsize=(10, 5))


file = open("dyn_prim.out")
counter = 0
data = []
Raman_xx = []
Raman_yy = []
Raman_zz = []
Raman_xy = []
Raman_xz = []
Raman_yz = []
frequency = []

############ Finding frequency
while True:
    line = file.readline()
    lineSplit = line.split()

    if len(lineSplit) >= 2:
        if lineSplit[1] == 'mode' and lineSplit[4] == 'IR' and lineSplit[5] == 'Raman':
            line = file.readline()
            lineSplit = line.split()
            frequency.append(lineSplit[1])
            flag = 0
            while flag != total_modes - 1:
                for k in np.arange(4):
                    line = file.readline()
                line = file.readline()
                lineSplit = line.split()
                frequency.append(lineSplit[1])
                flag = flag + 1

    if line == '':  # EOF
        break

file1 = open("dyn_prim.out")

while True:
    line1 = file1.readline()
    lineSplit1 = line1.split()

    if len(lineSplit1) >= 2:
        if lineSplit1[0] == 'Raman' and lineSplit1[1] == 'x:':
            Raman_xx.append(lineSplit1[2])
            Raman_xy.append((lineSplit1[3]))
            Raman_xz.append((lineSplit1[4]))
        if lineSplit1[0] == 'Raman' and lineSplit1[1] == 'y:':
            Raman_yy.append((lineSplit1[3]))
            Raman_yz.append((lineSplit1[4]))
        if lineSplit1[0] == 'Raman' and lineSplit1[1] == 'z:':
            Raman_zz.append((lineSplit1[4]))

    if line1 == '':  # EOF
        break

exp = np.array(frequency, dtype=float)

prefac_xx = Intensity_x_x = np.zeros(len(exp))
prefac_yy = Intensity_y_y = np.zeros(len(exp))
prefac_zz = Intensity_z_z = np.zeros(len(exp))
prefac_xy = Intensity_x_y = np.zeros(len(exp))
prefac_xz = Intensity_x_z = np.zeros(len(exp))
prefac_yz = Intensity_y_x = np.zeros(len(exp))
prefac_yz = Intensity_y_z = np.zeros(len(exp))
prefac_zx = Intensity_z_x = np.zeros(len(exp))
Intensity_z_y = np.zeros(len(exp))

boson_occ = boson_number(exp, temperature)
boson_occ[0:3] = 0

for b in range(len(exp)):
    if exp[b] == 0:
        prefac_xx[b] = 0
    elif exp[b] != 0:
        prefac_xx[b] = (boson_occ[b] + 1) * ((nm_to_cm(frequency_laser) - exp[b])**4 / exp[b])

prefac = prefac_xx * raman_xsection

X = np.array(Raman_xx, dtype=float) * cons * prefac
Y = np.array(Raman_yy, dtype=float) * cons * prefac
Z = np.array(Raman_zz, dtype=float) * cons * prefac
XY = YX = np.array(Raman_xy, dtype=float) * cons * prefac
XZ = ZX = np.array(Raman_xz, dtype=float) * cons * prefac
YZ = ZY = np.array(Raman_yz, dtype=float) * cons * prefac


pol_x = np.array([1, 0, 0])
pol_y = np.array([0, 1, 0])
pol_z = np.array([0, 0, 1])
T = np.zeros(len(exp))
T1 = np.zeros(len(exp))

for gamma in np.arange(0, 90, 90):
    for j in range(len(exp)):
        Rzz = [np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0, np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0, 0, 0, 1]
        Ryy = [np.cos(np.radians(gamma)), 0, np.sin(np.radians(gamma)), 0, 1, 0, -np.sin(np.radians(gamma)), 0, np.cos(np.radians(gamma))]
        Rxx = [1, 0, 0, 0, np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0, np.sin(np.radians(gamma)), np.cos(np.radians(gamma))]
        Rx = np.reshape(Rxx, (3, 3))
        Ry = np.reshape(Ryy, (3, 3))
        Rz = np.reshape(Rzz, (3, 3))

        T2 = [X[j], XY[j], XZ[j], YX[j], Y[j], YZ[j], ZX[j], ZY[j], Z[j]]
        New_raman_tensor = np.reshape(T2, (3, 3))

        raman_tensor = np.matmul(New_raman_tensor, Ry)  # Rotated Raman

        Intensity_x_x[j] = (np.matmul(pol_x.T, np.matmul(raman_tensor, pol_x)))**2
        Intensity_y_y[j] = (np.matmul(pol_y.T, np.matmul(raman_tensor, pol_y)))**2
        Intensity_z_z[j] = (np.matmul(pol_z.T, np.matmul(raman_tensor, pol_z)))**2
        Intensity_x_y[j] = (np.matmul(pol_y.T, np.matmul(raman_tensor, pol_x)))**2
        Intensity_x_z[j] = (np.matmul(pol_z.T, np.matmul(raman_tensor, pol_x)))**2
        Intensity_y_x[j] = (np.matmul(pol_x.T, np.matmul(raman_tensor, pol_y)))**2
        Intensity_y_z[j] = (np.matmul(pol_z.T, np.matmul(raman_tensor, pol_y)))**2
        Intensity_z_x[j] = (np.matmul(pol_x.T, np.matmul(raman_tensor, pol_z)))**2
        Intensity_z_y[j] = (np.matmul(pol_y.T, np.matmul(raman_tensor, pol_z)))**2

    def smooth_spectrum(x, intensity, sigma):
        x_high_res = np.linspace(x.min(), x.max() + 100, 1000)
        smoothed_intensity = np.zeros_like(x_high_res)
        for x_peak, intensity_peak in zip(x, intensity):
            gaussian_function = intensity_peak * np.exp(-(x_high_res - x_peak)**2 / (2 * sigma**2))
            smoothed_intensity += gaussian_function
        return x_high_res, smoothed_intensity

    sigma_value = 2
    
    x_smoothed, smoothed_intensity_xx=smooth_spectrum(exp, Intensity_x_x, sigma_value)
    x_smoothed, smoothed_intensity_yy=smooth_spectrum(exp, Intensity_y_y, sigma_value)
    x_smoothed, smoothed_intensity_zz=smooth_spectrum(exp, Intensity_z_z, sigma_value)
    x_smoothed, smoothed_intensity_xy=smooth_spectrum(exp, Intensity_x_y, sigma_value)
    x_smoothed, smoothed_intensity_xz=smooth_spectrum(exp, Intensity_x_z, sigma_value)
    x_smoothed, smoothed_intensity_yx=smooth_spectrum(exp, Intensity_y_x, sigma_value)
    x_smoothed, smoothed_intensity_yz=smooth_spectrum(exp, Intensity_y_z, sigma_value)
    x_smoothed, smoothed_intensity_zx=smooth_spectrum(exp, Intensity_z_x, sigma_value)
    x_smoothed, smoothed_intensity_zy=smooth_spectrum(exp, Intensity_z_y, sigma_value)



# Set plot limits, labels, and legend
plt.xlim(0, 220)

# plt.plot(x_smoothed, smoothed_intensity_xz,  linestyle='-',label='XZ')
plt.plot(x_smoothed, smoothed_intensity_xx,  linestyle='-', label='XX')

plt.ylabel('Raman Intensity (a.u.)', weight='bold', size=20)
plt.xlabel(r'Frequency (cm$^{-1}$)', weight='bold', size=20)
plt.legend()

plt.grid(True)
plt.show()

