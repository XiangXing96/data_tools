import h5py
import mmap
import numpy as np
import time

start_time = time.time()

def parse_dump_file(file_path):
    with open(file_path, 'r+') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # initialize variables
            timestep = []
            num_atoms = []
            lattice_vector = np.zeros([3,3])
            velocity = []

            # read data line by line
            start = 0
            while True:
                # find the end position for this line
                end = mm.find(b'\n', start)
                if end == -1:
                    break
                # read a line
                #line = mm[start:end].strip()
                line = mm[start:end].decode('utf-8').strip()
                start = end + 1

                # analyze the content
                if line.startswith("ITEM: TIMESTEP"):
                    # read time step
                    end = mm.find(b'\n', start)
                    #temp_timestep = int(mm[start:end].strip())
                    temp_timestep = int(mm[start:end].decode('utf-8').strip())
                    timestep.append(temp_timestep)
                    start = end + 1
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    # read number of atoms
                    end = mm.find(b'\n', start)
                    #temp_num_atoms = int(mm[start:end].strip())
                    temp_num_atoms = int(mm[start:end].decode('utf-8').strip())
                    num_atoms.append(temp_num_atoms)
                    start = end + 1
                elif line.startswith("ITEM: BOX BOUNDS"):
                    # read box bounds
                    box_bounds = []
                    
                    for _ in range(3):
                        end = mm.find(b'\n', start)
                        box_bounds.append(list(map(float, mm[start:end].split())))
                        #box_bounds.append(list(map(float, mm[start:end].decode('utf-8').split())))
                        start = end + 1

                    box_bounds = np.array(box_bounds)
                    # change box_bounds to normal lattice vector (https://docs.lammps.org/Howto_triclinic.html)
                    #print(np.shape(box_bounds))
                    if np.shape(box_bounds)[1] == 2:
                        xlo_bound = box_bounds[0,0]
                        xhi_bound = box_bounds[0,1]
                        xy = 0
                        ylo_bound = box_bounds[1,0]
                        yhi_bound = box_bounds[1,1]
                        xz = 0
                        zlo_bound = box_bounds[2,0]
                        zhi_bound = box_bounds[2,1]
                        yz = 0
                    else:
                        xlo_bound = box_bounds[0,0]
                        xhi_bound = box_bounds[0,1]
                        xy = box_bounds[0,2]
                        ylo_bound = box_bounds[1,0]
                        yhi_bound = box_bounds[1,1]
                        xz = box_bounds[1,2]
                        zlo_bound = box_bounds[2,0]
                        zhi_bound = box_bounds[2,1]
                        yz = box_bounds[2,2]
                    
                    xlo = xlo_bound - np.min([0.0,xy,xz,xy+xz])
                    xhi = xhi_bound - np.max([0.0,xy,xz,xy+xz])
                    ylo = ylo_bound - np.min([0.0,yz])
                    yhi = yhi_bound - np.max([0.0,yz])
                    zlo = zlo_bound
                    zhi = zhi_bound

                    lattice_vector = np.array([[xhi-xlo, xy,  xz],
                                               [0,  yhi-ylo,  yz],
                                               [0,   0,  zhi-zlo]]).T

                elif line.startswith("ITEM: ATOMS"):
                    # read atomic information
                    line_spilt = line[12:].split()
                    try:
                        vx_index = line_spilt.index('vx')
                        vy_index = line_spilt.index('vy')
                        vz_index = line_spilt.index('vz')
                    except:
                        raise RuntimeError("\n"
                                    f"     Velocity along x, y and z \n"
                                    f"     are not stored in the specified file, \n"
                                    f"     and please check!"
                                    )
                    
                    temp_configure_v = []

                    for _ in range(temp_num_atoms):
                        end = mm.find(b'\n', start)
                        #atom_data = mm[start:end].split()
                        atom_data = mm[start:end].decode('utf-8').split()
                        temp_v = [(atom_data[vx_index]),atom_data[vy_index],atom_data[vz_index]]
                        temp_v = np.array(temp_v,dtype=float)
                        temp_configure_v.append(temp_v)
                        start = end + 1

                    velocity.append(temp_configure_v)

    # return results
    file.close()
    return {
            'timestep':   np.array(timestep),
            'num_atoms':  np.array(num_atoms),
            'lattice_vector': lattice_vector,
            'velocity':   np.array(velocity)
            }

# read LAMMPS dump file
file = 'LiInCl_200_nve_pos.lammpstrj'
file_stru = 'lammps.data'

dump_data = parse_dump_file(file)
dt = 0.001 # ps

#f_max = 25  # THz
f_internal = 0.001  # THz

timestep = dump_data['timestep']
t_traj = (timestep - timestep[0]) * dt
velocity = dump_data['velocity']

# get the mass of each atom
from ase.io import read
atom = read(file_stru,format='lammps-data')
mass = atom.get_masses()
np.savetxt('mass.txt', mass, fmt='%.6f')

# do FFT 
import pyfftw
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

num_steps = len(timestep)
num_atoms = len(velocity[0])

print(f"Number of steps: {num_steps}")
print(f"Number of atoms: {num_atoms}")

time_inver = t_traj[1] - t_traj[0]
N_tran = (1 / f_internal) / time_inver  # total number of points used to do a FFT
N_tran = int(N_tran)
N_division = int(num_steps / N_tran)
point = []
for i in range(N_division):
    temp_points = [int(i*N_tran), int((i+1)*N_tran)]
    point.append(temp_points)

np.savetxt('point.txt', point, fmt='%d')
point = np.array(point)

fftw_x = np.zeros([num_atoms,N_division,N_tran])
fftw_y = np.zeros([num_atoms,N_division,N_tran])
fftw_z = np.zeros([num_atoms,N_division,N_tran])

for i in range(num_atoms):
    for j in range(N_division):
        index_1 = point[j,0]
        index_2 = point[j,1]
        temp_vx = velocity[index_1:index_2,i,0]
        temp_vy = velocity[index_1:index_2,i,1]
        temp_vz = velocity[index_1:index_2,i,2]

        #print('shape:',np.shape(temp_vx))

        fftw_x[i,j,:] = np.abs(pyfftw.interfaces.numpy_fft.fft(temp_vx, threads=cpu_count()))*time_inver
        fftw_y[i,j,:] = np.abs(pyfftw.interfaces.numpy_fft.fft(temp_vy, threads=cpu_count()))*time_inver
        fftw_z[i,j,:] = np.abs(pyfftw.interfaces.numpy_fft.fft(temp_vz, threads=cpu_count()))*time_inver
    fftw_x[i,:,:] = fftw_x[i,:,:] * mass[i]
    fftw_y[i,:,:] = fftw_y[i,:,:] * mass[i]
    fftw_z[i,:,:] = fftw_z[i,:,:] * mass[i]

freqs = np.fft.fftfreq(N_tran, time_inver)

fftw_x_avg = np.mean(fftw_x, axis=1)
fftw_y_avg = np.mean(fftw_y, axis=1)
fftw_z_avg = np.mean(fftw_z, axis=1)

fftw_x_avg_sum = np.sum(fftw_x_avg, axis=0)
fftw_y_avg_sum = np.sum(fftw_y_avg, axis=0)
fftw_z_avg_sum = np.sum(fftw_z_avg, axis=0)

fftw = fftw_x_avg_sum**2 + fftw_y_avg_sum**2 + fftw_z_avg_sum**2

# Normalize fftw by the area under the curve
fftw_positive = fftw[0:int(N_tran/2)]
freqs_positive = freqs[0:int(N_tran/2)]
area = np.trapezoid(fftw_positive, freqs_positive)
fftw_normalized = fftw_positive / area


# Save the normalized fftw
np.savetxt('fftw_normalized.txt', np.column_stack((freqs_positive, fftw_normalized)), fmt='%.6f')

np.savetxt('fftw.txt', np.column_stack((freqs_positive, fftw_positive)), fmt='%.6f')
