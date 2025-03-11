from ase.io import read
import h5py
import os
import mmap
import numpy as np
import time
import numba

start_time = time.time()

def parse_dump_file(file_path):
    with open(file_path, 'r+') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # initialize variables
            timestep = []
            num_atoms = []
            lattice_vector = np.zeros([3,3])
            velocity = []
            position = []
            vx_index = 0
            posx_index = 0

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
                       # box_bounds.append(list(map(float, mm[start:end].split())))
                        box_bounds.append(list(map(float, mm[start:end].decode('utf-8').split())))
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
                    if vx_index is not None:
                        try:
                            vx_index = line_spilt.index('vx')
                            vy_index = line_spilt.index('vy')
                            vz_index = line_spilt.index('vz')
                        except:
                            vx_index = None
                            vy_index = None
                            vz_index = None
                            print("Velocity along x, y and z are not stored in the specified file!")

                    if posx_index is not None:
                        try:
                            posx_index = line_spilt.index('xu')
                            posy_index = line_spilt.index('yu')
                            posz_index = line_spilt.index('zu')
                        except:
                            posx_index = None
                            posy_index = None
                            posz_index = None
                            print("Positions are not stored in the specified file!")
                    
                    
                    temp_configure_v = []
                    temp_configure_p = []

                    for _ in range(temp_num_atoms):
                        end = mm.find(b'\n', start)
                        #atom_data = mm[start:end].split()
                        atom_data = mm[start:end].decode('utf-8').split()
                        
                        if vx_index is not None:
                            temp_v = [(atom_data[vx_index]),atom_data[vy_index],atom_data[vz_index]]
                            temp_v = np.array(temp_v,dtype=float)
                            temp_configure_v.append(temp_v)

                        if posx_index is not None:
                            temp_p = [atom_data[posx_index],atom_data[posy_index],atom_data[posz_index]]
                            temp_p = np.array(temp_p,dtype=float)
                            temp_configure_p.append(temp_p)
                        
                        start = end + 1

                    velocity.append(temp_configure_v)
                    position.append(temp_configure_p)

    # return results
    file.close()

    if vx_index is not None and posx_index is not None:
        return {
            'timestep':   np.array(timestep),
            'num_atoms':  np.array(num_atoms),
            'lattice_vector': lattice_vector,
            'velocity':   np.array(velocity),
            'position':   np.array(position)
            }
    elif vx_index is None and posx_index is not None:
        return {
            'timestep':   np.array(timestep),
            'num_atoms':  np.array(num_atoms),
            'lattice_vector': lattice_vector,
            'position':   np.array(position)
            }
    elif vx_index is not None and posx_index is None:
        return {
            'timestep':   np.array(timestep),
            'num_atoms':  np.array(num_atoms),
            'lattice_vector': lattice_vector,
            'velocity':   np.array(velocity)
            }

 
@numba.jit(nopython=True, parallel=True)
def calculate_msd_type(atom_pos,atom_type,msd_blocks,msd_atom_Block,Blocks, N_t, N_skip):
    # atom_pos: atomic positions, shape should be (N_total,N_atom,3)
    # atom_type: atomic types, shape should be (N_atom)
    # N_type: number of types
    # msd_blocks: mean square displacement, shape should be (N_B, N_type, N_t, 3)
    # msd_atom_Block: mean square displacement for each atom, shape should be (N_B, N_atom, N_t, 3)
    # N_B: number of blocks used for average
    # N_t: length of time for msd

    # calculate msd
    N_sblock = Blocks[0,1]-Blocks[0,0]+1 # number of data points for one block
    N_average = N_sblock - N_t # number of data points for average in one block
    N_frame_used = N_average // N_skip # number of frames used for average    

    #print(N_sblock)
    #print(N_average)
    #print(N_t)

    for i,j in enumerate(Blocks):
        #print(j)
        temp_data = atom_pos[j[0]:j[1]+1]
        #print(np.shape(temp_data[0]))
        #print(temp_data[0,0])
        for k in numba.prange(N_t):
            for m in range(0,N_average,N_skip):
                msd_atom_Block[i,:,k,:] +=  (temp_data[k+m] - temp_data[m])**2 / N_frame_used
            
            #print((temp_data[500,0] - temp_data[0,0])**2)
            #print(msd_atom_Block[i,0,k])
    
    for i in range(N_B):
        for j,k in enumerate(atom_type):
            msd_blocks[i,k] += msd_atom_Block[i,j] / np.count_nonzero(atom_type == k)

    return msd_blocks




# read LAMMPS dump file
file_traj = './LiInCl_200_nve_pos.lammpstrj'
file_symmetry_hdf5 = 'LiSi_200_vel.hdf5'
file_structure = './lammps.data'
N_B = 4 # split the total time into N_B blocks to do the average
N_t = 10000 # length of time for msd
N_skip = 1 # skip N_skip frames to do the average in calculation of msd

structure = read(file_structure,format='lammps-data')
atom_symbols = structure.get_chemical_symbols()
from collections import Counter
unique_elements = list(Counter(atom_symbols).keys())
atom_type = []
N_type = len(unique_elements)

for i in atom_symbols:
    atom_type.append(unique_elements.index(i))

for i, j in enumerate(unique_elements):
    print(j + " -> " + str(i) + ", " + str(atom_type.count(i)) + " atoms in system",flush=True)

atom_type = np.array(atom_type)


#print(unique_elements)
dump_data = parse_dump_file(file_traj)
atom_pos = np.copy(dump_data['position'])

print(atom_pos[0,0])

del dump_data

print(atom_pos[100,0])

print((atom_pos[0,0] - atom_pos[500,0])**2)

temp_N_t = len(atom_pos)
temp_N = temp_N_t//N_B
N_total = temp_N*N_B
atom_pos = atom_pos[:N_total]
print("Steps in raw data: " + str(temp_N_t) + "; Used steps: " + str(N_total),flush=True)

Blocks = np.zeros([N_B,2],dtype=int)
id = np.linspace(0,N_total-1,N_B+1,dtype=int)
Blocks[:,0] = id[0:N_B] + 1
Blocks[0,0] = 0
Blocks[:,1] = id[1:N_B+1]

print(np.dtype(Blocks[1,1]))



msd_blocks = np.zeros([N_B,N_type,N_t,3])
msd_atom_Block = np.zeros([N_B,len(atom_pos[0]),N_t,3])
test = calculate_msd_type(atom_pos,atom_type,msd_blocks,msd_atom_Block,Blocks, N_t, N_skip)


np.savetxt('msd_blocks.txt',np.mean(test[:,0,:,:],axis=0))

import matplotlib.pyplot as plt

# Plot MSD for each type
for i in range(N_B):    
    msd = test[i,0,:,:] 
    print(np.shape(msd))
    #msd_total = np.sum(msd_mean, axis=1)
    time_step = range(np.shape(msd)[0]) #* 0.01  # unit is ps
    print(time_step)
    #print(time_step)
    plt.plot(time_step,msd[:,0], label=f'Block {i}')

plt.xlabel('Time step')
plt.ylabel('Mean Square Displacement (MSD)')
plt.legend()
plt.title('MSD vs Time step for different blocks')
plt.savefig("msd_block")

#print("Timestep:", dump_data['timestep'])
#print("Number of atoms:", dump_data['num_atoms'])
#print("Box bounds:", dump_data['box_bounds'])
#print("First atom:", dump_data['atoms'][0])

#velocity = dump_data['velocity']
#print(np.shape(velocity))

#print(velocity[0])

#hdf5_file = h5py.File(file_symmetry_hdf5, "w")
#hdf5_file.create_dataset('timestep', data=dump_data['timestep'], compression='gzip', compression_opts=9)
#hdf5_file.create_dataset('number_of_atoms', data=dump_data['num_atoms'], compression='gzip', compression_opts=9)
#hdf5_file.create_dataset('lattice_vector', data=dump_data['lattice_vector'], compression='gzip', compression_opts=9)
#hdf5_file.create_dataset('velocity', data=dump_data['velocity'], compression='gzip', compression_opts=9)
#hdf5_file.close()



# for consuming time
end_time = time.time()
execution_time = end_time - start_time
print(f"Consuming time: {execution_time} seconds")
