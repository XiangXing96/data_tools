'''
This script is written to transform the 2nd-order force constant file (.xml) from alamode 
into that in TDEP format (outfile.forceconstant). An outfile.forceconstant is used as the
reference of symmetry for force constant components, which is produced by TDEP with the 
same supercell.  
The script is written by XIANG Xing from HKUST (xxiangad@connect.ust.hk)
'''
import sys
from functools import reduce
import copy
from pprint import pprint
import struct
from os.path import splitext
from types import CellType
import numpy as np
from collections import deque

from numpy.core.fromnumeric import sort

# scale for the unit from Ry/a0^2 to eV/Å^2
# 1 Ry = 13.60569253 eV, 1 a0 = 0.5291772108 A
# 1 Ryd = 4.35974394e-18 / 2.0 J
# Born2A = 0.5291772108
# unit_scale_2nd = 13.60569253 /  pow(0.5291772108, 2)
# unit_scale_3rd = 13.60569253 /  pow(0.5291772108, 3)

Born2A = 0.52917721092
unit_scale_2nd = 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 2)
unit_scale_3rd = 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 3)
unit_scale_4th = 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 4)

# Set parameters

alamode_file = 'alamode.xml'                 # force constant file in the format of .xml from alamode
tdep_input_fc = "outfile.forceconstant"             # force constant file in the format of outfile.forceconstant from tdep
tdep_output_fc = "infile_test.forceconstant"             # force constant file in the format of infile.forceconstant to perturbo and tdep
Npatom = 10                                         # the number of atoms in primitive cell 
Nx = 2                                              # scell along x direction
Ny = 2                                              # scell along y direction
Nz = 2                                              # scell along z direction
scell = [Nx, Ny, Nz]
Total_cell =  Nx * Ny * Nz
Nsatom = int(Npatom * Total_cell)                     # the number of atoms in super cell

# Obtain model structure, read alamdoe force constant data
print("Read alamode force constant file.")

prim_lattice = np.zeros([3,3])                      # the lattice vector of prmitive cell
ss_lattice = np.zeros([3,3])                      # the lattice vector of super cell
atom_pos = np.zeros([Nsatom,3])                     # the position of atoms in supercell
index = np.zeros([Nsatom,1])                        # atomic index in supercell
atom_type = np.zeros([Nsatom,1])                    # atomic type in supercell, 1 for Hf, 2 for O
atom_in_cell = np.zeros([Total_cell, Npatom])       # classify atoms into cells
cell_address = np.zeros([Total_cell,3])
atom_pair_cell = []                                 # relative cell address, ws cell

fid = open(alamode_file)
lines = deque(fid.readlines())

while len(lines) > 0:
    line = lines.popleft()

    if "<LatticeVector>" in line:
        for i in range(3):
            line = lines.popleft()
            temp_x = float(line.split()[1])
            temp_y = float(line.split()[2])
            temp_z = line.split()[3]
            temp_z = float(temp_z[:-5])
            ss_lattice[i,0] = temp_x * Born2A
            ss_lattice[i,1] = temp_y * Born2A
            ss_lattice[i,2] = temp_z * Born2A

            prim_lattice[i,0] = temp_x / Nx * Born2A
            prim_lattice[i,1] = temp_y / Ny * Born2A
            prim_lattice[i,2] = temp_z / Nz * Born2A

    if "<Position>" in line:
        for i in range(Nsatom):
            line = lines.popleft()
            #print(line)
            temp_index = line.split()[1]
            temp_index = int(temp_index.split("\"")[1])
            index[i] = temp_index

            temp_element = line.split()[2]
            temp_element = temp_element.split("\"")[1]
            if temp_element == 'Hf':
                atom_type[i] = 0
            else:
                atom_type[i] = 1
            
            temp_x = line.split()[3]
            temp_y = line.split()[4]
            temp_z = line.split()[5]
            temp_z = temp_z[:-6]
            atom_pos[i,0] = float(temp_x)
            atom_pos[i,1] = float(temp_y)
            atom_pos[i,2] = float(temp_z)

    if "<Translations>" in line:
        for i in range(Nsatom):
            line = lines.popleft()

            temp_line = line.split()[1]
            temp_line = int(temp_line.split('\"')[1])
            temp_colum = line.split()[2]
            temp_colum = int(temp_colum.split('\"')[1])
            temp_index = line.split()[2]
            temp_index = temp_index.split('>')[1]
            temp_index = int(temp_index[:-5])
            atom_in_cell[temp_line-1,temp_colum-1] = temp_index
        
        # calculate cell address
        for i in range(Total_cell):

            index_here = int(atom_in_cell[i,0])

            #print(index_here)
            
            index_prim = int(atom_in_cell[0,0])
            distance = atom_pos[index_here - 1,:] - atom_pos[index_prim - 1,:]

            #print(atom_pos[index_here - 1,:] )
            #print(atom_pos[index_prim - 1,:] )
            #print(distance)
            

            cell_address[i,0] = round(distance[0]*Nx)
            cell_address[i,1] = round(distance[1]*Ny)
            cell_address[i,2] = round(distance[2]*Nz)

    if "<HARMONIC>" in line:
        # Construct matrix to store pairs of atoms
        pairs_of_atoms = np.zeros((Total_cell, Npatom * Nsatom, 2)) # First component indicates the position of cell, the next two dimensions are the information of pairs of atoms
        for i in range(Npatom):
            pairs_of_atoms[0, i*Nsatom:(i+1)*Nsatom, 0] = atom_in_cell[0,i]
        for i in range(Npatom):
            pairs_of_atoms[0, i*Nsatom:(i+1)*Nsatom, 1] = np.arange(1,Nsatom+1)

        # Find correspondence
        for i in range(1,Total_cell):
            for j in range(Npatom):
                pairs_of_atoms[i, j*Nsatom:(j+1)*Nsatom, 0] = atom_in_cell[i, j]
            for j in range(Npatom * Nsatom):
                cell_address2 = np.argwhere(atom_in_cell == pairs_of_atoms[0, j, 1])[0]  # finding the cell address (line) and corresponding atom in pri cell (column)for the second atom in pair
                address_diff = cell_address[cell_address2[0], :] - cell_address[0, :]           # the address difference of a pair of atoms with the first in the prim cell
                cell_address_of_2nd = cell_address[i, :] + address_diff                         # the cell address of 2nd atom with the first not in prim cell
                for k in range(3):
                    if cell_address_of_2nd[k] > scell[k]-1 :
                        cell_address_of_2nd[k] = cell_address_of_2nd[k] - scell[k]
                    elif cell_address_of_2nd[k] < 0 :                                           # 0 need to be modified for different expansion directions!!!!!
                        cell_address_of_2nd[k] = cell_address_of_2nd[k] + scell[k]
                x_temp = np.argwhere(np.abs(cell_address[:, 0] - cell_address_of_2nd[0]) < 0.000001)
                y_temp = np.argwhere(np.abs(cell_address[:, 1] - cell_address_of_2nd[1]) < 0.000001)
                z_temp = np.argwhere(np.abs(cell_address[:, 2] - cell_address_of_2nd[2]) < 0.000001)
                cell_index_temp = reduce(np.intersect1d, [x_temp, y_temp, z_temp])              # the line for cell_address of the second atom in the pair with the first atom in the ith cell
                pairs_of_atoms[i, j, 1] = atom_in_cell[cell_index_temp[0], cell_address2[1]]

        print("Begin to read 2nd-order force constant")
        # Construct matrix to store atom_pair and elements of force constant for 2nd-order force constant
        atom_pair = np.zeros((Nsatom * Nsatom, 2))
        for i in range(Nsatom):
            atom_pair[i*Nsatom:(i+1)*Nsatom, 0] = i+1
        for i in range(Nsatom):
            atom_pair[i*Nsatom:(i+1)*Nsatom, 1] = np.arange(1,Nsatom+1)

        fc_matrix = np.zeros((Nsatom * Nsatom,3,3))         # construct matrix to store force constant

    if "FC2 pair1=" in line:
        atom1_index = line.split()[1]
        atom1_index = int(atom1_index[7:])
        atom1_index = atom_in_cell[0,atom1_index-1]   # the atom must be in the primitive cell
        atom1_direc = line.split()[2]                 # 1 for x, 2 for y, 3 for z
        atom1_direc = int(atom1_direc[:-1])

        atom2_index = line.split()[3]
        atom2_index = int(atom2_index[7:])
        atom2_direc = int(line.split()[4])

        fc_temp = line.split('>')[1]
        fc_temp = float(fc_temp[:-5])

        fc_index = np.where((atom_pair == (atom1_index, atom2_index)).all(axis=1))[0]
        fc_index = fc_index[0]

        fc_matrix[fc_index][atom1_direc-1][atom2_direc-1] += fc_temp * unit_scale_2nd


    if "</HARMONIC>" in line:

        fid.close()

        # Fill fc_matrix with data from other cells
        for i in range(1, Total_cell):
            for j in range(Npatom * Nsatom):
                line_index_temp_1st = np.where((atom_pair == pairs_of_atoms[0, j, :]).all(axis=1))[0]                             # obtain the position of pairs with 1st atom in 1st cell in global matrix
                line_index_temp_1st = line_index_temp_1st[0]
                line_index_temp_new = np.where((atom_pair == pairs_of_atoms[i, j, :]).all(axis=1))[0]                              # obtain the position of pairs with 1st atom in ith cell in global matrix
                line_index_temp_new = line_index_temp_new[0]

                fc_matrix[line_index_temp_new, :, :] = fc_matrix[line_index_temp_1st, :, :]

        print('Begin to write 2nd-order force constant file')
        
        # Re-adjust cell address to introduce negative position
        re_cell_address = copy.deepcopy(cell_address)
        Tcell = int(Nx * Ny * Nz)
        for i in range(Tcell):
            if re_cell_address[i,0] > Nx/2:
                re_cell_address[i,0] = re_cell_address[i,0] - Nx
            if re_cell_address[i,1] > Ny/2:
                re_cell_address[i,1] = re_cell_address[i,1] - Ny
            if re_cell_address[i,2] > Nz/2:
                re_cell_address[i,2] = re_cell_address[i,2] - Nz


for i in atom_pair:#[24:27]:#[0:5]:
    temp_a1 = int(i[0])  # atomic index in position matrix
    temp_a2 = int(i[1])  # atomic index in position matrix
    temp_a1_pos = atom_pos[temp_a1-1]
    temp_a2_pos = atom_pos[temp_a2-1]
    # for periodical boundary condition
    diff_pos = temp_a2_pos - temp_a1_pos
    for j in range(3):
        diff_pos[j] = diff_pos[j] - round(diff_pos[j],0)
    temp_a2_new = temp_a1_pos + diff_pos

    # find the corresponding atom of a2 atom in unitcell of a1 atom
    temp_ss_index_uc = np.argwhere(atom_in_cell == temp_a1)[0]
    temp_ss_index_atom = np.argwhere(atom_in_cell == temp_a2)[0]
    #print(temp_ss_index_uc)
    temp_a2_index_a1uc = atom_in_cell[temp_ss_index_uc[0],temp_ss_index_atom[1]]
    #print(temp_a2_index_a1uc)
    temp_a2_tran_pos = atom_pos[int(temp_a2_index_a1uc)-1]

    # get ws cell address
    diff_a2_a2uc = temp_a2_new - temp_a2_tran_pos
    distance_in_real = np.matmul(diff_a2_a2uc, ss_lattice)
    inverse_pre_lattice = np.linalg.inv(prim_lattice)
    temp_relative_cell = np.matmul(distance_in_real,inverse_pre_lattice)
    int_temp_relative_cell = [round(temp_relative_cell[0]),round(temp_relative_cell[1]),round(temp_relative_cell[2])]
    atom_pair_cell.append(int_temp_relative_cell)
    
#print(atom_pair_cell)
#print(np.max(np.max(atom_pair_cell)))
#print(atom_in_cell)


# read tdep forceconstant file as reference file
print("Read tdep_input reference file")

neighbours = []
uc_map = []                   # the format is i, j, k. i is atomic index in unit cell, j is neighbour index, k is the index of focused atom 
ws_cell = []

fid = open(tdep_input_fc,'r')
lines = deque(fid.readlines())

line = lines.popleft()
atom_N_primi = int(line.split()[0])
# check number of atoms in unitcell
if atom_N_primi != Npatom:
    print("Error: the numer of atoms in alamode is different from that tdep!")
    sys.exit()
line = lines.popleft()
cutoff = float(line.split()[0])                                         # the cutoff in real space, unit is A

while len(lines) > 0:
    line = lines.popleft()
    if "How many neighbours does atom" in line:
        temp_neighbour = int(line.split()[0])
        temp_index = int(line.split()[6])
        neighbours.append([temp_neighbour,temp_index])
        for i in range(temp_neighbour):
            line = lines.popleft()
            #print(line)
            temp_i = int(line.split()[0])
            temp_j = int(line.split()[11])
            temp_k = int(line.split()[14])
            uc_map.append([temp_i,temp_j,temp_k])
        
            line = lines.popleft()
            temp_x = int(float(line.split()[0]))
            temp_y = int(float(line.split()[1]))
            temp_z = int(float(line.split()[2]))
            ws_cell.append([temp_x,temp_y,temp_z])

            # discard the initial force constant elements
            line = lines.popleft()
            line = lines.popleft()
            line = lines.popleft()

    if "Born charges" in line:
        fid.close()
neighbours = np.array(neighbours)
uc_map = np.array(uc_map)
ws_cell = np.array(ws_cell)
# find corresponding force constant elements
#print(re_cell_address)
#print(ws_cell)
#np.savetxt('ws_cell.txt',ws_cell)
#np.savetxt('atom_cell.txt',atom_pair_cell)


# write data to tdep forceconstant file
print("write tdep_output forceconstant file")

fid = open(tdep_output_fc, 'wt')
print('      {:>10d}               How many atoms per unit cell'.format(Npatom), file=fid)
print('      {:>18.7f}       Realspace cutoff (A)'.format(cutoff), file=fid)
        
count = 0
for i in range(Npatom):
    N_nergh = int(neighbours[i,0])
    id_atom = int(neighbours[i,1])
    print('      {:>10d}               How many neighbours does atom {:d} have'.format(N_nergh,id_atom), file=fid)
    for j in range(N_nergh):
        print('      {:>10d}       In the unit cell, what is the index of neighbour  {:d} of atom  {:d}'.format(uc_map[count,0],uc_map[count,1],uc_map[count,2]), file=fid)
        print('{:>20d}  {:>20d}  {:>20d}'.format(int(ws_cell[count,0]), int(ws_cell[count,1]), int(ws_cell[count,2])), file=fid)

        # find the second atom in this pair
        temp_i1_uc = uc_map[count,2]
        temp_i1_ss = atom_in_cell[0,temp_i1_uc-1]
        temp_i2_uc = uc_map[count,0]
        temp_i2_ss = atom_in_cell[:,temp_i2_uc-1]
        temp_i2_cell = ws_cell[count]

        #print(count)
        #print(temp_i1_ss)
        #print(temp_i2_ss)
        #print(temp_i2_cell)
        pairs_i1 = np.where(np.isin(atom_pair[:,0],temp_i1_ss))[0]
        pairs_i2 = np.where(np.isin(atom_pair[:,1],temp_i2_ss))[0]
        fc_pair_common = np.intersect1d(pairs_i1, pairs_i2)
        #print(fc_pair_common)
        for k in fc_pair_common:
            if (atom_pair_cell[k] == ws_cell[count]).all():
                #print(k)
                #print(atom_pair[k])
                #print(atom_pair_cell[k])
                for m in range(3):
                    print('{:>20.12f}  {:>20.12f}  {:>20.12f}'.format(fc_matrix[k][m][0],fc_matrix[k][m][1],fc_matrix[k][m][2]), file=fid)

        #print(temp_i2_cell)

        #print(np.shape(fc_pair_common))

        count += 1


        # Convert the atom index to the format of cell address in primitive cell
        #temp = i * Nsatom + j
        #a0 = atom_pair[temp, 0]
        #a1 = atom_pair[temp, 1]
        #print("a0=" + str(a0) + "  " + "a1=" + str(a1))
        #a2 = atom_pair[i, 2]
        #cell_add0 =  np.argwhere(atom_in_cell == a0)[0]
        #print( "i=" + str(i) + "  " + "j=" + str(j) + "  " + "temp=" + str(temp))
        #print(cell_add0)
        #cell_add1 =  np.argwhere(atom_in_cell == a1)[0]
        #print(cell_add1)
        #cell_add2 =  np.argwhere(atom_in_cell == a2)[0]
        #latt1 = np.dot(re_cell_address[cell_add1[0], :], prim_lattice)
        #temp_cell = re_cell_address[cell_add1[0], :]
        #latt2 = np.dot(re_cell_address[cell_add2[0], :], prim_lattice)
        #print('{:>.8f} {:>.8f} {:>.8f}'.format(latt2[0], latt2[1], latt2[2]), file=fid)
        #print('{:5d} {:5d} {:5d}'.format(atom_pair[i][0], int(atom_index_in_cells[0][cell_add1[1]]), int(atom_index_in_cells[0][cell_add2[1]])), file=fid)
        #print('      {:>10d}       In the unit cell, what is the index of neighbour  {:d} of atom  {:d}'.format(cell_add1[1]+1,int(j+1),cell_add0[1] + 1), file=fid)
        #print('{:>20.12f}  {:>20.12f}  {:>20.12f}'.format(temp_cell[0], temp_cell[1], temp_cell[2]), file=fid)
        
        #print('{:5d} {:5d} {:5d}'.format(atom_pair[i][0], int(atom_index_in_cells[0][cell_add1[1]]), int(atom_index_in_cells[0][cell_add2[1]])), file=fid)
        #print('{:5d} {:5d} {:5d}'.format(cell_add0[1] + 1, cell_add1[1] + 1), file=fid)


        #print(' ', file=fid)


fid.close
#print("3rd-order force constant has been finished")
#print(atom_in_cell)
