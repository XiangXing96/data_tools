'''
This script is written to transform the 2nd-order force constant file from alamode 
into that in shengBTE format. .fcs and model.txt are required for this script. The 
file of .fcs is produced by alamode. Model.txt is a part of input file of alamode. 
This is written by XIANG Xing from HKUST (xxiangad@connect.ust.hk)
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
file = 'HfO333_Monoclinic_3rd_fc.xml'               # force constant file in the format of .xml from alamode
Npatom = 12                                         # the number of atoms in primitive cell 
Nx = 3                                              # scell along x direction
Ny = 3                                              # scell along y direction
Nz = 3                                              # scell along z direction
scell = [Nx, Ny, Nz]
Total_cell =  Nx * Ny * Nz
Nsatom = int(Npatom * Total_cell)                     # the number of atoms in super cell

# Obtain model structure
fid = open(file)
lines = deque(fid.readlines())

prim_lattice = np.zeros([3,3])                      # the lattice vector of prmitive cell
atom_pos = np.zeros([Nsatom,3])                     # the position of atoms in supercell
index = np.zeros([Nsatom,1])                        # atomic index in supercell
atom_type = np.zeros([Nsatom,1])                    # atomic type in supercell, 1 for Hf, 2 for O
atom_in_cell = np.zeros([Total_cell, Npatom])       # classify atoms into cells
cell_address = np.zeros([Total_cell,3])


while len(lines) > 0:
    line = lines.popleft()

    if "<LatticeVector>" in line:
        for i in range(3):
            line = lines.popleft()
            temp_x = float(line.split()[1])
            temp_y = float(line.split()[2])
            temp_z = line.split()[3]
            temp_z = float(temp_z[:-5])
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
                pairs_of_atoms[i, j, 1] = atom_in_cell[cell_index_temp, cell_address2[1]]

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

        # Fill fc_matrix with data from other cells
        for i in range(1, Total_cell):
            for j in range(Npatom * Nsatom):
                line_index_temp_1st = np.where((atom_pair == pairs_of_atoms[0, j, :]).all(axis=1))[0]                             # obtain the position of pairs with 1st atom in 1st cell in global matrix
                line_index_temp_1st = line_index_temp_1st[0]
                line_index_temp_new = np.where((atom_pair == pairs_of_atoms[i, j, :]).all(axis=1))[0]                              # obtain the position of pairs with 1st atom in ith cell in global matrix
                line_index_temp_new = line_index_temp_new[0]

                fc_matrix[line_index_temp_new, :, :] = fc_matrix[line_index_temp_1st, :, :]

        print("Begin to write 2nd force constant")

        # write 2nd-order force constant in the format of shengBTE
        fid = open('FORCE_CONSTANTS_2ND', 'wt')
        print('{:6.0f} {:6.0f}'.format(Nsatom, Nsatom), file=fid)
        for i in range(Nsatom*Nsatom):
            print('{:3.0f} {:3.0f}'.format(atom_pair[i][0], atom_pair[i][1]), file=fid)
            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][0][0],fc_matrix[i][0][1],fc_matrix[i][0][2]), file=fid)
            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][1][0],fc_matrix[i][1][1],fc_matrix[i][1][2]), file=fid)
            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][2][0],fc_matrix[i][2][1],fc_matrix[i][2][2]), file=fid)
        fid.close

        print("2nd-order force constant has been finished")
  
    if "<ANHARM3>" in line:
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

        # Read 3rd-order force constant file to construct list to store atom_pair and force
        atom_pair_file = []
        direction_file = []
        fc_file = []

        print("Begin to read 3rd-order force constant.")

    if "FC3 pair1=" in line:
        atom1_index = line.split()[1]
        atom1_index = int(atom1_index[7:])
        atom1_index = atom_in_cell[0,atom1_index-1]   # the atom must be in the primitive cell
        atom1_direc = line.split()[2]                 # 1 for x, 2 for y, 3 for z
        atom1_direc = int(atom1_direc[:-1])

        atom2_index = line.split()[3]
        atom2_index = int(atom2_index[7:])
        atom2_direc = int(line.split()[4])

        atom3_index = line.split()[6]
        atom3_index = int(atom3_index[7:])
        atom3_direc = int(line.split()[7])

        fc_temp = line.split('>')[1]
        fc_temp = float(fc_temp[:-5])

        atom_pair_file.append([atom1_index, atom2_index, atom3_index])
        direction_file.append([atom1_direc, atom2_direc, atom3_direc])
        fc_file.append([fc_temp])

        # exchange the position of the last two atoms
        #if atom2_index != atom3_index:
        atom_pair_file.append([atom1_index, atom3_index, atom2_index])
        direction_file.append([atom1_direc, atom3_direc, atom2_direc])
        fc_file.append([fc_temp])

    if "</ANHARM3>" in line:
        atom_pair = list(set([tuple(t) for t in atom_pair_file]))
        atom_pair = [list(v) for v in atom_pair]
        atom_pair = np.array(atom_pair)
        
        Npair = np.shape(atom_pair)[0]
        # sort the atom_pair according to the cell address first and then atomic order
        refer_order = atom_in_cell.flatten()
        refer1 = []
        refer2 = []
        for i in range(Npair):
            temp1 = atom_pair[i, 1]
            temp2 = atom_pair[i, 2]
            order1 = np.where(refer_order == temp1)[0]
            order2 = np.where(refer_order == temp2)[0]
            refer1.append([order1[0]])
            refer2.append([order2[0]])
        atom_pair_order = np.append(atom_pair, refer1, axis=1)
        atom_pair_order = np.append(atom_pair_order, refer2, axis=1)
        atom_pair_order = sorted(atom_pair_order, key=(lambda x:[x[0],x[3],x[4]]))
        atom_pair_order = np.array(atom_pair_order)
        atom_pair = atom_pair_order[:, :3]

        fc_matrix = np.zeros((Npair, 3,3,3))
        for i in range(np.shape(atom_pair_file)[0]):
            index_in_ap = np.where((atom_pair == (atom_pair_file[i][0], atom_pair_file[i][1], atom_pair_file[i][2])).all(axis=1))[0]
            index_in_ap = index_in_ap[0]
            fc_matrix[index_in_ap, direction_file[i][0]-1, direction_file[i][1]-1, direction_file[i][2]-1] = fc_file[i][0]
        fc_matrix = fc_matrix * unit_scale_3rd # change unit

        print('Begin to write 3rd-order force constant file')

        fid = open('FORCE_CONSTANTS_3RD', 'wt')
        print(Npair, file=fid)
        print(' ', file=fid)
        for i in range(Npair):
            print('{:10d}'.format(i+1), file=fid)
            # Convert the atom index to the format of cell address in primitive cell
            a0 = atom_pair[i, 0]
            a1 = atom_pair[i, 1]
            a2 = atom_pair[i, 2]
            cell_add0 =  np.argwhere(atom_in_cell == a0)[0]
            cell_add1 =  np.argwhere(atom_in_cell == a1)[0]
            cell_add2 =  np.argwhere(atom_in_cell == a2)[0]
            latt1 = np.dot(re_cell_address[cell_add1[0], :], prim_lattice)
            latt2 = np.dot(re_cell_address[cell_add2[0], :], prim_lattice)
            print('{:>.8f} {:>.8f} {:>.8f}'.format(latt1[0], latt1[1], latt1[2]), file=fid)
            print('{:>.8f} {:>.8f} {:>.8f}'.format(latt2[0], latt2[1], latt2[2]), file=fid)
            #print('{:5d} {:5d} {:5d}'.format(atom_pair[i][0], int(atom_index_in_cells[0][cell_add1[1]]), int(atom_index_in_cells[0][cell_add2[1]])), file=fid)
            print('{:5d} {:5d} {:5d}'.format(cell_add0[1] + 1, cell_add1[1] + 1, cell_add2[1] + 1), file=fid)

            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        print('{:>5d} {:>5d} {:>5d} {:>20.10e}'.format(j+1, k+1, m+1, fc_matrix[i][j][k][m]), file=fid)
            print(' ', file=fid)
        fid.close
        print("3rd-order force constant has been finished")

    if "<ANHARM4>" in line:
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

        # Read 3rd-order force constant file to construct list to store atom_pair and force
        atom_pair_file = []
        direction_file = []
        fc_file = []

        print("Begin to read 4th-order force constant.")

    if "FC4 pair1=" in line:
        atom1_index = line.split()[1]
        atom1_index = int(atom1_index[7:])
        atom1_index = atom_in_cell[0,atom1_index-1]   # the atom must be in the primitive cell
        atom1_direc = line.split()[2]                 # 1 for x, 2 for y, 3 for z
        atom1_direc = int(atom1_direc[:-1])

        atom2_index = line.split()[3]
        atom2_index = int(atom2_index[7:])
        atom2_direc = int(line.split()[4])

        atom3_index = line.split()[6]
        atom3_index = int(atom3_index[7:])
        atom3_direc = int(line.split()[7])

        atom4_index = line.split()[9]
        atom4_index = int(atom4_index[7:])
        atom4_direc = int(line.split()[10])

        fc_temp = line.split('>')[1]
        fc_temp = float(fc_temp[:-5])

        atom_pair_file.append([atom1_index, atom2_index, atom3_index, atom4_index])
        direction_file.append([atom1_direc, atom2_direc, atom3_direc, atom4_direc])
        fc_file.append([fc_temp])

        # exchange the position of the last three atoms
        atom_pair_file.append([atom1_index, atom2_index, atom4_index, atom3_index])
        direction_file.append([atom1_direc, atom2_direc, atom4_direc, atom3_direc])
        fc_file.append([fc_temp])

        atom_pair_file.append([atom1_index, atom3_index, atom2_index, atom4_index])
        direction_file.append([atom1_direc, atom3_direc, atom2_direc, atom4_direc])
        fc_file.append([fc_temp])

        atom_pair_file.append([atom1_index, atom3_index, atom4_index, atom2_index])
        direction_file.append([atom1_direc, atom3_direc, atom4_direc, atom2_direc])
        fc_file.append([fc_temp])

        atom_pair_file.append([atom1_index, atom4_index, atom2_index, atom3_index])
        direction_file.append([atom1_direc, atom4_direc, atom2_direc, atom3_direc])
        fc_file.append([fc_temp])

        atom_pair_file.append([atom1_index, atom4_index, atom3_index, atom2_index])
        direction_file.append([atom1_direc, atom4_direc, atom3_direc, atom2_direc])
        fc_file.append([fc_temp])


    if "</ANHARM4>" in line:
        atom_pair = list(set([tuple(t) for t in atom_pair_file]))
        atom_pair = [list(v) for v in atom_pair]
        atom_pair = np.array(atom_pair)
        
        Npair = np.shape(atom_pair)[0]
        # sort the atom_pair according to the cell address first and then atomic order
        refer_order = atom_in_cell.flatten()
        refer1 = []
        refer2 = []
        refer3 = []
        for i in range(Npair):
            temp1 = atom_pair[i, 1]
            temp2 = atom_pair[i, 2]
            temp3 = atom_pair[i, 3]
            order1 = np.where(refer_order == temp1)[0]
            order2 = np.where(refer_order == temp2)[0]
            order3 = np.where(refer_order == temp3)[0]
            refer1.append([order1[0]])
            refer2.append([order2[0]])
            refer3.append([order3[0]])
        atom_pair_order = np.append(atom_pair, refer1, axis=1)
        atom_pair_order = np.append(atom_pair_order, refer2, axis=1)
        atom_pair_order = np.append(atom_pair_order, refer3, axis=1)
        atom_pair_order = sorted(atom_pair_order, key=(lambda x:[x[0],x[4],x[5],x[6]]))
        atom_pair_order = np.array(atom_pair_order)
        atom_pair = atom_pair_order[:, :4]

        fc_matrix = np.zeros((Npair, 3,3,3,3))
        for i in range(np.shape(atom_pair_file)[0]):
            index_in_ap = np.where((atom_pair == (atom_pair_file[i][0], atom_pair_file[i][1], atom_pair_file[i][2], atom_pair_file[i][3])).all(axis=1))[0]
            index_in_ap = index_in_ap[0]
            fc_matrix[index_in_ap, direction_file[i][0]-1, direction_file[i][1]-1, direction_file[i][2]-1, direction_file[i][3]-1] = fc_file[i][0]
        fc_matrix = fc_matrix * unit_scale_4th # change unit

        print('Begin to write 4th-order force constant file')

        fid = open('FORCE_CONSTANTS_4TH', 'wt')
        print(Npair, file=fid)
        print(' ', file=fid)
        for i in range(Npair):
            print('{:10d}'.format(i+1), file=fid)
            # Convert the atom index to the format of cell address in primitive cell
            a0 = atom_pair[i, 0]
            a1 = atom_pair[i, 1]
            a2 = atom_pair[i, 2]
            a3 = atom_pair[i, 3]
            cell_add0 =  np.argwhere(atom_in_cell == a0)[0]
            cell_add1 =  np.argwhere(atom_in_cell == a1)[0]
            cell_add2 =  np.argwhere(atom_in_cell == a2)[0]
            cell_add3 =  np.argwhere(atom_in_cell == a3)[0]
            latt1 = np.dot(re_cell_address[cell_add1[0], :], prim_lattice)
            latt2 = np.dot(re_cell_address[cell_add2[0], :], prim_lattice)
            latt3 = np.dot(re_cell_address[cell_add3[0], :], prim_lattice)
            print('{:>.8f} {:>.8f} {:>.8f}'.format(latt1[0], latt1[1], latt1[2]), file=fid)
            print('{:>.8f} {:>.8f} {:>.8f}'.format(latt2[0], latt2[1], latt2[2]), file=fid)
            print('{:>.8f} {:>.8f} {:>.8f}'.format(latt3[0], latt3[1], latt3[2]), file=fid)
            #print('{:5d} {:5d} {:5d}'.format(atom_pair[i][0], int(atom_index_in_cells[0][cell_add1[1]]), int(atom_index_in_cells[0][cell_add2[1]])), file=fid)
            print('{:5d} {:5d} {:5d} {:5d}'.format(cell_add0[1] + 1, cell_add1[1] + 1, cell_add2[1] + 1, cell_add3[1] + 1), file=fid)

            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        for n in range(3):
                            print('{:>5d} {:>5d} {:>5d} {:>5d} {:>20.10e}'.format(j+1, k+1, m+1, n+1, fc_matrix[i][j][k][m][n]), file=fid)
            print(' ', file=fid)
        fid.close
        print("4th-order force constant has been finished")
