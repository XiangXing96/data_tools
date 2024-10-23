'''
This script is written to reconstruct the 2nd-order force constant file from dfc2 
as the input of alm. Uncorrected and corrected 2nd-order force constant files are
required for the script. The corrected file is produced based on scph theory. And
uncorrected file is used to be the format reference for final output.
This is written by XIANG Xing from HKUST (xxiangad@connect.ust.hk).
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

import xml.etree.ElementTree as ET
import xml.dom.minidom

Born2A = 1 #0.52917721092
unit_scale_2nd = 1 # 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 2)
unit_scale_3rd = 1 # 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 3)
unit_scale_4th = 1 # 4.35974394e-18 / 2.0 / 1.6021766208e-19 / pow(0.52917721092, 4)

# Set parameters
file_uncorrected = 'uncorrected_harmonic.xml'       # uncorrected force constant file in the format of .xml from alamode
file_corrected = 'corrected_150K.xml'               # corrected force constant file in the format of .xml from alamode
output_file = "reconstructed.xml"                   # the reconstructed force constant
Npatom = 10                                         # the number of atoms in primitive cell 
Nx = 2                                              # scell along x direction
Ny = 2                                              # scell along y direction
Nz = 2                                              # scell along z direction
scell = [Nx, Ny, Nz]
Total_cell =  Nx * Ny * Nz
Nsatom = int(Npatom * Total_cell)                     # the number of atoms in super cell


### read corrected 2nd-order force constant 
# Obtain model structure

prim_lattice = np.zeros([3,3])                      # the lattice vector of prmitive cell
atom_pos = np.zeros([Nsatom,3])                     # the position of atoms in supercell
index = np.zeros([Nsatom,1])                        # atomic index in supercell
atom_type = np.zeros([Nsatom,1])                    # atomic type in supercell, 1 for Hf, 2 for O
atom_in_cell = np.zeros([Total_cell, Npatom])       # classify atoms into cells
cell_address = np.zeros([Total_cell,3])


fid = open(file_corrected)
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

        print("Read corrected 2nd-order force constant")
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

        print("Finish reading Corrected 2nd force constant")
        fid.close()

#        print("Begin to write 2nd force constant")

        # write 2nd-order force constant in the format of shengBTE
#        fid = open('FORCE_CONSTANTS_2ND', 'wt')
#        print(Nsatom, file=fid)
#        for i in range(Nsatom*Nsatom):
#            print('{:3.0f} {:3.0f}'.format(atom_pair[i][0], atom_pair[i][1]), file=fid)
#            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][0][0],fc_matrix[i][0][1],fc_matrix[i][0][2]), file=fid)
#            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][1][0],fc_matrix[i][1][1],fc_matrix[i][1][2]), file=fid)
#            print('{:>15.8f} {:>15.8f} {:>15.8f}'.format(fc_matrix[i][2][0],fc_matrix[i][2][1],fc_matrix[i][2][2]), file=fid)
#        fid.close

#        print("2nd-order force constant has been finished")



### read uncorrected 2nd-order force constant file 
# Obtain model structure

version = 0                                         # alm version to produce uncorrected fc file
atom_totalN = 0                                     # number of atoms in supercell
element_N = 0                                       # number of elements in system
elements = []                                       # element kinds in system
atomic_element = []                                 # element of each atom
Translations_N = 0                                  # number of translation
Translations = []                                   # store translation information
HarmonicUnique_N = 0                                # store number of unique force constants
HarmonicUnique = []                                 # store unique force constants
Harmonic = []                                       # store information fc2 in uncorrected file

Harmonic_fc = []                                    # store harmonic force constant divided by multi
FC_Harmonic_force = []                              # store harmonic force constant divided by multi, just force 

HarmonicUnique2map = []                             # find corresponse between HarmonicUnique and Harmonic_fc

prim_lattice = np.zeros([3,3])                      # the lattice vector of prmitive cell
atom_pos = np.zeros([Nsatom,3])                     # the position of atoms in supercell

#index = np.zeros([Nsatom,1])                        # atomic index in supercell
#atom_type = np.zeros([Nsatom,1])                    # atomic type in supercell, 1 for Hf, 2 for O
#atom_in_cell = np.zeros([Total_cell, Npatom])       # classify atoms into cells
#cell_address = np.zeros([Total_cell,3])


fid = open(file_uncorrected)
lines = deque(fid.readlines())

while len(lines) > 0:
    line = lines.popleft()

    if "<ALM_version>" in line:
        temp = line[15:]
        temp = temp[:-15]
        version = temp
    
    if "<NumberOfAtoms>" in line:
        temp = line[19:]
        temp = temp[:-17]
        atom_totalN = int(temp)

    if "<NumberOfElements>" in line:
        temp = line[22:]
        temp = temp[:-20]
        element_N = int(temp)

    if "<AtomicElements>" in line:
        for i in range(element_N):
            line = lines.popleft()
            temp = line.split(">")[1]
            temp = temp[:-9]
            elements.append(temp)
            
    if "<LatticeVector>" in line:
        for i in range(3):
            line = lines.popleft()
            temp_x = float(line.split()[1])
            temp_y = float(line.split()[2])
            temp_z = line.split()[3]
            temp_z = float(temp_z[:-5])
            prim_lattice[i,0] = temp_x
            prim_lattice[i,1] = temp_y
            prim_lattice[i,2] = temp_z

    if "<Position>" in line:
        for i in range(atom_totalN):
            line = lines.popleft()
            #print(line)
            #temp_index = line.split()[1]
            #temp_index = int(temp_index.split("\"")[1])
            #index[i] = temp_index

            temp_element = line.split()[2]
            temp_element = temp_element.split("\"")[1]
            atomic_element.append(temp_element)
            
            temp_x = line.split()[3]
            temp_y = line.split()[4]
            temp_z = line.split()[5]
            temp_z = temp_z[:-6]
            atom_pos[i,0] = float(temp_x)
            atom_pos[i,1] = float(temp_y)
            atom_pos[i,2] = float(temp_z)
        #print(atomic_element)

    if "<NumberOfTranslations>" in line:
        temp = line[26:]
        temp = temp[:-24]
        Translations_N = int(temp)
        #print(Translations_N)

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
            temp = [temp_line, temp_colum, temp_index]
            Translations.append(temp)


    if "<HarmonicUnique>" in line:
        line = lines.popleft()
        temp = line[12:]
        temp = temp[:-8]
        HarmonicUnique_N = int(temp)
        line = lines.popleft()
        
        # check basis
        temp_basis = line[13:]
        temp_basis = temp_basis[:-9]
        if temp_basis != 'Cartesian':
            print('!!!!Error: Please use FCSYM_BASIS=Cartesian in alm to produce .xlm file.')
            sys.exit()

        for i in range(HarmonicUnique_N):
            line = lines.popleft()

            temp_p1 = line.split()[1]
            temp_p1 = temp_p1[7:]

            temp_p2 = line.split()[2]
            temp_p2 = temp_p2[:-1]

            temp_multi = line.split()[3]
            temp_multi = temp_multi.split('\"')[1]
            
            temp_value = line.split()[3]
            temp_value = temp_value.split('>')[1]
            temp_value = temp_value[:-5]

            #print(temp_value)

            temp = [temp_p1, temp_p2, temp_multi, temp_value]
            HarmonicUnique.append(temp)

    if "<HARMONIC>" in line:

        print("Read uncorrected 2nd-order force constant")
        FC_Harmonic = []

    if "FC2 pair1=" in line:
        atom1_index = line.split()[1]
        atom1_index = atom1_index[7:]
        #atom1_index = atom_in_cell[0,atom1_index-1]   # the atom must be in the primitive cell
        atom1_direc = line.split()[2]                 # 1 for x, 2 for y, 3 for z
        atom1_direc = atom1_direc[:-1]

        atom2_index = line.split()[3]
        atom2_index = atom2_index[7:]
        atom2_direc = line.split()[4]

        temp_operation = line.split()[5]
        temp_operation = temp_operation.split('\"')[0]

        fc_temp = line.split('>')[1]
        fc_temp = fc_temp[:-5]

        FC_Harmonic_force.append(float(fc_temp))

        temp = [atom1_index, atom1_direc, atom2_index, atom2_direc, temp_operation, fc_temp]

        FC_Harmonic.append(temp)

        #fc_index = np.where((atom_pair == (atom1_index, atom2_index)).all(axis=1))[0]
        #fc_index = fc_index[0]

        #fc_matrix[fc_index][atom1_direc-1][atom2_direc-1] += fc_temp * unit_scale_2nd


    if "</HARMONIC>" in line:

        fid.close()
        
        # reconstruct Harmonicunique
        # Find corresponse
        print("Reconstruct Harmonicunique")
        FC_Harmonic_force = np.array(FC_Harmonic_force)
        for i in HarmonicUnique:
            temp = float(i[3]) / float(i[2])
            #print(temp)
            pairs = np.where(np.abs(FC_Harmonic_force-temp) < 1e-8)[0]
            #print(pairs)
            temp_pair = pairs[0]

            HarmonicUnique2map.append(temp_pair)
        #print(HarmonicUnique2map)

        for i in range(HarmonicUnique_N):
            index_FC2 = HarmonicUnique2map[i]
            temp_a1 = int(FC_Harmonic[index_FC2][0])
            temp_a1_dir = int(FC_Harmonic[index_FC2][1])
            temp_a2 = int(FC_Harmonic[index_FC2][2])
            temp_a2_dir = int(FC_Harmonic[index_FC2][3])
            
            temp_fc_index = np.where((atom_pair == (temp_a1, temp_a2)).all(axis=1))[0]
            temp_fc_index = temp_fc_index[0]
            #print(temp_a2_dir)
            #print(np.shape(temp_fc_index))
            #print(np.shape(fc_matrix))

            HarmonicUnique[i][3] = str(fc_matrix[temp_fc_index,temp_a1_dir-1,temp_a2_dir-1])
        
        # reconstruct Harmonic_FC2
        print("Reconstruct Harmonic_FC2")
        fc2_multi = np.zeros_like(fc_matrix)
        #print(np.shape(fc2_multi))
        for i in FC_Harmonic:
            temp_a1 = int(i[0])
            temp_a1_dir = int(i[1])
            temp_a2 = int(i[2])
            temp_a2_dir = int(i[3])
            
            temp_fc_index = np.where((atom_pair == (temp_a1, temp_a2)).all(axis=1))[0]
            temp_fc_index = temp_fc_index[0]

            fc2_multi[temp_fc_index,temp_a1_dir-1,temp_a2_dir-1] += 1
    
        for i in range(np.shape(FC_Harmonic)[0]):
        #for i in FC_Harmonic:
            temp_fc_harmonic = FC_Harmonic[i]
            temp_a1 = int(temp_fc_harmonic[0])
            temp_a1_dir = int(temp_fc_harmonic[1])
            temp_a2 = int(temp_fc_harmonic[2])
            temp_a2_dir = int(temp_fc_harmonic[3])
            
            temp_fc_index = np.where((atom_pair == (temp_a1, temp_a2)).all(axis=1))[0]
            temp_fc_index = temp_fc_index[0]

            temp_multi = fc2_multi[temp_fc_index,temp_a1_dir-1,temp_a2_dir-1]
            temp_forfce = fc_matrix[temp_fc_index,temp_a1_dir-1,temp_a2_dir-1] / temp_multi
            FC_Harmonic[i][5] = str(temp_forfce)

#print(HarmonicUnique[1])
#print(FC_Harmonic[1])

print("write reconstructed 2nd-order force constant")

root = ET.Element("Data")

# add terms
item1 = ET.SubElement(root, "ALM_version")
#item1.set("name", "item1")
item1.text = "1.5.0"

item2 = ET.SubElement(root, "Optimize")
item2_1 = ET.SubElement(item2,"DFSET")
item2_1.text = "./DFSET_harmonic"
item2_2 = ET.SubElement(item2,"Constraint")
item2_2.text = "1"

item3 = ET.SubElement(root, "Structure")
item3_1 = ET.SubElement(item3,"NumberOfAtoms")
item3_1.text = str(atom_totalN)
item3_2 = ET.SubElement(item3,"NumberOfElements")
item3_2.text = str(element_N)
item3_3 = ET.SubElement(item3,"AtomicElements")
for i in range(element_N):
    item3_3_1 = ET.SubElement(item3_3,"element")
    item3_3_1.set("number",str(i+1))
    item3_3_1.text = elements[i]
item3_4 = ET.SubElement(item3,"LatticeVector")
for i in range(3):
    item3_4_1 = ET.SubElement(item3_4,"a"+str(i+1))
    item3_4_1.text = ' ' + f"{prim_lattice[i,0]:.15e}" + ' ' + f"{prim_lattice[i,1]:.15e}" + ' ' + f"{prim_lattice[i,2]:.15e}"
item3_5 = ET.SubElement(item3,"Periodicity")
item3_5.text = "1 1 1"
item3_6 = ET.SubElement(item3,"Position")
for i in range(atom_totalN):
    item3_6_1 = ET.SubElement(item3_6,"pos")
    item3_6_1.set("index",str(i+1))
    item3_6_1.set("element", atomic_element[i])
    #item3_6_1.set("index",str(i+1), "element", atomic_element[i])
    item3_6_1.text = ' ' + f"{atom_pos[i,0]:.15e}" + ' ' + f"{atom_pos[i,1]:.15e}" + ' ' + f"{atom_pos[i,2]:.15e}"

item4 = ET.SubElement(root, "Symmetry")
item4_1 = ET.SubElement(item4,"NumberOfTranslations")
item4_1.text = str(Translations_N)
item4_2 = ET.SubElement(item4,"Translations")
for i in Translations:
    item4_2_1 = ET.SubElement(item4_2,"map")
    item4_2_1.set("tran", str(i[0]))
    item4_2_1.set("atom", str(i[1]))
    item4_2_1.text = str(i[2])

item5 = ET.SubElement(root, "ForceConstants")
item5_1 = ET.SubElement(item5,"HarmonicUnique")
item5_1_1 = ET.SubElement(item5_1,"NFC2")
item5_1_1.text = str(HarmonicUnique_N)
item5_1_2 = ET.SubElement(item5_1,"Basis")
item5_1_2.text = "Cartesian"
for i in HarmonicUnique:
    item5_1_3 = ET.SubElement(item5_1,"FC2")
    item5_1_3.set("pairs", i[0] + ' ' + i[1] )
    item5_1_3.set("multiplicity", i[2])
    item5_1_3.text = f"{float(i[3]):.15e}"

item5_2 = ET.SubElement(item5,"HARMONIC")
for i in FC_Harmonic:
    item5_2_1 = ET.SubElement(item5_2,"FC2")
    item5_2_1.set("pair1", i[0] + ' ' + i[1])
    item5_2_1.set("pair2", i[2] + ' ' + i[3] + ' ' + i[4])
    item5_2_1.text = f"{float(i[5]):.15e}"

# create ElementTree object
xml_str = ET.tostring(root,encoding='utf-8')

dom = xml.dom.minidom.parseString(xml_str)
pretty_xml_as_string = dom.toprettyxml(indent="  ")

# write xml files
with open(output_file, "w", encoding='utf-8') as f:
    f.write(pretty_xml_as_string)

print("reconstructed XML has been finished!")

