"""
This script is used to convert force constant in format of xml to that of dyn,
which is produced by ph.x in qe. It is written by XIANG Xing (mexingx@uset.hk).
"""

import xml.etree.ElementTree as ET
import numpy as np

def extract_force_constants(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    N_type = int(root.find('GEOMETRY_INFO/NUMBER_OF_TYPES').text)
    #print("N_type:", N_type)
    N_atoms = int(root.find('GEOMETRY_INFO/NUMBER_OF_ATOMS').text)
    #print("N_atoms:", N_atoms)
    BL_index = int(root.find('GEOMETRY_INFO/BRAVAIS_LATTICE_INDEX').text)
    #print("BL_index:", BL_index)
    cell_parameters = root.find('GEOMETRY_INFO/CELL_DIMENSIONS').text.strip().split()
    cell_parameters = [float(param) for param in cell_parameters]
    cell_parameters = np.array(cell_parameters)
    #print("Cell Parameters:", cell_parameters)
    lattice_vectors = root.find('GEOMETRY_INFO/AT').text.strip().split()
    lattice_vectors = [float(param) for param in lattice_vectors]
    lattice_vectors = np.array(lattice_vectors).reshape((3, 3))
    #print("Lattice Vectors:\n", lattice_vectors)

    masses = []
    for i in range(N_type):
        mass = float(root.find(f'GEOMETRY_INFO/MASS.{i+1}').text.strip())
        masses.append(mass)
    masses = np.array(masses) * 115666.348606255 / 126.904470000000
    #print("Masses (in atomic units):", masses)

    atomic_pos = []
    atom_index = []
    for i in range(N_atoms):
        temp_data = root.find(f'GEOMETRY_INFO/ATOM.{i+1}')
        temp_pos = temp_data.attrib['TAU'].strip().split()
        temp_pos = [float(p) for p in temp_pos]
        temp_pos = np.array(temp_pos)
        atomic_pos.append(temp_pos)

        temp_index = int(temp_data.attrib['INDEX'].strip()) 
        atom_index.append(temp_index)
    atomic_pos = np.array(atomic_pos)
    atom_index = np.array(atom_index)

    N_qpoint = int(root.find('GEOMETRY_INFO/NUMBER_OF_Q').text.strip())

    geometry = {"positions": atomic_pos, "indices": atom_index,"masses": masses, 
             "cell_parameters": cell_parameters, "lattice_vectors": lattice_vectors,
             "N_type": N_type, "N_atoms": N_atoms, "BL_index": BL_index, "N_qpoint": N_qpoint}
    print("Number of qpoints:", N_qpoint)

    try:
        epsilon = root.find('DIELECTRIC_PROPERTIES/EPSILON').text.strip().split()
        epsilon = [float(param) for param in epsilon]
        epsilon = np.array(epsilon).reshape((3, 3))
        print("Epsilon:", epsilon)
        
        charges_xt = root.find('DIELECTRIC_PROPERTIES/ZSTAR')
        charges = []
        for i in range(N_atoms):
            charge = charges_xt.find(f'Z_AT_.{i+1}').text.strip().split()
            charge = [float(param) for param in charge]
            charge = np.array(charge).reshape((3,3))
            charges.append(charge)

        #charges.append(charge)
    #print("Charges:", np.shape(charges))

        dielectric_info = {"epsilon": epsilon, "charges": charges}
    except:
        print("Dielectric properties not found in the XML file.")
        dielectric_info = None

    qpoint = []
    for i in range(N_qpoint):
        temp_qpoint = root.find(f'DYNAMICAL_MAT_.{i+1}/Q_POINT').text.strip().split()
        temp_qpoint = [float(param) for param in temp_qpoint]
        qpoint.append(temp_qpoint)
    qpoint = np.array(qpoint)
#    print(qpoint)

    phi = []
    for q in range(N_qpoint):
        phi_q = []
        for i in range(N_atoms):
            phi_i = []
            for j in range(N_atoms):
                phi_ij = root.find(f'DYNAMICAL_MAT_.{q+1}/PHI.{i+1}.{j+1}').text.strip().split()
                phi_ij = [float(param) for param in phi_ij]
                phi_lk = []
                for l in range(int(len(phi_ij)/2)):
                    temp_phi = complex(phi_ij[2*l],phi_ij[2*l+1])
                    phi_lk.append(temp_phi)
                phi_lk = np.array(phi_lk).reshape((3,-1))
                phi_ij = np.transpose(phi_lk)
                phi_i.append(phi_ij)
            phi_q.append(phi_i)
        phi.append(phi_q)
    phi = np.array(phi)
    
    print("Phi shape:", np.shape(phi))

    mode = root.find('FREQUENCIES_THZ_CMM1')
    N_modes = 3 * N_atoms
    freq_THz = []
    freq_cm = []
    for i in range(N_modes):
        freq = mode.find(f'OMEGA.{i+1}').text.strip().split()
#        freq = [float(param) for param in freq]
        freq_THz.append(float(freq[0]))
        freq_cm.append(float(freq[1]))
#    print("Frequencies (THz):", freq_THz)
#    print("Frequencies (cm-1):", freq_cm)

    disp = []
    for i in range(N_modes):
        disp_i = mode.find(f'DISPLACEMENT.{i+1}').text.strip().split()
        disp_i = [float(param) for param in disp_i]
        disp_lk = []
        for l in range(int(len(disp_i)/2)):
                temp_disp = complex(disp_i[2*l],disp_i[2*l+1])
                disp_lk.append(temp_disp)
        disp_i = np.array(disp_lk).reshape((N_atoms, 3))
#        print(f"Displacement for mode {i+1}:\n", disp_i)
        disp.append(disp_i)
    disp = np.array(disp)

    mode_info = {"qpoint": qpoint, "phi": phi, "frequencies_THz": freq_THz, "frequencies_cm": freq_cm, "displacements": disp}

    return geometry, dielectric_info, mode_info

def write_fc_dyn(geometry, dielectric_info, mode_info, output_file, atom_elements, out_dielectric=False):
    # out fc file in format of dyn file

    with open(output_file, 'w') as f:
        f.write("Dynamical matrix file\n")
        f.write("--                                                                         \n")
        f.write(f"  {geometry['N_type']}    {geometry['N_atoms']}   {geometry['BL_index']}")
        for i in geometry['cell_parameters']:
            f.write(f"   {i:.7f}")
        f.write("\n")
        f.write("Basis vectors\n")
        for i in range(3):
            f.write(f"  {geometry['lattice_vectors'][i,0]:>15.9f}{geometry['lattice_vectors'][i,1]:>15.9f}{geometry['lattice_vectors'][i,2]:>15.9f}\n")
        for i in range(geometry['N_type']):
            f.write(f"           {i+1}  '{atom_elements[i]}      '    {geometry['masses'][i]:16f}\n")
        for i in range(geometry['N_atoms']):
            f.write(f"{i+1:>5}{geometry['indices'][i]:>5}{geometry['positions'][i,0]:>18.10f}{geometry['positions'][i,1]:>18.10f}{geometry['positions'][i,2]:>18.10f}\n")
        f.write("\n")

        for q in range(geometry['N_qpoint']):
            f.write("     Dynamical  Matrix in cartesian axes\n")
            f.write("\n")
            f.write(f"     q = ( {mode_info['qpoint'][q,0]:>14.9f}{mode_info['qpoint'][q,1]:>14.9f}{mode_info['qpoint'][q,2]:>14.9f} )\n")
            f.write("\n")
            for i in range(geometry['N_atoms']):
                for j in range(geometry['N_atoms']):
                    f.write(f"{i+1:>5}{j+1:>5}")
                    f.write("\n")
                    for k in range(3):
                        for l in range(3):
                            f.write(f"{mode_info['phi'][q,i,j,k,l].real:>12.8f}{mode_info['phi'][q,i,j,k,l].imag:>13.8f}   ")
                        f.write("\n")
                #f.write("\n")
            f.write("\n")

        if out_dielectric:
            f.write("Dielectric Tensor:\n")
        
        f.write(f"     Diagonalizing the dynamical matrix\n")
        f.write("\n")

        f.write(f"     q = ( {mode_info['qpoint'][0,0]:>14.9f}{mode_info['qpoint'][0,1]:>14.9f}{mode_info['qpoint'][0,2]:>14.9f} )\n")
        f.write("\n")
        f.write(f" **************************************************************************\n")
        for i in range(len(mode_info['frequencies_THz'])):
            f.write(f"     freq ({i+1:>5}) ={mode_info['frequencies_THz'][i]:>15.6f} [THz] ={mode_info['frequencies_cm'][i]:>15.6f} [cm-1]\n")
            for j in range(geometry['N_atoms']):
                f.write(f" ({mode_info['displacements'][i,j,0].real:>10.6f}{mode_info['displacements'][i,j,0].imag:>10.6f}{mode_info['displacements'][i,j,1].real:>10.6f}{mode_info['displacements'][i,j,1].imag:>10.6f}{mode_info['displacements'][i,j,2].real:>10.6f}{mode_info['displacements'][i,j,2].imag:>10.6f} )\n")
#                for k in range(6):
#                    f.write(f"{mode_info['displacements'][i,j,k]:>12.8f}   ")
#                f.write("\n")
#            f.write("\n")
        f.write(f" **************************************************************************\n")


#        f.write(f"Number of atoms: {geometry['N_atoms']}\n")
#        f.write(f"Number of modes: {len(mode_info['frequencies_THz'])}\n")



if __name__ == "__main__":
#    xml_file = "../NbOI2.dyn1.xml"  # 替换为你的xml文件路径
#    output_file = "force_constants.txt"
#    geometry, dielectric_info, mode_info =extract_force_constants(xml_file, output_file)
    atom_elements = ["I", "Nb", "O"]  # elements for atoms
#    write_fc_dyn(geometry, dielectric_info, mode_info, output_file, atom_elements, out_dielectric=False)

    for i in range(6):
        xml_file = f"../NbOI2.dyn{i+1}.xml"
        output_file = f"NbOI2.dyn{i+1}"
        print(f"Processing {xml_file}...")
        geometry, dielectric_info, mode_info =extract_force_constants(xml_file)
        write_fc_dyn(geometry, dielectric_info, mode_info, output_file, atom_elements, out_dielectric=False)
